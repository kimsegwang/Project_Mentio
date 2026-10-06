"""
server/services/weather_service.py
OpenWeatherMap 현재 날씨 API 연동 + TTL 캐시 + 룰 기반 날씨 발화 생성 서비스.

- LLM 호출 없이 정형 템플릿으로 발화를 만들어, 날씨 질의도 시간 질의처럼 즉답한다.
- 캐시(WEATHER_CACHE_TTL_SEC) 적중 시 네트워크 왕복이 없고, 미스 시에도 WEATHER_API_TIMEOUT_SEC로 블로킹 상한을 둔다.
- API 키 부재/호출 실패/응답 형식 이상 시에는 가짜 날씨를 지어내지 않고, 정보를 받아올 수 없다는
  자연스러운 안내 멘트로 폴백한다 (로봇이 침묵하거나 ERROR 표정을 띄우지 않도록).
- HTTP 호출부는 fetcher 콜러블로 주입받아(DIP), 테스트에서 네트워크 없이 검증하거나 향후 기상청 API로 교체할 수 있다.
"""
import json
import logging
import re
import threading
import time
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

from config import settings
from server.schemas.action import EmotionType, LLMResponse
from server.schemas.weather import WeatherSnapshot

logger = logging.getLogger(__name__)

OPENWEATHERMAP_URL = "https://api.openweathermap.org/data/2.5/weather"

# (city, api_key, timeout_sec) -> 원시 JSON dict
WeatherFetcher = Callable[[str, str, float], dict]

# API 질의용 영문 도시명 -> 발화용 한국어 표시명
CITY_DISPLAY_NAMES = {
    "seoul": "서울",
    "busan": "부산",
    "incheon": "인천",
    "daegu": "대구",
    "daejeon": "대전",
    "gwangju": "광주",
    "ulsan": "울산",
    "suwon": "수원",
    "sejong": "세종",
    "jeju": "제주",
}

RAIN_CONDITIONS = {"Rain", "Drizzle", "Thunderstorm"}
SNOW_CONDITIONS = {"Snow"}

WEATHER_UNAVAILABLE_SPEECH = (
    "지금은 날씨 정보를 받아올 수가 없어서 정확히는 모르겠어. 창밖을 한번 같이 볼까?"
)


def _josa(word: str, with_batchim: str, without_batchim: str) -> str:
    """마지막 글자의 받침 유무에 따라 조사를 붙인다 (예: 서울은/대구는, 맑음이고/안개고)."""
    last = word[-1] if word else ""
    if "가" <= last <= "힣":
        has_batchim = (ord(last) - ord("가")) % 28 != 0
    else:
        has_batchim = False
    return word + (with_batchim if has_batchim else without_batchim)


def fetch_openweathermap(city: str, api_key: str, timeout_sec: float) -> dict:
    query = urlencode({"q": city, "appid": api_key, "units": "metric", "lang": "kr"})
    with urlopen(f"{OPENWEATHERMAP_URL}?{query}", timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


class WeatherService:
    # 발화 세부 분기용 패턴 (날씨 인텐트 자체의 감지는 IntentService 담당)
    RAIN_QUESTION_PATTERN = re.compile(r"(비\s*(와|오|올|내리|내려)|우산)")
    SNOW_QUESTION_PATTERN = re.compile(r"눈\s*(와|오|올|내리|내려)")
    # 현재 날씨 API로는 답할 수 없는 예보성 질의
    FORECAST_PATTERN = re.compile(r"(내일|모레|주말|이번\s*주|다음\s*주)")

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_city: Optional[str] = None,
        cache_ttl_sec: Optional[float] = None,
        timeout_sec: Optional[float] = None,
        fetcher: WeatherFetcher = fetch_openweathermap,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.api_key = settings.WEATHER_API_KEY if api_key is None else api_key
        self.default_city = default_city or settings.WEATHER_DEFAULT_CITY
        self.cache_ttl_sec = settings.WEATHER_CACHE_TTL_SEC if cache_ttl_sec is None else cache_ttl_sec
        self.timeout_sec = settings.WEATHER_API_TIMEOUT_SEC if timeout_sec is None else timeout_sec
        self._fetcher = fetcher
        self._clock = clock
        self._cache: Dict[str, Tuple[float, WeatherSnapshot]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _display_city(city: str, api_name: Optional[str] = None) -> str:
        return CITY_DISPLAY_NAMES.get(city.strip().lower()) or api_name or city

    def _parse(self, city: str, data: dict) -> WeatherSnapshot:
        weather = data["weather"][0]
        main = data["main"]
        return WeatherSnapshot(
            city=self._display_city(city, data.get("name")),
            condition=weather.get("main", ""),
            description=weather.get("description", ""),
            temperature=float(main["temp"]),
            feels_like=main.get("feels_like"),
            humidity=main.get("humidity"),
        )

    def get_weather(self, city: Optional[str] = None) -> Optional[WeatherSnapshot]:
        """캐시가 신선하면 캐시를, 아니면 API를 호출해 갱신한다. 키 부재/실패 시 None."""
        city = city or self.default_city
        if not self.api_key:
            logger.warning("[WeatherService] WEATHER_API_KEY 미설정 -> 날씨 안내 폴백")
            return None

        cache_key = city.strip().lower()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and self._clock() - cached[0] < self.cache_ttl_sec:
                return cached[1]

            try:
                snapshot = self._parse(city, self._fetcher(city, self.api_key, self.timeout_sec))
            except Exception as e:
                logger.warning(f"[WeatherService] 날씨 조회 실패 ({city}) -> 폴백: {e}")
                return None

            self._cache[cache_key] = (self._clock(), snapshot)
            return snapshot

    @staticmethod
    def _format_temperature(temp: float) -> str:
        rounded = round(temp)
        return f"영하 {abs(rounded)}도" if rounded < 0 else f"{rounded}도"

    @staticmethod
    def _emotion_for(snapshot: WeatherSnapshot) -> EmotionType:
        if snapshot.condition in RAIN_CONDITIONS:
            return EmotionType.SAD
        if snapshot.condition in SNOW_CONDITIONS:
            return EmotionType.SURPRISED
        if snapshot.condition in {"Clear", "Clouds"}:
            return EmotionType.HAPPY
        return EmotionType.CURIOUS

    def _advice_for(self, snapshot: WeatherSnapshot) -> str:
        if snapshot.condition in RAIN_CONDITIONS:
            return " 나갈 때 우산 꼭 챙겨!"
        if snapshot.condition in SNOW_CONDITIONS:
            return " 길 미끄러우니까 조심해!"
        if snapshot.temperature <= 5:
            return " 따뜻하게 입고 나가!"
        if snapshot.temperature >= 28:
            return " 물 자주 마셔!"
        return ""

    def build_weather_response(self, text: str = "") -> LLMResponse:
        """최신(캐시된) 날씨로 LLM 없이 즉답 발화를 생성한다. 날씨를 모르면 안내 멘트로 폴백."""
        snapshot = self.get_weather()
        if snapshot is None:
            return LLMResponse(emotion=EmotionType.SAD, speech=WEATHER_UNAVAILABLE_SPEECH)

        text = text or ""
        prefix = ""
        if self.FORECAST_PATTERN.search(text):
            prefix = "예보까지는 아직 못 봐서, 지금 날씨만 알려줄게! "

        # 기온 문자열은 항상 '도'로 끝나므로 서술격 조사는 '야'로 고정
        temp = self._format_temperature(snapshot.temperature)
        city = snapshot.city
        desc_and = _josa(snapshot.description, "이고", "고")
        is_raining = snapshot.condition in RAIN_CONDITIONS
        is_snowing = snapshot.condition in SNOW_CONDITIONS

        if self.RAIN_QUESTION_PATTERN.search(text):
            if is_raining:
                body = f"응, 지금 {city}에 비가 오고 있어. 기온은 {temp}야. 우산 꼭 챙겨!"
            else:
                body = f"아니, 지금 {city}엔 비 안 와! {desc_and} 기온은 {temp}야."
        elif self.SNOW_QUESTION_PATTERN.search(text):
            if is_snowing:
                body = f"응, 지금 {city}에 눈이 오고 있어! 기온은 {temp}야. 길 미끄러우니까 조심해!"
            else:
                body = f"아니, 지금 {city}엔 눈 안 와. {desc_and} 기온은 {temp}야."
        else:
            body = f"지금 {_josa(city, '은', '는')} {desc_and}, 기온은 {temp}야."
            if snapshot.feels_like is not None and abs(snapshot.feels_like - snapshot.temperature) >= 3:
                body += f" 체감 온도는 {self._format_temperature(snapshot.feels_like)}야."
            body += self._advice_for(snapshot)

        return LLMResponse(emotion=self._emotion_for(snapshot), speech=prefix + body)


weather_service = WeatherService()
