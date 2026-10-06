"""
tests/test_weather_service.py
WeatherService(OpenWeatherMap 연동 + TTL 캐시 + 룰 기반 날씨 발화)를 네트워크 없이 검증하는 pytest 단위 테스트.

검증 대상:
- API 정상 응답 파싱 및 발화 생성 (일반/비/눈/예보 질의 분기, 한국어 조사)
- TTL 캐시: 만료 전 재호출 없음, 만료 후 재조회
- API 키 부재 / 호출 예외 / 응답 형식 이상 시 외부 호출 없이(또는 실패를 삼키고) 안내 멘트로 폴백
"""
from unittest.mock import Mock

import pytest

from server.schemas.action import EmotionType
from server.services.weather_service import WEATHER_UNAVAILABLE_SPEECH, WeatherService


def _owm_payload(main="Clear", description="맑음", temp=18.4, feels_like=17.0, humidity=40, name="Seoul"):
    return {
        "name": name,
        "weather": [{"main": main, "description": description}],
        "main": {"temp": temp, "feels_like": feels_like, "humidity": humidity},
    }


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _service(fetcher, clock=None, api_key="test-key", ttl=1800, city="Seoul") -> WeatherService:
    return WeatherService(
        api_key=api_key,
        default_city=city,
        cache_ttl_sec=ttl,
        timeout_sec=1.0,
        fetcher=fetcher,
        clock=clock or FakeClock(),
    )


# --- 정상 응답 파싱 ---

def test_get_weather_parses_openweathermap_payload():
    fetcher = Mock(return_value=_owm_payload())
    service = _service(fetcher)

    snapshot = service.get_weather()

    assert snapshot.city == "서울"
    assert snapshot.condition == "Clear"
    assert snapshot.description == "맑음"
    assert snapshot.temperature == pytest.approx(18.4)
    assert snapshot.humidity == 40
    fetcher.assert_called_once_with("Seoul", "test-key", 1.0)


def test_build_weather_response_general_clear():
    service = _service(Mock(return_value=_owm_payload()))

    response = service.build_weather_response("오늘 날씨 어때?")

    assert response.emotion == EmotionType.HAPPY
    assert response.speech == "지금 서울은 맑음이고, 기온은 18도야."


def test_build_weather_response_uses_josa_for_city_without_batchim():
    service = _service(Mock(return_value=_owm_payload(description="안개", main="Mist", name="Daegu")), city="Daegu")

    response = service.build_weather_response("날씨 알려줘")

    assert response.speech.startswith("지금 대구는 안개고, 기온은 18도야.")


def test_build_weather_response_mentions_feels_like_and_cold_advice():
    service = _service(Mock(return_value=_owm_payload(description="맑음", temp=-2.6, feels_like=-7.0)))

    response = service.build_weather_response("날씨 어때")

    assert "기온은 영하 3도야." in response.speech
    assert "체감 온도는 영하 7도야." in response.speech
    assert "따뜻하게 입고 나가!" in response.speech


def test_build_weather_response_rain_question_when_raining():
    service = _service(Mock(return_value=_owm_payload(main="Rain", description="실 비", temp=15)))

    response = service.build_weather_response("비 와?")

    assert response.emotion == EmotionType.SAD
    assert response.speech == "응, 지금 서울에 비가 오고 있어. 기온은 15도야. 우산 꼭 챙겨!"


def test_build_weather_response_rain_question_when_clear():
    service = _service(Mock(return_value=_owm_payload(description="맑음", temp=21)))

    response = service.build_weather_response("지금 비 오니")

    assert response.speech == "아니, 지금 서울엔 비 안 와! 맑음이고 기온은 21도야."


def test_build_weather_response_snow_question_when_snowing():
    service = _service(Mock(return_value=_owm_payload(main="Snow", description="눈", temp=-1)))

    response = service.build_weather_response("눈 와?")

    assert response.emotion == EmotionType.SURPRISED
    assert response.speech.startswith("응, 지금 서울에 눈이 오고 있어!")


def test_build_weather_response_forecast_question_is_honest_about_current_only():
    service = _service(Mock(return_value=_owm_payload()))

    response = service.build_weather_response("내일 날씨 어때?")

    assert response.speech.startswith("예보까지는 아직 못 봐서, 지금 날씨만 알려줄게! 지금 서울은")


# --- TTL 캐시 ---

def test_cache_hit_within_ttl_skips_api_call():
    fetcher = Mock(return_value=_owm_payload())
    clock = FakeClock(1000.0)
    service = _service(fetcher, clock=clock, ttl=1800)

    first = service.get_weather()
    clock.now += 1799
    second = service.get_weather()

    assert first == second
    assert fetcher.call_count == 1


def test_cache_expires_after_ttl_and_refetches():
    fetcher = Mock(side_effect=[_owm_payload(temp=10), _owm_payload(temp=12)])
    clock = FakeClock(1000.0)
    service = _service(fetcher, clock=clock, ttl=1800)

    first = service.get_weather()
    clock.now += 1800
    second = service.get_weather()

    assert fetcher.call_count == 2
    assert first.temperature == 10
    assert second.temperature == 12


def test_failed_fetch_is_not_cached():
    fetcher = Mock(side_effect=[TimeoutError("timeout"), _owm_payload()])
    service = _service(fetcher)

    assert service.get_weather() is None
    assert service.get_weather() is not None
    assert fetcher.call_count == 2


# --- 폴백: API 키 부재 / 호출 실패 / 응답 형식 이상 ---

def test_missing_api_key_falls_back_without_calling_api():
    fetcher = Mock()
    service = _service(fetcher, api_key="")

    response = service.build_weather_response("날씨 알려줘")

    fetcher.assert_not_called()
    assert response.speech == WEATHER_UNAVAILABLE_SPEECH
    assert response.emotion != EmotionType.ERROR


def test_api_exception_falls_back_to_guidance_speech():
    service = _service(Mock(side_effect=OSError("network down")))

    response = service.build_weather_response("오늘 날씨 어때?")

    assert response.speech == WEATHER_UNAVAILABLE_SPEECH
    assert response.emotion != EmotionType.ERROR


def test_malformed_payload_falls_back_to_guidance_speech():
    # OpenWeatherMap 인증 실패 응답 형태 (weather/main 키 없음)
    service = _service(Mock(return_value={"cod": 401, "message": "Invalid API key"}))

    response = service.build_weather_response("날씨")

    assert response.speech == WEATHER_UNAVAILABLE_SPEECH
