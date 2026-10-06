from typing import Optional

from pydantic import BaseModel, Field


class WeatherSnapshot(BaseModel):
    """
    외부 날씨 API(OpenWeatherMap) 응답에서 발화 생성에 필요한 필드만 정규화한 DTO.
    """
    city: str = Field(description="발화에 사용할 도시 표시명 (예: 서울)")
    condition: str = Field(description="API 기상 분류 키 (Clear, Clouds, Rain, Snow, Drizzle, Thunderstorm 등)")
    description: str = Field(description="한국어 날씨 설명 (예: 맑음, 실 비)")
    temperature: float = Field(description="현재 기온 (섭씨)")
    feels_like: Optional[float] = Field(default=None, description="체감 기온 (섭씨)")
    humidity: Optional[int] = Field(default=None, description="상대 습도 (%)")
