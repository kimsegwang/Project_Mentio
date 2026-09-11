# server/services/audio_player_service.py
import io
import time
import logging
import pygame
from config.settings import AUDIO_REVERB_GUARD_SEC

logger = logging.getLogger("AudioPlayerService")


class AudioPlayerService:
    def __init__(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=2048)
            logger.info("pygame.mixer 초기화 완료 (24kHz)")

    def play_bytes_and_wait(self, audio_bytes: bytes, reverb_guard_sec: float = AUDIO_REVERB_GUARD_SEC) -> None:
        if not audio_bytes:
            return

        try:
            audio_io = io.BytesIO(audio_bytes)
            pygame.mixer.music.load(audio_io)
            pygame.mixer.music.play()

            # 재생 완료 시점까지 대기
            while pygame.mixer.music.get_busy():
                time.sleep(0.02)

            # 재생 종료 후 잔향 가드 대기
            if reverb_guard_sec > 0:
                time.sleep(reverb_guard_sec)

        except Exception as e:
            logger.error(f"오디오 출력 에러: {e}", exc_info=True)