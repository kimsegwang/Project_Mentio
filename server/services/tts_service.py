# server/services/tts_service.py
import asyncio
import io
import logging
import edge_tts
from config.settings import TTS_VOICE_NAME, TTS_RATE, TTS_PITCH

logger = logging.getLogger("TTSService")


class TTSService:
    def __init__(
        self,
        voice: str = TTS_VOICE_NAME,
        rate: str = TTS_RATE,
        pitch: str = TTS_PITCH
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def _synthesize_stream(self, text: str) -> bytes:
        clean_text = text.strip() if text else ""
        if not clean_text:
            logger.warning("합성할 텍스트가 비어 있습니다.")
            return b""

        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])

        buffer.seek(0)
        return buffer.read()

    def synthesize(self, text: str) -> bytes:
        return asyncio.run(self._synthesize_stream(text))