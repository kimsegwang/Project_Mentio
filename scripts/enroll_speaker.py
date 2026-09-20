"""
scripts/enroll_speaker.py
기준 화자(primary_user)의 목소리를 마이크로 녹음해 화자 임베딩(.npy)으로 등록하는 스크립트.

등록된 임베딩 파일은 SpeakerService.verify()가 기준값으로 사용한다.
등록 전(파일 없음) 상태에서는 SpeakerService가 화자 검증을 자동으로 스킵(통과)하므로,
이 스크립트를 실행하지 않아도 기존 파이프라인 동작에는 영향이 없다.

실행:
    python scripts/enroll_speaker.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sounddevice as sd

from config import settings
from server.services.speaker_service import speaker_service

SAMPLE_RATE = 16000
RECORD_SECONDS = 8.0


def record_reference_audio(duration_sec: float = RECORD_SECONDS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """지정된 시간 동안 마이크로 녹음해 16kHz float32 1D 오디오 배열을 반환한다."""
    print(f"\n[화자 등록] {duration_sec:.0f}초 동안 평소 말투로 자유롭게 말씀해주세요...")
    print("(예: \"안녕, 나는 이 로봇의 주인이야. 오늘 날씨가 참 좋네. 저녁엔 뭘 먹을지 고민이야.\")")
    audio = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("[화자 등록] 녹음 완료.")
    return audio.flatten()


def enroll(duration_sec: float = RECORD_SECONDS) -> str:
    """마이크 녹음 -> 임베딩 추출 -> .npy 저장까지의 등록 파이프라인을 실행하고 저장 경로를 반환한다."""
    audio = record_reference_audio(duration_sec=duration_sec)

    print("[화자 등록] 화자 임베딩 추출 중...")
    embedding = speaker_service.embed(audio, sample_rate=SAMPLE_RATE)

    output_path = settings.SPEAKER_REFERENCE_EMBEDDING_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embedding)

    speaker_service.reload_reference_embedding()

    print(f"[화자 등록] 기준 화자(primary_user) 임베딩 저장 완료 -> {output_path}")
    print(f"[화자 등록] 검증 임계값(코사인 유사도): {settings.SPEAKER_VERIFICATION_THRESHOLD}")
    return output_path


if __name__ == "__main__":
    enroll()
