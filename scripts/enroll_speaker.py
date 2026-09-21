"""
scripts/enroll_speaker.py
가족 구성원 등 다중 화자의 목소리를 마이크로 녹음해 화자 임베딩을 DB(speaker_profiles)에
등록/갱신하는 CLI 스크립트.

기존 단일 .npy 파일 방식(primary_user.npy)에서 탈피해 DB 기반으로 관리하며, SpeakerService가
1:N 코사인 유사도로 가장 일치하는 화자를 찾는 identify_speaker()의 등록 대상이 된다.
--user-id를 생략하면 settings.DEFAULT_USER_ID("primary_user")로 등록되어 기존 단일 사용자
호환 동작(가장 먼저/유일하게 등록된 화자)을 그대로 유지한다.

실행:
    python scripts/enroll_speaker.py --user-id dad --name 아빠
    python scripts/enroll_speaker.py                      # --user-id 생략 시 primary_user로 등록
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sounddevice as sd

from config import settings
from server.repositories.speaker_repository import upsert_speaker_profile
from server.repositories.connection import init_db_pool, close_db_pool
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


def enroll(user_id: str, display_name: str, duration_sec: float = RECORD_SECONDS) -> bool:
    """
    마이크 녹음 -> 임베딩 추출 -> speaker_profiles UPSERT까지의 등록 파이프라인을 실행한다.
    성공 시 SpeakerService의 활성 화자 캐시를 무효화해, 재시작 없이 즉시 식별 대상에 반영한다.
    """
    audio = record_reference_audio(duration_sec=duration_sec)

    print("[화자 등록] 화자 임베딩 추출 중...")
    embedding = speaker_service.embed(audio, sample_rate=SAMPLE_RATE)

    success = upsert_speaker_profile(user_id=user_id, display_name=display_name, embedding=embedding.tolist())
    if not success:
        print("[화자 등록] DB 저장 실패. 위 로그를 확인하세요.")
        return False

    speaker_service.reload_speaker_profiles()

    print(f"[화자 등록] 화자 등록 완료 -> user_id: {user_id}, name: {display_name}")
    print(f"[화자 등록] 검증 임계값(코사인 유사도): {settings.SPEAKER_VERIFICATION_THRESHOLD}")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="화자 프로필을 DB에 등록/갱신합니다.")
    parser.add_argument(
        "--user-id",
        default=settings.DEFAULT_USER_ID,
        help=f"등록할 화자의 고유 식별자 (생략 시 기존 호환값: {settings.DEFAULT_USER_ID})",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="화자의 표시 이름 (생략 시 --user-id 값을 그대로 사용)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=RECORD_SECONDS,
        help=f"녹음 시간(초) (기본값: {RECORD_SECONDS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    display_name = args.name or args.user_id

    init_db_pool()
    try:
        enroll(user_id=args.user_id, display_name=display_name, duration_sec=args.duration)
    finally:
        close_db_pool()
