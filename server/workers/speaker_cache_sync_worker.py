"""
server/workers/speaker_cache_sync_worker.py
화자 프로필 캐시 핫 리로드(Hot Reload) 폴링 워커.

[왜 필요한가]
SpeakerService는 활성 화자 임베딩을 최초 1회 DB에서 읽어 메모리에 캐싱한다. 그래서 대시보드
(web/app.py)에서 호칭 변경/비활성화/삭제를 해도 엔진을 재기동하기 전까지 반영되지 않았다.
이 워커는 백그라운드 스레드에서 speaker_profiles의 경량 변경 지문(MAX(updated_at), 행 수)만
주기적으로 조회하고, 지문이 바뀌었을 때에만 캐시를 재적재한다.

[안전성]
- 오디오 루프/VAD/STT와 분리된 전용 스레드에서 동작하므로 음성 처리를 블로킹하지 않는다.
- 로봇이 발화 처리 중(is_busy)이거나 음성 온보딩이 진행 중이면 갱신을 보류하고, 지문을
  "확인됨"으로 기록하지 않아 다음 주기에 다시 시도한다.
- 재적재가 실패(조회 결과 불일치 등)해도 기존 캐시를 유지하고 다음 주기에 재시도한다.
"""
import logging
import threading
from typing import Callable, Optional, Tuple

from config import settings
from server.repositories.speaker_repository import get_speaker_profiles_fingerprint
from server.services.speaker_service import SpeakerService, speaker_service
from server.services.voice_enrollment_service import VoiceEnrollmentService, voice_enrollment_service

logger = logging.getLogger(__name__)


class SpeakerCacheSyncWorker:
    def __init__(
        self,
        speaker_service_instance: SpeakerService = speaker_service,
        enrollment_service_instance: VoiceEnrollmentService = voice_enrollment_service,
        is_busy: Callable[[], bool] = lambda: False,
        poll_interval_sec: Optional[float] = None,
        fingerprint_loader: Callable[[], Optional[Tuple]] = get_speaker_profiles_fingerprint,
    ):
        self.speaker_service = speaker_service_instance
        self.enrollment_service = enrollment_service_instance
        self._is_busy = is_busy
        self.poll_interval_sec = (
            poll_interval_sec if poll_interval_sec is not None else settings.SPEAKER_CACHE_POLL_INTERVAL_SEC
        )
        self._load_fingerprint = fingerprint_loader
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # 마지막으로 캐시에 반영된 지문. 폴링 스레드에서만 읽고 쓰므로 별도 락이 필요 없다.
        self._last_fingerprint: Optional[Tuple] = None

    def start(self) -> None:
        """기준 지문을 기록한 뒤 백그라운드 폴링 스레드를 구동한다."""
        self._stop_event.clear()
        self._last_fingerprint = self._load_fingerprint()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[SpeakerCacheSync] 화자 캐시 핫 리로드 워커 시작 (주기: {self.poll_interval_sec:.0f}초).")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("[SpeakerCacheSync] 워커 스레드 종료 완료.")

    def _loop(self) -> None:
        # Event.wait로 대기해 stop() 호출 시 폴링 주기를 기다리지 않고 즉시 종료한다
        while not self._stop_event.wait(self.poll_interval_sec):
            try:
                self.poll_once()
            except Exception as e:
                logger.warning(f"[SpeakerCacheSync] 화자 캐시 폴링 중 예외 발생: {e}")

    def poll_once(self) -> bool:
        """
        지문을 한 번 확인해 변경 시 캐시를 재적재한다. 실제로 캐시를 교체했으면 True.
        조회 실패(None), 변경 없음, 갱신 보류, 재적재 실패 시에는 False.
        """
        fingerprint = self._load_fingerprint()
        if fingerprint is None or fingerprint == self._last_fingerprint:
            return False

        if self._is_busy():
            logger.info("[SpeakerCacheSync] 화자 프로필 변경 감지, 발화 처리 중이라 다음 주기로 보류")
            return False
        if self.enrollment_service.is_active():
            logger.info("[SpeakerCacheSync] 화자 프로필 변경 감지, 음성 온보딩 진행 중이라 다음 주기로 보류")
            return False

        active_count = fingerprint[2]
        if not self.speaker_service.refresh_speaker_profiles(expected_active_count=active_count):
            return False

        self._last_fingerprint = fingerprint
        return True
