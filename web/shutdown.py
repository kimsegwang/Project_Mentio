"""
web/shutdown.py
관리 대시보드 종료 처리 (Ctrl+C / Ctrl+Break / 콘솔 창 닫기 / 정상 종료).

[Windows 종료 지연 원인]
Streamlit은 Windows에서 SelectorEventLoop를 강제하는데, 메인 스레드가 select()에서 대기하는 동안에는
Ctrl+C가 와도 Python 시그널 핸들러(=Streamlit의 server.stop())가 실행되지 않는다. 브라우저 탭이 열려 있으면
1초 주기 웹소켓 ping이 루프를 깨워 곧바로 종료되지만, 탭을 닫은 유휴 상태에서는 루프를 깨울 이벤트가 없어
프로세스가 수 분간 종료되지 않는다 (DB 풀과 무관하게 재현됨).

[해결]
SetConsoleCtrlHandler로 등록한 콘솔 제어 핸들러는 OS가 새 스레드에서 호출하므로 select()에 막히지 않는다.
여기서 DB 커넥션 풀을 즉시 closeall()하고 표준 출력을 flush한 뒤, 마지막으로 스레드 안전한 공개 API
Runtime.stop()을 호출해(내부적으로 call_soon_threadsafe로 루프를 깨움) Streamlit 정상 종료를 시작시킨다.
POSIX에서는 select()가 시그널에 의해 EINTR로 깨어나므로 atexit 정리만으로 충분하다.
"""
import atexit
import logging
import os
import sys
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Windows 콘솔 제어 이벤트 코드
CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2
_HANDLED_EVENTS = {CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT}


class ShutdownCoordinator:
    """
    종료 이벤트 처리 로직 (OS API와 분리해 단위 테스트 가능).

    - 1번째 종료 이벤트: 런타임 정지 요청 → 리소스 정리(풀 closeall) 순으로 즉시 수행.
    - 2번째 이후: 정상 종료가 어떤 이유로든 지연 중이라는 뜻이므로 강제 종료(escape hatch).
    - 정리 콜백은 atexit 경로와 중복 호출되어도 한 번만 실행된다.
    """

    def __init__(
        self,
        cleanup: Callable[[], None],
        stop_runtime: Callable[[], None],
        force_exit: Callable[[int], None] = os._exit,
    ):
        self._cleanup = cleanup
        self._stop_runtime = stop_runtime
        self._force_exit = force_exit
        self._lock = threading.Lock()
        self._cleaned_up = False
        self._event_count = 0

    def cleanup_once(self) -> None:
        with self._lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True
        try:
            self._cleanup()
        except Exception as e:
            logger.error(f"[Shutdown] 리소스 정리 실패: {e}")

    def handle_console_event(self, event: int) -> bool:
        """
        콘솔 제어 이벤트 콜백. 항상 False를 반환해 다음 핸들러(Python 기본 SIGINT 처리 → Streamlit
        signal_handler)도 이어서 실행되게 한다.

        ⚠️ 순서 불변식: [리소스 정리 → stdout/stderr flush] 를 모두 끝낸 뒤 **마지막 동작으로만**
        런타임 정지를 요청한다. 이 콜백은 OS가 만든 외부 스레드(데몬 취급)에서 실행되므로, 정지 요청으로
        메인 스레드가 인터프리터 종료(finalizing)에 들어간 뒤에도 이 스레드가 print 중이면 stdout 버퍼 락을
        쥔 채로 남아 "Fatal Python error: _enter_buffered_busy"로 프로세스가 중단된다.
        """
        if event not in _HANDLED_EVENTS:
            return False
        if sys.is_finalizing():
            # 메인 스레드가 이미 인터프리터를 정리 중: 이 스레드에서는 어떤 Python I/O도 하면 안 된다.
            return False

        with self._lock:
            self._event_count += 1
            count = self._event_count

        if count > 1:
            # 두 번째 Ctrl+C는 대개 메인 스레드가 종료(finalizing) 절차를 밟는 도중에 도착한다.
            # 여기서 print/flush로 stdout 버퍼 락을 잡으면 finalizing과 경합해 Fatal Python error가 나므로,
            # 버퍼를 거치지 않는 os.write로만 안내하고 finalizing을 건너뛰는 os._exit으로 즉시 종료한다.
            # (풀 정리는 첫 번째 이벤트에서 이미 끝났으므로 cleanup_once는 no-op)
            self.cleanup_once()
            _write_raw_stderr("  Force exit (second Ctrl+C).\n")
            self._force_exit(1)
            return False

        self.cleanup_once()
        _flush_std_streams()
        try:
            self._stop_runtime()  # 이후 이 스레드는 Python I/O를 건드리지 않는다
        except Exception as e:
            # 정지 요청 실패 = 메인 스레드가 종료 단계에 들어가지 않음 → 여기서의 출력은 안전하다
            logger.error(f"[Shutdown] Streamlit 런타임 정지 요청 실패: {e}")
            _flush_std_streams()
        return False


def _write_raw_stderr(message: str) -> None:
    """
    Python 버퍼(BufferedWriter)와 그 락을 거치지 않고 fd 2에 직접 쓴다 (종료 경합 구간 전용).
    콘솔 코드 페이지(cp949 등)와 무관하게 깨지지 않도록 ASCII 메시지만 사용한다.
    """
    try:
        os.write(2, message.encode("ascii", errors="replace"))
    except Exception:
        pass


def _flush_std_streams() -> None:
    """콘솔 핸들러 스레드가 쓴 출력을 버퍼에 남기지 않도록 비운다 (종료 중 닫힌 스트림은 무시)."""
    for stream in {id(s): s for s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__) if s}.values():
        try:
            stream.flush()
        except Exception:
            pass


_installed_coordinator: Optional[ShutdownCoordinator] = None
_install_lock = threading.Lock()
_console_handler_ref = None  # ctypes 콜백이 GC되면 OS가 해제된 함수 포인터를 호출하므로 참조를 유지한다


def _stop_streamlit_runtime() -> None:
    from streamlit.runtime import Runtime

    if Runtime.exists():
        Runtime.instance().stop()


def _register_windows_console_handler(coordinator: ShutdownCoordinator) -> None:
    global _console_handler_ref
    import ctypes
    from ctypes import wintypes

    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetConsoleCtrlHandler.argtypes = (handler_type, wintypes.BOOL)
    kernel32.SetConsoleCtrlHandler.restype = wintypes.BOOL

    _console_handler_ref = handler_type(lambda event: coordinator.handle_console_event(int(event)))
    if not kernel32.SetConsoleCtrlHandler(_console_handler_ref, True):
        raise ctypes.WinError(ctypes.get_last_error())


def install_shutdown_handlers(cleanup: Callable[[], None]) -> ShutdownCoordinator:
    """
    프로세스당 한 번만 종료 핸들러를 설치한다 (Streamlit 재실행/다중 세션에서 중복 호출돼도 안전).
    - 모든 OS: atexit로 정상 종료 시 정리.
    - Windows: 콘솔 제어 핸들러로 유휴 상태 Ctrl+C 종료 지연 해소 + 즉시 정리.
    """
    global _installed_coordinator
    with _install_lock:
        if _installed_coordinator is not None:
            return _installed_coordinator

        coordinator = ShutdownCoordinator(cleanup=cleanup, stop_runtime=_stop_streamlit_runtime)
        atexit.register(coordinator.cleanup_once)
        if sys.platform == "win32":
            try:
                _register_windows_console_handler(coordinator)
            except Exception as e:
                logger.error(f"[Shutdown] Windows 콘솔 제어 핸들러 등록 실패 (atexit 정리만 사용): {e}")
        _installed_coordinator = coordinator
        return coordinator
