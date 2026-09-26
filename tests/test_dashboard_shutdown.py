"""
tests/test_dashboard_shutdown.py
대시보드 종료 처리(web/shutdown.py)와 종료 중 커넥션 반납 가드(server/repositories/connection.py) 검증.
OS 콘솔 API/실제 DB 없이 콜백·Fake 풀로 검증한다.
"""
import pytest

from server.repositories import connection
from web import shutdown
from web.shutdown import CTRL_BREAK_EVENT, CTRL_C_EVENT, CTRL_CLOSE_EVENT, ShutdownCoordinator


class Recorder:
    def __init__(self):
        self.calls = []

    def cleanup(self):
        self.calls.append("cleanup")

    def stop_runtime(self):
        self.calls.append("stop_runtime")

    def force_exit(self, code):
        self.calls.append(("force_exit", code))


def _coordinator(rec: Recorder) -> ShutdownCoordinator:
    return ShutdownCoordinator(cleanup=rec.cleanup, stop_runtime=rec.stop_runtime, force_exit=rec.force_exit)


@pytest.fixture
def rec(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(shutdown, "_flush_std_streams", lambda: recorder.calls.append("flush"))
    monkeypatch.setattr(shutdown, "_write_raw_stderr", lambda msg: recorder.calls.append("raw_write"))
    return recorder


# --- ShutdownCoordinator ---

@pytest.mark.parametrize("event", [CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT])
def test_first_event_cleans_up_and_flushes_before_stopping_runtime(rec, event):
    """
    런타임 정지 요청이 메인 스레드를 인터프리터 종료로 보내므로, 그 이전에 출력(print)과 flush가 끝나야 하고
    정지 요청 이후 이 스레드는 아무 I/O도 하지 않아야 한다 (Fatal Python error: _enter_buffered_busy 방지).
    """
    passed_on = _coordinator(rec).handle_console_event(event)

    assert rec.calls == ["cleanup", "flush", "stop_runtime"]
    # False: Python 기본 SIGINT 처리(Streamlit signal_handler)도 이어서 실행되어야 한다
    assert passed_on is False


def test_unrelated_console_event_is_ignored(rec):
    assert _coordinator(rec).handle_console_event(5) is False  # CTRL_LOGOFF_EVENT
    assert rec.calls == []


def test_second_event_force_exits_without_touching_python_stdout_buffer(rec, capsys):
    """
    실기 재현: 종료 도중 두 번째 Ctrl+C의 print/flush가 finalizing과 stdout 버퍼 락을 경합해
    "Fatal Python error: _enter_buffered_busy"가 발생했다. 두 번째 이벤트는 버퍼를 거치지 않는
    raw write만 사용해야 하며 flush/print를 해서는 안 된다.
    """
    coordinator = _coordinator(rec)
    coordinator.handle_console_event(CTRL_C_EVENT)
    rec.calls.clear()
    capsys.readouterr()

    coordinator.handle_console_event(CTRL_C_EVENT)

    assert rec.calls == ["raw_write", ("force_exit", 1)]
    assert capsys.readouterr().out == ""


def test_event_during_interpreter_finalization_does_nothing(rec, monkeypatch):
    monkeypatch.setattr(shutdown.sys, "is_finalizing", lambda: True)

    assert _coordinator(rec).handle_console_event(CTRL_C_EVENT) is False
    assert rec.calls == []


def test_write_raw_stderr_bypasses_python_buffers(monkeypatch):
    written = []
    monkeypatch.setattr(shutdown.os, "write", lambda fd, data: written.append((fd, data)))

    shutdown._write_raw_stderr("  Force exit (second Ctrl+C).\n")

    assert written == [(2, b"  Force exit (second Ctrl+C).\n")]


def test_cleanup_runs_once_across_signal_and_atexit_paths(rec):
    coordinator = _coordinator(rec)

    coordinator.handle_console_event(CTRL_C_EVENT)
    coordinator.cleanup_once()  # atexit 경로
    coordinator.cleanup_once()

    assert rec.calls.count("cleanup") == 1


def test_runtime_stop_failure_is_logged_and_flushed(rec):
    def broken_stop():
        raise RuntimeError("runtime not started")

    coordinator = ShutdownCoordinator(cleanup=rec.cleanup, stop_runtime=broken_stop, force_exit=rec.force_exit)
    coordinator.handle_console_event(CTRL_C_EVENT)

    assert rec.calls == ["cleanup", "flush", "flush"]


def test_flush_std_streams_tolerates_closed_streams(monkeypatch):
    import io

    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(shutdown.sys, "stdout", closed)

    shutdown._flush_std_streams()  # 종료 중 닫힌 스트림이어도 예외가 전파되면 안 된다


def test_cleanup_exception_is_swallowed():
    def broken_cleanup():
        raise RuntimeError("pool already gone")

    coordinator = ShutdownCoordinator(cleanup=broken_cleanup, stop_runtime=lambda: None, force_exit=lambda c: None)

    # 종료 경로에서 예외가 전파되면 안 된다
    coordinator.cleanup_once()


def test_install_shutdown_handlers_is_idempotent(monkeypatch):
    registered = []
    monkeypatch.setattr(shutdown, "_installed_coordinator", None)
    monkeypatch.setattr(shutdown.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(shutdown, "_register_windows_console_handler", lambda c: registered.append("console"))

    first = shutdown.install_shutdown_handlers(cleanup=lambda: None)
    second = shutdown.install_shutdown_handlers(cleanup=lambda: None)

    assert first is second
    expected = 2 if shutdown.sys.platform == "win32" else 1
    assert len(registered) == expected


# --- get_db_connection(): 종료 중(closeall 이후) 반납/롤백 가드 ---

class FakeConn:
    def __init__(self):
        self.closed = 0
        self.rolled_back = False

    def rollback(self):
        if self.closed:
            raise AssertionError("closed connection must not be rolled back")
        self.rolled_back = True


class FakePool:
    def __init__(self):
        self.closed = False
        self.conn = FakeConn()
        self.returned = []

    def getconn(self):
        return self.conn

    def putconn(self, conn):
        if self.closed:
            raise AssertionError("must not putconn into a closed pool")
        self.returned.append(conn)

    def closeall(self):
        self.closed = True
        self.conn.closed = 1


@pytest.fixture
def fake_pool(monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(connection, "_db_pool", pool)
    return pool


def test_get_db_connection_returns_conn_to_open_pool(fake_pool):
    with connection.get_db_connection() as conn:
        assert conn is fake_pool.conn

    assert fake_pool.returned == [fake_pool.conn]


def test_get_db_connection_still_rolls_back_on_error(fake_pool):
    with pytest.raises(ValueError):
        with connection.get_db_connection():
            raise ValueError("query failed")

    assert fake_pool.conn.rolled_back is True
    assert fake_pool.returned == [fake_pool.conn]


def test_get_db_connection_survives_pool_closed_mid_query(fake_pool):
    """종료 신호로 사용 중 close_db_pool()이 실행돼도 원래 예외만 전파되고 PoolError로 가려지지 않는다."""
    with pytest.raises(ValueError, match="query failed"):
        with connection.get_db_connection():
            connection.close_db_pool()  # 전역 _db_pool=None + closeall
            raise ValueError("query failed")

    assert fake_pool.returned == []
    assert connection._db_pool is None


def test_get_db_connection_normal_exit_after_pool_closed(fake_pool):
    with connection.get_db_connection():
        connection.close_db_pool()

    assert fake_pool.returned == []
