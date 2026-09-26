from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2 import pool

from config import settings

_db_pool: pool.ThreadedConnectionPool | None = None


def init_db_pool() -> pool.ThreadedConnectionPool:
    global _db_pool
    if _db_pool is None:
        try:
            _db_pool = pool.ThreadedConnectionPool(
                minconn=settings.DB_POOL_MIN_CONN,
                maxconn=settings.DB_POOL_MAX_CONN,
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                dbname=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
            )
            print("[Database] PostgreSQL ThreadedConnectionPool 초기화 완료.")
        except Exception as e:
            print(f"[Database Error] 커넥션 풀 초기화 실패: {e}")
            raise e
    return _db_pool


def close_db_pool() -> None:
    global _db_pool
    if _db_pool is not None:
        _db_pool.closeall()
        _db_pool = None
        print("[Database] 모든 DB 커넥션이 안전하게 종료되었습니다.")


@contextmanager
def get_db_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """커넥션 대여 및 작업 후 풀로 자동 반납하는 컨텍스트 매니저"""
    # 대여한 풀을 지역 변수로 고정: 사용 중 close_db_pool()이 전역을 None으로 바꿔도(종료 신호 처리 등)
    # 반납 대상이 바뀌지 않도록 한다.
    db_pool = _db_pool if _db_pool is not None else init_db_pool()

    conn = db_pool.getconn()
    try:
        yield conn
    except Exception:
        # 💡 실패한 트랜잭션 상태로 커넥션이 풀에 반납되면, 다음 대여자의 정상 쿼리까지
        #    InFailedSqlTransaction으로 연쇄 실패시키므로 반드시 롤백 후 반납한다.
        #    (종료 중 closeall()로 이미 닫힌 커넥션은 롤백 대상이 없고, 롤백 시도 자체가 원래 예외를 가린다)
        if not conn.closed:
            conn.rollback()
        raise
    finally:
        # 이미 closeall()된 풀에 putconn하면 PoolError가 발생하므로 열린 풀에만 반납한다.
        if not db_pool.closed:
            db_pool.putconn(conn)