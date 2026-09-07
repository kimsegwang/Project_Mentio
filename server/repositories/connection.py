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
    if _db_pool is None:
        init_db_pool()

    conn = _db_pool.getconn()
    try:
        yield conn
    finally:
        _db_pool.putconn(conn)