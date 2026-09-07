import os
import sys

# 프로젝트 루트 경로 등록
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.repositories.connection import init_db_pool, close_db_pool, get_db_connection
from server.repositories.log_repository import insert_interaction_log
from server.schemas.action import RobotAction


def run_db_test():
    print("[Test] PostgreSQL 연결 및 스키마 검증 시작...")
    init_db_pool()

    # 1. schema.sql 실행하여 테이블 자동 생성
    schema_path = os.path.join("server", "repositories", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        ddl = f.read()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    print("[Test] 테이블 스키마 준비 완료.")

    # 2. 더미 RobotAction 생성 및 INSERT 테스트
    dummy_action = RobotAction(
        emotion="HAPPY",
        speech="DB 연동 테스트 대사입니다!",
        led_rgb=[0, 255, 120]
    )

    insert_interaction_log(
        trigger_type="GESTURE",
        prompt="사용자가 손하트를 했습니다.",
        action=dummy_action,
        latency_seconds=1.234
    )

    close_db_pool()
    print("[Test] DB 테스트 완료!")


if __name__ == "__main__":
    run_db_test()