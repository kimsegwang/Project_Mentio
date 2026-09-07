from server.repositories.connection import get_db_connection
from server.schemas.action import RobotAction


def insert_interaction_log(
    trigger_type: str,
    prompt: str,
    action: RobotAction,
    latency_seconds: float
) -> None:
    """Gemini 추론 완료 결과를 PostgreSQL에 비동기/동기 적재"""
    query = """
        INSERT INTO interaction_logs (
            trigger_type, prompt, emotion, speech,
            led_r, led_g, led_b, latency_ms
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        );
    """
    latency_ms = round(latency_seconds * 1000, 2)
    led_r, led_g, led_b = action.led_rgb

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        trigger_type,
                        prompt,
                        action.emotion,
                        action.speech,
                        led_r,
                        led_g,
                        led_b,
                        latency_ms
                    )
                )
            conn.commit()
            print(f"[Database] 로그 적재 완료 (Type: {trigger_type}, Latency: {latency_ms}ms)")
    except Exception as e:
        print(f"[Database Error] 로그 적재 실패: {e}")