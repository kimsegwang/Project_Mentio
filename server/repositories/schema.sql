-- Mentio 상호작용 로그 테이블
CREATE TABLE IF NOT EXISTS interaction_logs (
    id BIGSERIAL PRIMARY KEY,
    trigger_type VARCHAR(20) NOT NULL,          -- 'GESTURE' 또는 'SNAPSHOT'
    prompt TEXT NOT NULL,                        -- AI 질의 프롬프트
    emotion VARCHAR(30) NOT NULL,                -- 로봇 표정 상태 (HAPPY, HEART_EYES 등)
    speech TEXT NOT NULL,                        -- 로봇 한국어 대사
    led_r SMALLINT NOT NULL,                     -- WS2812B R (0~255)
    led_g SMALLINT NOT NULL,                     -- WS2812B G (0~255)
    led_b SMALLINT NOT NULL,                     -- WS2812B B (0~255)
    latency_ms NUMERIC(6, 2) NOT NULL,           -- AI 응답 지연 시간 (밀리초)
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 시계열 조회 및 통계 집계 최적화 인덱스
CREATE INDEX IF NOT EXISTS idx_interaction_created_at ON interaction_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_interaction_trigger_type ON interaction_logs (trigger_type);