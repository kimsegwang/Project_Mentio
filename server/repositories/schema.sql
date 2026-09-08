-- 기존 테이블 초기화 (의존성 고려해 역순 삭제)
DROP TABLE IF EXISTS interaction_logs CASCADE;
DROP TABLE IF EXISTS emotion_presets CASCADE;

-- 1. 감정별 고정 RGB 및 기본 동작 시간 프리셋 테이블
CREATE TABLE IF NOT EXISTS emotion_presets (
    emotion VARCHAR(30) PRIMARY KEY,
    r INT NOT NULL CHECK (r BETWEEN 0 AND 255),
    g INT NOT NULL CHECK (g BETWEEN 0 AND 255),
    b INT NOT NULL CHECK (b BETWEEN 0 AND 255),
    duration NUMERIC(4, 1) NOT NULL DEFAULT 4.0,
    description TEXT
);

-- 2. 14종 초기 감정 프리셋 시드 적재
INSERT INTO emotion_presets (emotion, r, g, b, duration, description) VALUES
('NEUTRAL', 100, 100, 100, 0.0, '평상시 대기 상태 (눈 깜빡임, 두리번)'),
('HAPPY', 255, 200, 0, 4.0, '긍정 반응, 칭찬, 반가운 인사'),
('PROUD', 255, 215, 0, 4.0, '성공, 과제 완료, 뿌듯함 표현'),
('CURIOUS', 140, 240, 80, 3.5, '새로운 물체 감지, 갸웃거리는 호기심'),
('SAD', 30, 80, 255, 4.0, '서운함, 부정적 상황, 위로 필요'),
('ANGRY', 255, 40, 20, 3.5, '과도한 방해 시 투덜거림, 뾰로통함'),
('HEART_EYES', 255, 20, 120, 5.0, '양손 하트 인식, 머리 정전식 터치'),
('SURPRISED', 0, 220, 255, 3.0, '들어올림, 낙하(0G 근접) 감지'),
('SCARED', 160, 32, 240, 3.0, '낙하 위험, 급격한 위협 충격 감지'),
('DIZZY', 160, 255, 0, 4.0, '흔들기 감지(MPU-6050 가속도/자이로 급변)'),
('TIRED', 255, 120, 0, 4.5, '고온 지속, 장시간 착석 스트레칭 유도'),
('SLEEPING', 15, 25, 60, 0.0, '조도 저하 암전 상태, 수면 모드'),
('FOCUS', 0, 180, 220, 0.0, '뽀모도로 25분 집중 타이머 집중 조명'),
('ERROR', 255, 0, 0, 3.0, '통신 장애 및 예외 발생')
ON CONFLICT (emotion) DO UPDATE 
SET r = EXCLUDED.r, g = EXCLUDED.g, b = EXCLUDED.b, duration = EXCLUDED.duration, description = EXCLUDED.description;

-- 3. 텔레메트리 상호작용 로그 테이블 (KST 타임존 적용)
CREATE TABLE IF NOT EXISTS interaction_logs (
    id SERIAL PRIMARY KEY,
    trigger_type VARCHAR(50) NOT NULL,
    emotion VARCHAR(30) NOT NULL REFERENCES emotion_presets(emotion),
    speech TEXT,
    led_r INT NOT NULL,
    led_g INT NOT NULL,
    led_b INT NOT NULL,
    duration NUMERIC(4, 1) NOT NULL,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);