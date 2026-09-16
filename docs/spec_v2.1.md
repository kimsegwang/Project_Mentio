# [Project Mentio] 시스템 기능 요약 명세서 (v2.1)

본 문서는 **DFRobot FireBeetle 2 ESP32-S3 (N16R8)** 기반 계층형 엣지-클라우드 AI 데스크 감성 로봇 'Project Mentio'의 하드웨어 규격, 데이터베이스 스키마 및 소프트웨어 파이프라인 세부 명세서입니다.

---

## 1. 시스템 아키텍처 다이어그램

```text
[물리 환경 / 사용자]
  │ (마이크 음성 / 카메라 스냅샷 / 정전식 터치 / 6축 모션 / 온습도·조도)
  ▼
┌─────────────────────────────────────────────────────────────┐
│ [에지 디바이스: DFRobot FireBeetle 2 ESP32-S3 (N16R8)]      │
│  - Core 0: Wi-Fi 비동기 통신 & I2S 오디오 버퍼링/스트리밍    │
│  - Core 1: 센서 폴링, I2C OLED 표정 렌더링, 터치/LED 제어    │
└──────────────┬──────────────────────────────▲───────────────┘
               │ (Wi-Fi: WebSocket / HTTP)    │ (JSON 제어 명령 + TTS 스트림)
               ▼                              │
┌─────────────────────────────────────────────┴───────────────┐
│ [로컬 엣지 게이트웨이: Python Fast Engine (PC)]             │
│  - Speaker Verification: 사용자 음성 임베딩 유사도 검증      │
│  - STT: faster-whisper (base, int8, CPU 구동, ~1.0s)        │
│  - Intent Engine: 5단계 분기 (부정어/룰 기반 시간/비전/텍스트)│
│  - RAG Memory: PostgreSQL (pgvector) 사용자 장기 기억 검색  │
│  - Vision Gesture: Google MediaPipe Hands (손하트 실시간)    │
│  - AI Brain: Gemini 3.6 Flash (thinking_budget=1, E2E ~2.6s)│
│  - TTS: Edge-TTS (비동기 합성 ➔ 청크 스트리밍 + 에코 가드)   │
└──────────────┬──────────────────────────────────────────────┘
               ▼ (Async Connection Pool)
┌─────────────────────────────────────────────────────────────┐
│ [데이터베이스: PostgreSQL]                                   │
│  - emotion_presets: 감정별 LED RGB 및 복귀 지속시간(SSOT)   │
│  - interaction_logs: 대화/비전/제스처 E2E 레이턴시 프로파일링│
│  - user_long_term_memory: pgvector 기반 대화 에피소드 기억  │
│  - sensor_telemetry: 시계열 센서 텔레메트리 적재             │
└─────────────────────────────────────────────────────────────┘

```

---

## 2. 하드웨어 구성 및 핀 배정 (FireBeetle 2)

| 기능 블록 | 부품명 | 할당 GPIO | 인터페이스 / 연결 방식 |
| :--- | :--- | :--- | :--- |
| **비전 영상 입력** | OV2640 카메라 | *(온보드 FPC)* | 온보드 FPC 전용 슬롯 직결 (점퍼선 0개) |
| **I2C 공용 버스** | 0.96"/1.3" OLED, AHT20, MPU-6050 | **GPIO 1 (SDA)<br>GPIO 2 (SCL)** | 병렬 공유 버스 (주소: 0x3C, 0x38, 0x68 충돌 없음) |
| **I2S 스피커 출력** | MAX98357A DAC 앰프 | **GPIO 6 (BCLK)<br>GPIO 7 (LRC)<br>GPIO 8 (DIN)** | 디지털 오디오 출력 (16kHz, 16bit PCM 스트리밍) |
| **I2S 마이크 입력** | INMP441 디지털 마이크 | **GPIO 9 (SCK)<br>GPIO 10 (WS)<br>GPIO 11 (SD)** | 디지털 음성 수집 (16kHz Mono 녹음 버퍼) |
| **감정 상태 LED** | WS2812B NeoPixel | **GPIO 5 (D2)** | 단일 데이터선 제어 (RGB 색상 및 브리딩 효과) |
| **주변 조도 감지** | CDS 광센서 모듈 | **GPIO 17 (A0)** | ADC1 전용 아날로그 입력 (암전 감지 및 수면 연동) |
| **정전식 터치** | 정전식 터치 패드 | **GPIO 12 (Touch12)** | ESP32-S3 내장 Touch 채널 (단발 탭 / 3초 롱터치 Mute) |

---

## 3. 데이터베이스(PostgreSQL) 스키마 명세

```sql
-- 1. pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 감정 상태 및 LED/표정 설정 (SSOT)
CREATE TABLE IF NOT EXISTS emotion_presets (
    emotion_name VARCHAR(32) PRIMARY KEY,
    oled_expression VARCHAR(32) NOT NULL,
    led_r INT NOT NULL,
    led_g INT NOT NULL,
    led_b INT NOT NULL,
    duration_sec FLOAT NOT NULL DEFAULT 4.5
);

-- 기본 프리셋 초기값
INSERT INTO emotion_presets (emotion_name, oled_expression, led_r, led_g, led_b, duration_sec) VALUES
('NEUTRAL', 'DEFAULT', 0, 0, 0, 0.0),
('HAPPY', 'HAPPY_EYES', 255, 180, 0, 4.5),
('HEART_EYES', 'HEART_EYES', 255, 20, 120, 5.0),
('SAD', 'SAD_EYES', 0, 80, 255, 4.0),
('ANGRY', 'ANGRY_EYES', 255, 30, 30, 4.0),
('SURPRISED', 'BIG_EYES', 255, 255, 255, 4.0),
('DIZZY', 'SPIRAL_EYES', 160, 32, 240, 4.5),
('THINKING', 'LOOK_UP', 0, 200, 255, 0.0)
ON CONFLICT (emotion_name) DO NOTHING;

-- 3. 장기 기억(RAG) 저장소
CREATE TABLE IF NOT EXISTS user_long_term_memory (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) DEFAULT 'primary_user',
    fact_text TEXT NOT NULL,
    embedding vector(384),  -- all-MiniLM-L6-v2 기준 384차원
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_memory_embedding 
ON user_long_term_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 4. 레이턴시 및 인터랙션 프로파일링 로그
CREATE TABLE IF NOT EXISTS interaction_logs (
    id BIGSERIAL PRIMARY KEY,
    trace_id VARCHAR(64) NOT NULL,
    intent_type VARCHAR(32) NOT NULL,
    user_text TEXT,
    bot_reply TEXT,
    stt_latency_ms INT,
    rag_latency_ms INT,
    llm_latency_ms INT,
    tts_latency_ms INT,
    total_e2e_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. 센서 텔레메트리
CREATE TABLE IF NOT EXISTS sensor_telemetry (
    id BIGSERIAL PRIMARY KEY,
    temperature FLOAT,
    humidity FLOAT,
    light_level INT,
    motion_state VARCHAR(32),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

```

---

## 4. 소프트웨어 파이프라인 세부 명세

### 1) AI 두뇌 & 장기 기억(RAG) 파이프라인
* **초저지연 추론 엔진 (Gemini 3.6 Flash)**:
  * `thinking_budget=1` 설정으로 내부 추론 시간 단축 (순수 LLM 1.5초대 달성).
  * `max_output_tokens=1024` 이상 유지하여 멀티바이트 한국어 잘림 방어.
  * **Auto-healing 파서**: 슬라이싱(`find`/`rfind`) 및 미완성 괄호 자동 보정(`..."}`) 탑재.
* **장기 기억 검색 & 적재 (RAG)**:
  * **검색**: 사용자 질문 인입 시 로컬 경량 임베딩(0.03초) $\rightarrow$ `pgvector` 코사인 유사도 상위 1~2개 문장(`Top-K=2`)만 프롬프트에 주입.
  * **적재**: 대화 턴 종료 후 Background Task에서 비동기로 대화 요약/추출 및 DB 적재 (대화 레이턴시 영향 0초).
* **5단계 의도 분석 엔진 (IntentService)**:
  1. **호출어 정규화**: "멘티오는", "멘티오가" 등 후행 조사 제거.
  2. **룰 기반 즉시 처리**: "몇 시야?", "지금 몇 시" 등 시간 질의 감지 시 로컬 시스템 시계(`datetime.now()`) 기반 0.1초 즉시 음성 안내.
  3. **시각 부정어(Veto) 차단**: "보지 마", "사진 찍지 마" 검출 시 `VOICE_CHAT`으로 강제 우회.
  4. **순수 시각 요구 감지**: "봐봐", "사진 찍어줘" 검출 시 480p 단발 스냅샷 캡처 및 `VOICE_VISION` 모드 진입.
  5. **기본 Fallback**: 일상 발화는 모두 초고속 순수 텍스트 대화(`VOICE_CHAT`)로 안전 수렴.
* **이원화 비전 처리**:
  * **온디맨드 스냅샷(VLM)**: 영상 상시 스트리밍을 배제하고 필요 시에만 1장 전송하여 대역폭 고갈 및 발열 차단.
  * **로컬 제스처 인식**: PC 웹캠 기반 MediaPipe Hands로 손하트 감지 시 즉각 `HEART_EYES` 및 핫핑크 LED(`[255, 20, 120]`) 표출.

### 2) 음성 처리 & 스마트 인터랙션 제어
* **로컬 고속 STT 엔진 (faster-whisper base, int8)**:
  * CPU int8 양자화 및 스레드 4개 할당으로 1.0~1.2초대 변환 시간 유지.
  * `initial_prompt="멘티오, Mentio"` 주입으로 고유명사 오인식 차단.
* **화자 식별 (Speaker Verification)**:
  * 기준 화자의 음성 임베딩 벡터 사전 등록 후 발화 시 코사인 유사도 비교. 임계치 미달(제3자 잡음, TV 소리) 시 파이프라인 진입 차단.
* **대화 모드 제어 (Conversation Mute)**:
  * **음성 제어**: "대화 그만", "조용히 해" 발화 시 Mute 플래그 활성화.
  * **하드웨어 제어**: 정전식 터치 패드(GPIO 12) 3초 롱터치 시 Mute ON/OFF 토글.
  * **Mute 상태**: 마이크 입력 버퍼 즉시 폐기 및 OLED에 취침(`Zzz`) 아이콘 표시.
* **동시성 제어 및 잔향 가드**:
  * **Busy-Dropping**: 로봇 발화 및 생각 중 인입된 마이크 버퍼 즉각 폐기.
  * **에코 가드**: 스피커 재생 완료 후 0.4초간 마이크 청취를 차단하여 하울링 방지.

### 3) 감성 동기화 & 물리 반응
* **발화-UI 0ms 동기화 및 타이머 앵커링**:
  * 음성 첫 음절 출력 직전에 OLED 표정과 LED 색상을 즉시 전환.
  * 표정 타이머는 발화 완료 시점부터 카운트다운을 시작하여 설정된 지속시간(4.0~5.0s) 유지 후 `NEUTRAL` 자동 복귀.
* **6축 모션 및 물리 인터랙션 (MPU-6050 & Touch)**:
  * **흔들림 감지**: 임계치 초과 시 `DIZZY(어지러움)` 표정 및 회전 연출.
  * **낙하 감지**: 가속도 합 $0G$ 근접 감지 시 `SURPRISED(놀람)` 표정 연출.
  * **머리 쓰다듬기**: 정전식 터치 단발 탭 감지 시 `HAPPY(기쁨)` 반응 및 골드 LED 점등.

### 4) 데스크 유틸리티 & 환경 반응
* **AHT20 온습도 센서**: 고온 다습 지속 감지 시 피로(`TIRED`) 표정 전환 및 환기 알림.
* **CDS 조도 센서**: 암전 지속 감지 시 디스플레이 OFF 및 취침 모드 진입.
* **데스크 생산성**: 뽀모도로(25분 집중 타이머) 및 날씨 모닝 브리핑 지원.

---

## 5. 펌웨어 태스크 분리 (FreeRTOS)
* **Core 0 (통신 & 오디오 스트리밍 전담)**:
  * Wi-Fi WebSocket 세션 유지 및 자동 재연결(Auto-Reconnect).
  * I2S 오디오 청크 스트리밍 송수신 (PC $\rightarrow$ ESP32 스피커, ESP32 마이크 $\rightarrow$ PC).
* **Core 1 (실시간 I/O 및 렌더링 전담)**:
  * I2C 센서 폴링(AHT20, MPU-6050), ADC 조도 계측, 정전식 터치 인터럽트 감지.
  * OLED 눈동자 애니메이션 프레임 버퍼 드라이빙 및 WS2812B NeoPixel 제어.
* **전원 구성**:
  * **프로토타입 단계**: USB-C 5V 직결 (브라운아웃 차단).
  * **무선화 패키징 단계**: FireBeetle 2 온보드 내장 PMU 기반 3.7V 리튬 배터리 연동.