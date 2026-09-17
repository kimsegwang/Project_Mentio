# [Project Mentio] 시스템 기능 요약 명세서 (v2.2)

본 문서는 **DFRobot FireBeetle 2 ESP32-S3 (N16R8)** 기반 계층형 엣지-클라우드 AI 데스크 감성 로봇 'Project Mentio'의 하드웨어 규격, 데이터베이스 스키마 및 소프트웨어 파이프라인 세부 명세서입니다. v2.1 대비 실제 코드베이스(`main` 기준)를 전수 검토하여 DB 스키마와 소프트웨어 파이프라인 항목을 최신화했습니다.

## v2.1 → v2.2 변경 이력

| 구분 | 내용 |
| :--- | :--- |
| DB 스키마 | `docs/spec_v2.1.md`의 DB 스키마 초안을 **`server/repositories/schema.sql` 실제 운영 스키마 기준**으로 전면 정정 (컬럼명/테이블 구조 상이). `user_long_term_memory`에 Invalidation 지원 컬럼(`is_active`, `superseded_by`, `updated_at`) 및 부분 인덱스 반영 |
| RAG 파이프라인 | 근접 중복 방지(Dedup, 임계값 0.85), 모순 해결(Invalidation, 0.55~0.85 후보 깔때기 + 소프트 딜리트), 순차 적재 큐(`MemoryWriteWorker`), 의문문 동적 임계값 완화(`QuestionDetector`, 0.68) 신규 반영 |
| BrainService | 공용 JSON 추출 헬퍼(`_extract_json_object`) 분리, 장기 기억 모순 판정 전용 `classify_memory_relation()` 추가 |
| DB 연결 안정성 | `get_db_connection()` 컨텍스트 매니저의 실패 트랜잭션 롤백 가드, `pgvector` 파라미터 바인딩 시 `::vector` 명시적 캐스팅 반영 |
| 동시성 제어 | `processing_lock`(Busy-Dropping), `MemoryWriteWorker` 단일 소비자 큐를 통한 기억 적재 순서 보장 구조 명문화 |

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
│ [로컬 엣지 게이트웨이: Python Fast Engine (PC, server/main.py)]│
│  - Busy-Dropping Mutex: processing_lock으로 중복 인입 차단   │
│  - STT: faster-whisper (base, int8, CPU 구동, ~1.0s)        │
│  - Intent Engine: 5단계 분기 (호출어/시간룰/부정어/비전/텍스트)│
│  - RAG Memory: pgvector 검색 + QuestionDetector 동적 완화    │
│  - MemoryWriteWorker: 순차 큐 기반 비동기 Dedup/Invalidation │
│  - Vision Gesture: Google MediaPipe Hands (손하트 실시간)    │
│  - AI Brain: Gemini 3.6 Flash (thinking_budget=1, Auto-heal) │
│  - TTS: Edge-TTS (비동기 합성 ➔ 청크 스트리밍 + 에코 가드)   │
└──────────────┬──────────────────────────────────────────────┘
               ▼ (ThreadedConnectionPool + Rollback Guard)
┌─────────────────────────────────────────────────────────────┐
│ [데이터베이스: PostgreSQL]                                   │
│  - emotion_presets: 감정별 LED RGB 및 복귀 지속시간(SSOT)   │
│  - interaction_logs: 트리거/감정/발화/레이턴시 텔레메트리    │
│  - user_long_term_memory: pgvector + Invalidation 소프트 딜리트│
└─────────────────────────────────────────────────────────────┘

```

---

## 2. 하드웨어 구성 및 핀 배정 (FireBeetle 2)

> v2.1과 동일 (변경 없음). 현재 소프트웨어 구현은 PC 프로토타입(웹캠/PC 마이크·스피커) 단계이며, 아래 핀 배정은 무선 패키징 단계의 목표 하드웨어 규격입니다.

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

> `server/repositories/schema.sql` 실제 운영 스키마 기준 (v2.1 초안의 컬럼명/테이블 구조 오차 정정). `user_long_term_memory`는 서버 재기동/스키마 재적용 시에도 절대 `DROP`하지 않으며, 이력 컬럼은 기존 테이블에 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`로만 증분 반영합니다.

```sql
-- 1. 감정별 고정 RGB 및 기본 동작 시간 프리셋 테이블 (SSOT)
CREATE TABLE IF NOT EXISTS emotion_presets (
    emotion VARCHAR(30) PRIMARY KEY,
    r INT NOT NULL CHECK (r BETWEEN 0 AND 255),
    g INT NOT NULL CHECK (g BETWEEN 0 AND 255),
    b INT NOT NULL CHECK (b BETWEEN 0 AND 255),
    duration NUMERIC(4, 1) NOT NULL DEFAULT 4.0,
    description TEXT
);
-- 14종 감정 프리셋 시드: NEUTRAL/HAPPY/PROUD/CURIOUS/SAD/ANGRY/HEART_EYES/
-- SURPRISED/SCARED/DIZZY/TIRED/SLEEPING/FOCUS/ERROR

-- 2. 텔레메트리 상호작용 로그 테이블 (KST 타임존 적용)
CREATE TABLE IF NOT EXISTS interaction_logs (
    id SERIAL PRIMARY KEY,
    trigger_type VARCHAR(50) NOT NULL,
    prompt TEXT,
    emotion VARCHAR(30) NOT NULL REFERENCES emotion_presets(emotion),
    speech TEXT,
    led_r INT NOT NULL,
    led_g INT NOT NULL,
    led_b INT NOT NULL,
    duration NUMERIC(4, 1) NOT NULL,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 3. 장기 기억(RAG) 저장소
-- ⚠️ 절대 DROP하지 않는다 (서버 재기동/스키마 재적용 시 사용자 기억 유실 방지).
--    all-MiniLM-L6-v2 로컬 임베딩 기준 384차원 고정.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS user_long_term_memory (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL DEFAULT 'primary_user',
    fact_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_memory_embedding
ON user_long_term_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 4. [v2.2 신규] 장기 기억 모순 해결(Invalidation) 지원 컬럼
-- ⚠️ 기존 행을 DELETE하지 않고 is_active 플래그로만 비활성화한다 (기억 유실 방지, 이력 조회/복구 여지 보존).
ALTER TABLE user_long_term_memory
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS superseded_by BIGINT REFERENCES user_long_term_memory(id),
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_user_memory_active
ON user_long_term_memory (user_id) WHERE is_active = TRUE;
```

### 스키마 설계 노트
- `interaction_logs`는 STT/RAG/LLM/TTS 구간별 레이턴시를 분리 저장하지 않고 총 소요시간 `latency_ms` 단일 컬럼으로 적재하며, 구간별 분해는 콘솔 로그(`[⏱️ 속도 분석]`)로만 확인합니다.
- `idx_user_memory_active`는 `is_active = TRUE`에 대한 부분 인덱스로, RAG 검색/Dedup 조회 시 무효화된(과거) 기억을 스캔 대상에서 제외해 조회 비용을 낮춥니다.
- `superseded_by`는 자기 참조 FK로, 모순 판정 시 무효화되는 구 기억 행이 이를 대체한 신규 기억의 `id`를 가리키도록 하여 기억 계보(lineage)를 추적할 수 있게 합니다.

---

## 4. 소프트웨어 파이프라인 세부 명세

### 1) AI 두뇌 & 장기 기억(RAG) 파이프라인

* **초저지연 추론 엔진 (Gemini 3.6 Flash, `BrainService`)**:
  * `thinking_config=types.ThinkingConfig(thinking_budget=1)` 고정 유지. `0` 주입 시 `400 INVALID_ARGUMENT`, 미설정 시 기본 추론 프로세스로 약 1.8초 지연이 추가되므로 반드시 `1`을 유지 (절대 훼손 금지 가드레일).
  * `max_output_tokens=1024` 이상 유지하여 비전 모드 한글 멀티바이트 문자열 잘림 방어.
  * **`_extract_json_object()` 공용 Auto-healing 파서**: 마크다운 코드블록 제거 → 바깥쪽 중괄호 슬라이싱(`find`/`rfind`) → 잘린 문자열 자동 괄호 보정(`..."}`) 3단계로 구성되며, `parse_action_json`(감정/대사 추론)과 `classify_memory_relation`(기억 모순 판정) 양쪽이 이 로직을 공유. `json.loads` 단순 교체나 SDK `response_schema` 대체 금지.
  * **재시도/장애 대응**: `503` 응답 시 `MAX_RETRIES` 한도 내 1초 대기 후 재시도, `429` 쿼터 초과 시 즉시 `DEFAULT_LLM_FALLBACK`으로 안전 탈출.
* **장기 기억 검색 (Retrieval, `_retrieve_memory_context`)**:
  * 로컬 경량 임베딩(FastEmbed/ONNX `all-MiniLM-L6-v2`, ~0.05초) → `pgvector` 코사인 유사도 상위 `Top-K=2` 문장만 프롬프트에 `[참고 기억] ...` 형태로 주입.
  * **[v2.2 신규] 의문문 동적 임계값 완화 (`QuestionDetector`)**: 서술문("난 포도 좋아")과 그에 대응하는 질문("무슨 과일 좋아한다고 했지?") 사이 임베딩 비대칭성으로 인해 일반 임계값(`RAG_SIMILARITY_THRESHOLD=0.80`)을 통과하지 못하는 현상을 해소하기 위해, 정규식 기반(`server/services/question_detector.py`) 의문형 어미(`지/까/니/냐/나`)·의문사(`뭐/무엇/무슨/누구/언제/어디/어떻게/왜/얼마`)·회상 질의(`기억나`) 감지 시에만 완화된 `RAG_QUESTION_SIMILARITY_THRESHOLD=0.68`을 적용. LLM 호출 없는 순수 정규식 판별로 지연 추가 없음. `RAG_TOP_K=2` 상한이 그대로 유지되어 완화로 인한 과도한 노이즈 주입은 방어됨.
* **장기 기억 적재 (Write, `MemoryService` + `MemoryWriteWorker`)**:
  * **비동기 적재**: 대화에서 기억할 정보 추출 및 DB 저장은 반드시 로봇 TTS 완료 후 Background Task로 처리 (대화 턴 지연 영향 0).
  * **[v2.2 신규] 순차 적재 큐 (`MemoryWriteWorker`)**: 연속 발화("사과 좋아해" → "역시 사과 싫어")마다 개별 스레드를 띄우면 모순 무효화 순서가 실제 발화 순서와 어긋나는 경쟁 상태가 발생할 수 있어, 단일 소비자 스레드가 큐에 쌓인 적재 요청을 발화 순서 그대로 순차 처리. `submit()`은 `queue.put_nowait`로 즉시 반환되어 지연 없음.
  * **[v2.2 신규] 근접 중복 방지 (Dedup, `RAG_DEDUP_SIMILARITY_THRESHOLD=0.85`)**: 신규 발화 임베딩과 가장 가까운 기존 기억의 유사도가 0.85 이상이면(예: "나 커피 좋아해" ≈ "커피 좋아해") 저장을 스킵해 사실상 동일한 문장의 중복 적재를 방지.
  * **[v2.2 신규] 모순 해결(Invalidation) 깔때기**: `RAG_CONFLICT_CANDIDATE_THRESHOLD(0.55)` 이상 `RAG_DEDUP_SIMILARITY_THRESHOLD(0.85)` 미만 구간의 후보만 경량 LLM Reflection(`BrainService.classify_memory_relation`)에 전달해 모순(선호/상태 변경) 여부를 판정. 완전 무관(<0.55)한 기억까지 매번 LLM에 태우지 않도록 후보 구간을 좁혀 호출 비용을 최소화. 모순으로 판정된 기존 기억은 물리 삭제 대신 `is_active=FALSE` + `superseded_by=<신규 id>`로 소프트 딜리트해 이력 추적 가능.
* **5단계 의도 분석 엔진 (`IntentService`, 엄격 고정 순서)**:
  1. **호출어 정규화**: "멘티오는", "멘티오가" 등 후행 조사 제거.
  2. **룰 기반 즉시 처리**: "몇 시야?", "지금 몇 시" 등 시간 질의 감지 시 LLM 호출을 건너뛰고 로컬 시스템 시계(`datetime.now()`) 기반 즉시 음성 안내(`build_time_response`). "몇 시간"(소요 시간) 등과의 오탐 방지 정규식 포함.
  3. **시각 부정어(Veto) 차단**: "보지 마", "사진 찍지 마" 검출 시 최우선으로 `VOICE_CHAT`으로 강제 우회.
  4. **순수 시각 요구 감지**: "봐봐", "사진 찍어줘" 검출 시에만 480p 단발 스냅샷 캡처 및 `VOICE_VISION` 모드 진입.
  5. **기본 Fallback**: 일상 발화는 모두 초고속 순수 텍스트 대화(`VOICE_CHAT`)로 안전 수렴.
* **이원화 비전 처리**:
  * **온디맨드 스냅샷(VLM)**: 영상 상시 스트리밍을 배제하고 필요 시에만 1장 전송하여 대역폭 고갈 및 발열 차단.
  * **로컬 제스처 인식**: PC 웹캠 기반 MediaPipe Hands로 손하트 감지 시 즉각 `HEART_EYES` 및 핫핑크 LED(`[255, 20, 120]`) 표출, `COOLDOWN_SECONDS` 쿨다운 적용.

### 2) 데이터베이스 연결 안정성 & 동시성 가드레일 [v2.2 신규 섹션]

* **`ThreadedConnectionPool` + 실패 트랜잭션 롤백 가드 (`server/repositories/connection.py`)**:
  * `get_db_connection()` 컨텍스트 매니저 내에서 예외 발생 시 커넥션을 실패 상태 그대로 풀에 반납하면, 다음 대여자의 정상 쿼리까지 `InFailedSqlTransaction`으로 연쇄 실패시키는 문제가 있어 반드시 `conn.rollback()` 후 `finally`에서 `putconn()`으로 반납.
  * `init_db_pool()`/`close_db_pool()`로 서버 기동/종료 시점에 커넥션 풀을 명시적으로 생성·해제.
* **`pgvector` 명시적 캐스팅**: `INSERT`/`SELECT` 쿼리 파라미터 바인딩 시 `%s::vector`로 명시적 캐스팅해 psycopg2 파라미터 타입 추론 오류를 방지 (`memory_repository.py`의 `insert_memory`, `search_similar_memories`).
* **Busy-Dropping Mutex (`server/main.py`)**: `processing_lock`으로 보호되는 전역 `is_processing` 플래그를 통해 로봇이 이미 생각/발화 중일 때 인입되는 음성 입력을 즉시 폐기, 이중 처리 경쟁 상태 방지.
* **순차 적재 큐 (`MemoryWriteWorker`)**: 위 4-1절 참조. DB 기록 순서와 실제 발화 순서의 일치를 보장하는 상호 배제 구조.

### 3) 음성 처리 & 스마트 인터랙션 제어
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

### 4) 감성 동기화 & 물리 반응
* **발화-UI 0ms 동기화 및 타이머 앵커링**:
  * 음성 첫 음절 출력 직전에 OLED 표정과 LED 색상을 즉시 전환.
  * 표정 타이머는 발화 완료 시점부터 카운트다운을 시작하여 설정된 지속시간(4.0~5.0s) 유지 후 `NEUTRAL` 자동 복귀.
* **6축 모션 및 물리 인터랙션 (MPU-6050 & Touch)**:
  * **흔들림 감지**: 임계치 초과 시 `DIZZY(어지러움)` 표정 및 회전 연출.
  * **낙하 감지**: 가속도 합 $0G$ 근접 감지 시 `SURPRISED(놀람)` 표정 연출.
  * **머리 쓰다듬기**: 정전식 터치 단발 탭 감지 시 `HAPPY(기쁨)` 반응 및 골드 LED 점등.

### 5) 데스크 유틸리티 & 환경 반응
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
