# 🤖 Project Mentio (멘티오)

> **DFRobot FireBeetle 2 ESP32-S3 기반 계층형 엣지-클라우드 AI 데스크 감성 반려로봇**

Mentio는 단순한 음성 비서를 넘어 사용자와의 감정 동기화, 물리 반응, 능동적 기억 회상을 제공하는 피지컬 데스크 메이트입니다.  
ESP32-S3 마이크로컨트롤러, 로컬 파이썬 엣지 게이트웨이, 그리고 클라우드 LLM을 결합한 계층형 아키텍처를 채택하여 초저지연 상호작용과 안정적인 기억 보존을 동시에 구현합니다.

> 🚧 **Project Status: Active Development (WIP)**  
> 현재 코어 백엔드(STT/RAG/LLM/TTS) 및 DB 파이프라인 검증을 마치고, **ESP32-S3 하드웨어 통신 게이트웨이 및 임베디드 펌웨어 연동을 진행 중**입니다. 지속적으로 기능이 업데이트되고 있습니다.

---

## ✨ 핵심 기능

* **초저지연 감성 대화 파이프라인**: 로컬 faster-whisper(STT)와 Gemini 3.6 Flash(`thinking_budget=1`)를 연동하여 E2E 약 2초대의 빠른 응답 속도를 보장합니다.
* **자기 치유(Auto-healing) 액션 파서**: LLM의 응답에서 감정과 대사를 실시간 분리 추출하며, 잘린 JSON이나 마크다운 노이즈를 자체 복구합니다.
* **RAG 기반 3단계 장기 기억 시스템**:
  * **근접 중복 방지 (Dedup)**: 코사인 유사도 0.85 이상 발화는 불필요한 저장 생략.
  * **모순 해결 (Invalidation)**: 0.55~0.85 유사도 구간의 기억은 경량 LLM Reflection을 통해 모순 여부를 판정하고, 기존 기억을 소프트 딜리트(`is_active=FALSE`, `superseded_by`)로 무효화.
  * **의문문 검색 최적화**: 의문사 및 질문 어미 감지 시 유사도 임계값을 0.68로 동적 완화하여 회상 정확도 향상.
* **0ms 발화-UI 동기화**: TTS 음성 출력 직전 OLED 표정과 NeoPixel LED 색상을 즉각 전환하며, 발화 종료 시점부터 복귀 타이머를 앵커링합니다.
* **5단계 의도 분석 엔진**: 호출어 정규화 $\rightarrow$ 룰 기반 시간 즉시 응답 $\rightarrow$ 시각 부정어 차단 $\rightarrow$ 비전 모드 진입 $\rightarrow$ 일반 텍스트 대화로 안전하게 수렴.
* **다중 사용자 화자 식별 (가족 프로필)**: Resemblyzer 256차원 d-vector 기반 1:N 화자 식별로 가족 구성원을 목소리만으로 구분하고, 화자별로 장기 기억·프로필 요약·호칭을 격리합니다. 미등록 화자(제3자/TV 소리)는 파이프라인 진입을 차단합니다. (아래 **화자 식별 & 가족 프로필** 섹션 참고)

---

## 🎙️ 화자 식별 & 가족 프로필

화자 식별은 VAD 직후·STT 이전에 수행되며, 판정된 `user_id`가 RAG 검색 → 프롬프트 페르소나 → 장기 기억 적재까지 일관되게 전달됩니다. 판별 로직은 모두 LLM 호출 없는 로컬 연산/정규식 룰로 동작해 대화 지연을 추가하지 않습니다.

| 기능 | 동작 | 관련 설정 |
| :--- | :--- | :--- |
| **실제 발화 길이 기반 커트라인** | VAD 사전 버퍼·종료 무음(약 1.1초)을 트리밍한 **실제 발화 길이** 기준으로 임계값을 분기합니다. 일반 발화 `0.65`, 1.0초 이하 단문(“응”, “네”) `0.60`. | `SPEAKER_VERIFICATION_THRESHOLD`, `SPEAKER_SHORT_UTTERANCE_*` |
| **마진(Margin) 모호성 가드** | Top-1/Top-2 화자 원점수 차이가 `0.04` 미만이면 “모호”로 판정하고, 해당 발화는 **장기 기억 적재를 억제**해 형제/자매 간 개인 기억 오염을 막습니다. | `SPEAKER_MARGIN_THRESHOLD` |
| **대화 세션 락 (Session Lock)** | 마지막으로 통과한 화자를 120초간 “현재 대화 상대”로 유지하고, 판정 점수에 `+0.03` 가산 및 모호 시 우선권을 부여합니다. 다른 화자가 마진 이상 확실히 앞서면 세션이 자동 교체됩니다. 별도로 직전 통과 10초 이내 발화는 임계값 미달이어도 소프트패스로 통과시킵니다. | `SPEAKER_SESSION_TIMEOUT_SEC`, `SPEAKER_SESSION_BONUS`, `SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC` |
| **대화형 호칭 정정** | “나 민수인데?”, “나 민수 아니고 지훈이야”처럼 발화 **전체**가 정정 표현이고 이름이 다른 등록 화자 1명과 일치하면, 세션 화자를 즉시 전환하고 “앗, 지훈아 미안해!”로 응답합니다. 정정 발화는 기억 적재/EMA 대상에서 제외됩니다. | — |
| **대화형 음성 온보딩** | “내 목소리 등록해줘” → 이름 1회 + 샘플 문장 2회 발화를 받아 **임베딩 평균 + L2 정규화**로 기준 벡터를 등록합니다. 샘플은 실제 발화 1.5초 이상만 인정하며, 30초 무응답 시 세션이 자동 만료됩니다. | `SPEAKER_ENROLLMENT_*` |
| **중복 등록 가드** | 기존 화자와 유사도 `≥ 0.85`면 재등록으로 보고 **기존 프로필 목소리만 갱신**(새 `user_id` 미발급), `0.75~0.85`면 유사 화자 경고 멘트와 함께 신규 등록을 허용합니다. | `SPEAKER_ENROLLMENT_DUPLICATE_THRESHOLD`, `SPEAKER_ENROLLMENT_SIMILAR_WARNING_THRESHOLD` |
| **EMA 점진적 임베딩 갱신** | 원점수 `≥ 0.82`, 비모호, 실제 발화 1.0초 이상인 고신뢰 발화만 `new = normalize(0.95·current + 0.05·input)`으로 기준 벡터를 천천히 갱신합니다. DB 쓰기는 TTS 완료 후 `MemoryWriteWorker` 순차 큐에서 처리됩니다. | `SPEAKER_EMA_*` |
| **화자 캐시 Hot Reload** | 백그라운드 워커가 10초마다 `speaker_profiles`의 경량 지문(`MAX(updated_at)`, 전체/활성 행 수)만 조회해, 변경 시에만 캐시를 원자적으로 교체합니다(**Zero-Downtime**, 재기동 불필요). 발화 처리/온보딩 중에는 다음 주기로 보류합니다. | `SPEAKER_CACHE_POLL_INTERVAL_SEC` |

화자 등록 방법은 세 가지입니다:
- **음성 온보딩**: 로봇에게 “내 목소리 등록해줘”라고 말하기
- **CLI**: `python scripts/enroll_speaker.py --user-id dad --name 아빠`
- **관리 대시보드**: `streamlit run web/app.py` (가족/화자 프로필 및 RAG 기억 관리, 변경 사항은 Hot Reload로 즉시 반영)

---

## 🎭 감정 상태 프리셋 (14 Expressions)

로봇은 대화 맥락에 따라 14가지 감정을 자율 판단하며, OLED 눈동자 애니메이션과 NeoPixel 색상을 즉각 동기화합니다.

| 감정 (Emotion) | NeoPixel RGB | 기본 지속 시간 | 연출 및 상호작용 맥락 |
| :--- | :--- | :--- | :--- |
| `NEUTRAL` | `[255, 255, 255]` | 4.0s | 기본 대기 상태 (소프트 화이트) |
| `HAPPY` | `[255, 200, 0]` | 4.0s | 긍정/칭찬/기쁨 반응 (골드 옐로우) |
| `HEART_EYES` | `[255, 20, 120]` | 5.0s | 손하트 제스처 인식 및 애정 표현 (핫핑크) |
| `PROUD` | `[0, 200, 255]` | 4.0s | 칭찬 수용 및 성취감 (스카이 블루) |
| `CURIOUS` | `[180, 0, 255]` | 4.0s | 사용자 질문 및 탐색 (바이올렛) |
| `SAD` | `[0, 50, 255]` | 4.0s | 위로/공감 모드 (딥 블루) |
| `ANGRY` | `[255, 0, 0]` | 4.0s | 불만 및 거절 반응 (레드) |
| `SURPRISED` | `[255, 100, 0]` | 4.0s | 급작스러운 낙하 감지(0G) 및 깜짝 놀람 |
| `DIZZY` | `[150, 0, 255]` | 4.0s | 6축 자이로 기반 심한 흔들림 감지 시 연출 |
| `TIRED` | `[50, 50, 50]` | 4.0s | 고온다습 환경 지속 감지 시 피로 표정 |
| `SLEEPING` | `[0, 0, 0]` | 4.0s | 암전 지속 시 절전/수면 모드 진입 |
| `FOCUS` | `[0, 255, 150]` | 4.0s | 뽀모도로 타이머 가동 중 집중 상태 |
| `SCARED` | `[100, 0, 0]` | 4.0s | 공포 반응 및 경계 |
| `ERROR` | `[255, 0, 0]` | 4.0s | 시스템 예외 상황 알림 |

---

## 🖐️ 물리 반응 및 환경 감지 인터랙션

* **6축 모션 감지 (MPU-6050)**:
  * **어지러움**: 책상 흔들림이나 들어 올림 감지 시 `DIZZY` 표정 전환 및 눈동자 회전 연출.
  * **낙하 감지**: 가속도 합 0G 근접 감지 시 즉각 `SURPRISED` 표정 및 경고 연출.
* **정전식 정밀 터치 (ESP32-S3 Touch12)**:
  * **머리 쓰다듬기 (단발 탭)**: `HAPPY` 감정 표현 및 골드 LED 점등.
  * **대화 음소거 (3초 롱터치)**: Mute ON/OFF 토글, 취침(`Zzz`) 아이콘 표출 및 마이크 버퍼 폐기.
* **환경 센싱 반응**:
  * **AHT20 온습도 센서**: 밀폐 환경 고온다습 지속 감지 시 `TIRED` 표정과 함께 환기 제안.
  * **CDS 조도 센서**: 실내 암전 감지 시 디스플레이 슬립 및 취침 모드 전환.

---

## 🏗️ 시스템 아키텍처

```text
[물리 환경 / 사용자]
  │ (I2S 마이크 / OV2640 카메라 / 터치 / 6축 모션 / 온습도)
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
│  - STT: faster-whisper (base, int8, CPU 구동)               │
│  - Speaker ID: Resemblyzer 1:N 식별 (마진/세션 락/EMA)      │
│  - SpeakerCacheSyncWorker: 10초 주기 화자 캐시 Hot Reload   │
│  - Intent Engine: 5단계 분기 (룰 기반 시간, 시각 제어 등)   │
│  - RAG Memory: FastEmbed + pgvector + QuestionDetector      │
│  - MemoryWriteWorker: 순차 큐 (기억 적재 + 화자 EMA 갱신)   │
│  - AI Brain: Gemini 3.6 Flash (Auto-healing Parser)         │
│  - TTS: Edge-TTS 스트리밍                                   │
└──────────────┬──────────────────────────────────────────────┘
               ▼ (ThreadedConnectionPool + Rollback Guard)
┌─────────────────────────────────────────────────────────────┐
│ [데이터베이스: PostgreSQL + pgvector]                        │
│  - emotion_presets: 감정별 LED RGB 및 복귀 지속시간         │
│  - user_long_term_memory: pgvector(384차원) + 기억 계보 추적│
│  - user_profile_summary: 화자별 페르소나 요약               │
│  - speaker_profiles: 화자 임베딩(256차원) + 호칭            │
│  - interaction_logs: 텔레메트리 로그                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 하드웨어 핀아웃 매핑

| 기능 블록 | 부품명 | 인터페이스 / 핀 배정 |
| :--- | :--- | :--- |
| **비전 입력** | OV2640 카메라 | 온보드 FPC 전용 슬롯 직결 |
| **공용 I2C** | 0.96"/1.3" OLED, AHT20, MPU-6050 | GPIO 1 (SDA) / GPIO 2 (SCL) |
| **I2S 스피커 출력** | MAX98357A DAC 앰프 | GPIO 6 (BCLK) / GPIO 7 (LRC) / GPIO 8 (DIN) |
| **I2S 마이크 입력** | INMP441 디지털 마이크 | GPIO 9 (SCK) / GPIO 10 (WS) / GPIO 11 (SD) |
| **감정 상태 LED** | WS2812B NeoPixel | GPIO 5 (D2) |
| **주변 조도 감지** | CDS 광센서 모듈 | GPIO 17 (A0 / ADC1) |
| **정전식 터치** | 정전식 터치 패드 | GPIO 12 (Touch12) |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 종속성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (`.env`)

프로젝트 루트 디렉토리에 `.env` 파일을 구성합니다:

```env
GEMINI_API_KEY=your_gemini_api_key   # 필수 (미설정 시 기동 실패)

# PostgreSQL 접속 정보 (기본값: localhost / 5432 / mentio_db / postgres / postgres)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mentio_db
DB_USER=postgres
DB_PASSWORD=postgres

# 화자 식별 (선택, 괄호는 기본값)
SPEAKER_VERIFICATION_ENABLED=true         # (true) false면 식별 스킵, 전원 기본 사용자로 통과
SPEAKER_VERIFICATION_THRESHOLD=0.65       # (0.65) 일반 발화 식별 임계값
SPEAKER_CACHE_POLL_INTERVAL_SEC=10        # (10) 화자 캐시 Hot Reload 폴링 주기(초)
SPEAKER_EMA_ENABLED=true                  # (true) 고신뢰 발화 기반 EMA 임베딩 갱신
```

#### 화자 식별 튜닝 상수 (`config/settings.py`)

환경변수가 아닌 코드 상수로, 실기 로그 기반 튜닝 히스토리가 `config/settings.py` 주석에 기록되어 있습니다.

| 설정 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `SPEAKER_SHORT_UTTERANCE_MAX_SEC` | `1.0` | 무음 트리밍 후 실제 발화가 이 길이(초) 이하면 단문으로 간주 |
| `SPEAKER_SHORT_UTTERANCE_THRESHOLD` | `0.60` | 단문 전용 완화 임계값 |
| `SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC` | `10.0` | 직전 통과 후 이 시간 이내면 임계값 미달이어도 소프트패스 |
| `SPEAKER_MARGIN_THRESHOLD` | `0.04` | Top-1/Top-2 차이가 이 값 미만이면 모호 판정 (기억 적재 억제) |
| `SPEAKER_SESSION_TIMEOUT_SEC` | `120.0` | 세션 락 유지 시간 |
| `SPEAKER_SESSION_BONUS` | `0.03` | 세션 화자 판정 가산점 (반환 similarity는 원점수) |
| `SPEAKER_EMA_MIN_SIMILARITY` | `0.82` | EMA 갱신 허용 최소 원점수 |
| `SPEAKER_EMA_ALPHA` | `0.05` | EMA 반영 비율 |
| `SPEAKER_EMA_MIN_SPEECH_SEC` | `1.0` | EMA 갱신 최소 실제 발화 길이(초) |
| `SPEAKER_ENROLLMENT_SESSION_TIMEOUT_SEC` | `30.0` | 온보딩 무응답 자동 만료 시간 |
| `SPEAKER_ENROLLMENT_REQUIRED_SAMPLES` | `2` | 이름 발화 이후 수집하는 샘플 문장 수 |
| `SPEAKER_ENROLLMENT_MIN_SAMPLE_SEC` | `1.5` | 샘플로 인정하는 최소 실제 발화 길이(초) |
| `SPEAKER_ENROLLMENT_MIN_NAME_SAMPLE_SEC` | `0.5` | 이름 발화를 평균에 포함시키는 최소 실제 발화 길이(초) |
| `SPEAKER_ENROLLMENT_DUPLICATE_THRESHOLD` | `0.85` | 이 이상이면 기존 화자 재등록 → 기존 프로필 목소리 갱신 |
| `SPEAKER_ENROLLMENT_SIMILAR_WARNING_THRESHOLD` | `0.75` | 이 이상 ~ 0.85 미만이면 유사 화자 경고 후 신규 등록 |
| `CAMERA_REOPEN_AFTER_FAILURES` | `100` | 카메라 프레임 연속 실패 시 재오픈 주기 (실패 중에는 음성 전용 모드) |

### 3. 데이터베이스 초기화

PostgreSQL에 접속하여 pgvector 확장을 활성화하고 운영 스키마를 적용합니다:

```bash
psql -U postgres -d mentio_db -f server/repositories/schema.sql
```

### 4. 서버 구동 및 테스트

```bash
# 로컬 게이트웨이 서버 실행
python -m server.main

# 전체 단위 테스트 실행
pytest tests/ -v
```
---

## 📂 프로젝트 구조

```text
mentio/
├── config/settings.py         # 환경 변수 및 공통 설정 (임계값 튜닝 히스토리 포함)
├── docs/                      # 시스템 명세서 및 기술 설계 문서
├── scripts/enroll_speaker.py  # CLI 화자 등록 스크립트
├── server/
│   ├── adapters/              # 오디오 입출력 어댑터 (PC 사운드 ↔ ESP32 스트림 교체 지점)
│   ├── repositories/          # DB 커넥션 풀, pgvector 쿼리 및 SQL 스키마
│   ├── schemas/               # Pydantic DTO (데이터 검증 계층)
│   ├── services/              # AI Brain, Intent, RAG Memory, Speaker ID, Voice Enrollment, Vision
│   ├── workers/               # AIWorker, MemoryWriteWorker(순차 큐), SpeakerCacheSyncWorker(Hot Reload)
│   └── main.py                # 로컬 엣지 게이트웨이 진입점
├── web/app.py                 # Streamlit 관리 대시보드 (가족 프로필 / RAG 기억)
└── tests/                     # 단위 및 통합 테스트 스위트 (하드웨어 테스트는 --run-hardware로 opt-in)
```

---

## 📄 License

본 프로젝트는 [MIT License](LICENSE)를 따릅니다.


