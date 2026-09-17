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
│  - Intent Engine: 5단계 분기 (룰 기반 시간, 시각 제어 등)   │
│  - RAG Memory: FastEmbed + pgvector + QuestionDetector      │
│  - MemoryWriteWorker: 순차 큐 기반 비동기 기억 적재         │
│  - AI Brain: Gemini 3.6 Flash (Auto-healing Parser)         │
│  - TTS: Edge-TTS 스트리밍                                   │
└──────────────┬──────────────────────────────────────────────┘
               ▼ (ThreadedConnectionPool + Rollback Guard)
┌─────────────────────────────────────────────────────────────┐
│ [데이터베이스: PostgreSQL + pgvector]                        │
│  - emotion_presets: 감정별 LED RGB 및 복귀 지속시간         │
│  - user_long_term_memory: pgvector(384차원) + 기억 계보 추적│
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
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://user:password@localhost:5432/mentio_db
```

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
├── docs/                      # 시스템 명세서 및 기술 설계 문서
├── server/
│   ├── config/                # 환경 변수 및 공통 설정
│   ├── repositories/          # DB 커넥션 풀, pgvector 쿼리 및 SQL 스키마
│   ├── schemas/               # Pydantic DTO (데이터 검증 계층)
│   ├── services/              # AI Brain, Intent, RAG Memory, Audio, Vision
│   ├── workers/               # 비동기 순차 기억 적재 큐 (MemoryWriteWorker)
│   └── main.py                # 로컬 엣지 게이트웨이 진입점
└── tests/                     # 단위 및 통합 테스트 스위트
```

---

## 📄 License

본 프로젝트는 [MIT License](LICENSE)를 따릅니다.


