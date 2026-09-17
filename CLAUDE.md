# Project Mentio (멘티오) - 개발 가이드 및 엔지니어링 가드레일

## 1. 프로젝트 개요 & 문서 참조
- **정체성**: Anki Vector / Cozmo 스타일의 피지컬 AI 데스크 반려로봇. 단순한 음성 비서가 아닌, 감정 동기화와 고유 인터랙션을 갖춘 탁상형 감성 메이트를 지향한다.
- **아키텍처**: 계층형 엣지-클라우드 (ESP32-S3 엔드포인트 + PC 로컬 엣지 게이트웨이 + Gemini 3.6 Flash 클라우드).
- **상세 명세서**: 전체 기능 스펙, DB 스키마, 하드웨어 핀아웃 매핑의 세부 내용은 `docs/system_spec_v2.2.md`를 필히 참고할 것.

---

## 2. ⚠️ 절대 훼손/수정 금지 핵심 가드레일 (Critical Invariants)

코드 리팩토링, 신규 기능 추가, 최적화 작업 시 아래 확정된 설정과 아키텍처는 **절대 단순화하거나 임의로 수정/되돌리지 말 것**:

### 1) BrainService (`server/services/brain_service.py`)
- **Thinking Budget 설정**: 반드시 `thinking_config=types.ThinkingConfig(thinking_budget=1)`을 유지할 것.
  - `0`을 주입하면 Google API에서 `400 INVALID_ARGUMENT` 오류가 발생함.
  - 설정을 누락하거나 삭제하면 기본 추론 프로세스로 인해 약 1.8초의 불필요한 지연이 추가됨.
- **출력 토큰 상한**: `max_output_tokens`는 최소 `1024` 이상 유지할 것 (비전 모드에서 한글 멀티바이트 문자열 잘림 방어).
- **Auto-healing JSON 파서 보존**: 바깥쪽 중괄호 슬라이싱(`find`/`rfind`), 마크다운 코드블록 제거, 닫는 괄호 자동 보정(`..."}`) 로직을 절대 삭제하지 말 것. 단순 `json.loads`나 SDK의 `response_schema`로 교체 금지.
- **`_extract_json_object()` 공용 헬퍼 보존**: 위 Auto-healing 파싱 로직은 `parse_action_json`(감정/대사 추론)과 `classify_memory_relation`(장기 기억 모순 판정)이 함께 공유하는 단일 헬퍼다. 각 호출부에 파싱 로직을 개별 복제하거나 어느 한쪽만 다른 방식(예: 단순 `json.loads`)으로 교체하지 말 것.

### 2) IntentService (`server/services/intent_service.py`)
의도 분석 판별 순서는 반드시 다음 5단계를 엄격히 준수할 것:
1. **호출어 및 조사 정규화**: "멘티오는", "멘티오가" 등 후행 조사 정밀 제거.
2. **룰 기반 즉시 처리**: "몇 시야?", "지금 몇 시" 등 시간 질문 감지 시 LLM 호출을 건너뛰고 로컬 시계(`datetime.now()`) 기반 0.1초 즉시 음성 안내.
3. **시각 부정어(Veto) 차단**: "보지 마", "사진 찍지 마" 감지 시 최우선으로 `VOICE_CHAT` 모드로 우회.
4. **순수 시각 요구 감지**: "봐봐", "사진 찍어줘" 검출 시에만 480p 스냅샷 캡처 및 `VOICE_VISION` 모드 진입.
5. **기본 Fallback**: 일상 발화는 모두 초저지연(E2E 2.6초대) 순수 텍스트 대화(`VOICE_CHAT`)로 안전 수렴.

### 3) 동시성 제어, 타이머 앵커링 & 상태 락
- **0ms UI 동기화**: 음성 첫 음절 출력 직전에 OLED 표정과 NeoPixel LED 색상을 선제 전환할 것.
- **타이머 앵커링**: 표정 복귀 타이머는 발화가 시작될 때가 아니라, **로봇의 스피커 출력이 완전히 끝난 시점부터 카운트다운**하여 DB에 설정된 지속시간(4~5초) 유지 후 `NEUTRAL`로 자동 복귀시킬 것.
- 서비스 전반에서 의존성 역전 원칙(DIP)과 상호 배제(Mutex) 락 구조를 유지할 것.

### 4) DB 연결 및 정합성 (`server/repositories/connection.py`, `memory_repository.py`)
- **실패 트랜잭션 롤백 가드**: `get_db_connection()` 컨텍스트 매니저는 내부에서 예외가 발생하면 반드시 `conn.rollback()`을 호출한 뒤 커넥션을 풀에 반납할 것. 롤백 없이 실패 상태 그대로 반납하면 다음 대여자의 정상 쿼리까지 `InFailedSqlTransaction`으로 연쇄 실패한다.
- **`user_long_term_memory` 테이블 보존**: 서버 재기동/스키마 재적용 시에도 절대 `DROP`하지 말 것. 신규 컬럼 추가는 반드시 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`로만 증분 반영하여 기존 사용자 기억을 유실시키지 않는다.

---

## 3. 백엔드 & RAG(장기 기억) 개발 가이드라인

### 오디오 및 입출력 추상화 (DIP 원칙 준수)
- PC 마이크/스피커(`sounddevice`) 로직을 `brain_service`, `intent_service` 등 도메인 서비스 계층에 직접 하드코딩하지 말 것.
- 사운드 입출력은 반드시 인터페이스/어댑터 뒤로 추상화하여, 향후 **"PC 사운드 입출력"에서 "ESP32 WebSocket 스트리밍 입출력"으로 교체할 때 핵심 비즈니스 로직 수정이 0줄**이어야 함.

### RAG(장기 기억) 파이프라인 규칙
- **저지연 로컬 임베딩**: 대화 중 외부 API 호출을 배제하고 로컬 경량 모델(FastEmbed 또는 ONNX 기반 `all-MiniLM-L6-v2` 계열)을 사용하여 벡터 변환 시간을 0.05초 이내로 유지할 것.
- **비동기 메모리 적재**: 대화에서 기억할 정보 추출 및 `pgvector` 저장은 **반드시 로봇의 음성 응답(TTS)이 완료된 후 Background Task(비동기 워커)**로 처리할 것. 대화 턴 중 DB 저장을 동기로 실행해 지연을 유발하지 말 것.
  - **순차 적재 큐 필수**: 적재 요청을 위해 `threading.Thread`를 직접 생성하지 말 것. 반드시 `MemoryWriteWorker.submit()`을 통해 단일 소비자 순차 큐에 위임한다. 연속 발화마다 개별 스레드를 띄우면 Invalidation 처리 순서가 실제 발화 순서와 어긋나는 경쟁 상태가 발생한다.
- **Top-K 제한**: 프롬프트 주입 시 유사도 높은 기억은 최대 2개 문장(`Top-K=2`)으로 제한하여 입력 토큰 증가로 인한 LLM 추론 지연을 원천 방어할 것.
- **근접 중복 방지 (Dedup)**: 신규 발화 임베딩과 가장 가까운 기존 기억의 유사도가 `RAG_DEDUP_SIMILARITY_THRESHOLD(0.85)` 이상이면 사실상 동일 문장으로 간주해 단순 저장 스킵한다.
- **모순 해결 (Invalidation)**: 유사도가 `0.55 <= sim < 0.85` 구간인 후보에 한해서만 경량 LLM Reflection(`BrainService.classify_memory_relation`)으로 모순(선호/상태 변경) 여부를 판정한다.
  - **삽입 선행 원칙**: 신규 기억을 먼저 `INSERT`해 `new_id`를 확보한 뒤, 모순으로 판정된 기존 기억에 `superseded_by=new_id`를 채워야 한다 (순서를 뒤집지 말 것).
  - **소프트 딜리트 전용**: 모순 판정된 기존 기억은 `is_active=FALSE` + `superseded_by=new_id`로만 무효화한다. 물리 `DELETE`는 절대 금지 (이력 추적/복구 여지 보존).
- **의문문 검색 최적화**: `QuestionDetector`가 의문형 어미/의문사/회상 질의를 감지하면 해당 조회에 한해서만 `RAG_QUESTION_SIMILARITY_THRESHOLD(0.68)`로 동적 완화한다. LLM 호출 없는 정규식 판별만 사용하여 지연을 추가하지 말 것.
- **pgvector 파라미터 바인딩**: 임베딩 벡터를 쿼리에 바인딩할 때는 반드시 `%s::vector`로 명시적 캐스팅할 것 (psycopg2 파라미터 타입 추론 오류 방지).

### 소프트웨어 아키텍처 및 디자인 패턴 (Layered Architecture)
본 프로젝트는 **계층형 클린 아키텍처(Layered Architecture)**와 **의존성 역전 원칙(DIP)**을 엄격히 따른다. 코드를 작성할 때 각 레이어의 경계를 침범하지 말 것:

1. **Schemas Layer (`server/schemas/`)**:
   - Pydantic 기반의 DTO(Data Transfer Object) 정의. 데이터 검증 및 입출력 규격 전담 (비즈니스 로직 포함 금지).
2. **Repositories Layer (`server/repositories/`)**:
   - DB(PostgreSQL/pgvector) 접근 및 SQL 쿼리 전담. 데이터 조회/적재 외의 AI 추론이나 도메인 로직 포함 금지.
3. **Services Layer (`server/services/`)**:
   - 핵심 비즈니스 로직(STT, 의도 분석, Gemini 추론, TTS 합성, 제스처 인식) 전담. 하드웨어 I/O나 특정 프레임워크에 종속되지 않는 순수 비즈니스 레이어 유지.
4. **Workers / Orchestrator Layer (`server/workers/`, `server/main.py`)**:
   - 전체 파이프라인 흐름 조율, 비동기 큐/루프 관리, 동시성 Mutex 락 및 타이머 앵커링 제어.
5. **Adapters / Drivers Layer (향후 I/O)**:
   - 외부 환경(PC 사운드카드 또는 ESP32 WebSocket 스트림)과의 통신을 서비스 레이어로부터 완전히 격리.

---

## 4. Git 워크플로우 & 안전 규칙

### 브랜치 전략
- `main`: 상용/배포 브랜치 (직접 푸시 금지).
- `develop`: 통합 개발 브랜치.
- 기능 개발: `feat/<기능명>` (예: `feat/rule-based-time`, `feat/rag-memory`)
- 버그 수정: `fix/<이슈명>`

### 커밋 메시지 규칙 (Conventional Commits)
커밋 메시지는 다음 접두사와 함께 명확한 한글로 작성할 것:
- `feat:` 새로운 기능 추가
- `fix:` 버그 수정
- `refactor:` 코드 구조 개선 (동작 변경 없음)
- `docs:` 문서 수정
- `test:` 테스트 코드 작성

### 안전 수칙
- `git push --force` 명령은 절대 실행하지 않는다.
- 코드를 수정한 후 독단적으로 커밋하지 말고, `git status`와 변경 요약을 사용자에게 보고한 뒤 커밋 승인을 받을 것.
- `.env`, 가상환경 폴더, 모델 캐시 파일은 커밋에 포함하지 않는다.

---

## 5. 검증 및 실행 명령어

Claude Code 작업 후 자체 점검 시 사용할 표준 커맨드:

```bash
# 가상환경 활성화
source venv/bin/activate  # Windows: venv\Scripts\activate

# 종속성 설치
pip install -r requirements.txt

# 서버 게이트웨이 구동
python -m server.main

# 테스트 러너 실행
pytest tests/ -v