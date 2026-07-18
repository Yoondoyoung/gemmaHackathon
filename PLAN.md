# 온디바이스 시각 보조 내비게이션 어시스턴트 — 24h 해커톤 기획서

## Context

시각장애인(또는 시야 제한 상황의 사용자)을 위한 완전 로컬 온디바이스 보행 보조 어시스턴트.
카메라 프레임에서 YOLO26으로 장애물을, Florence-2로 표지판 텍스트를 읽어 장면 상태(Scene State)로
정리하고, 위험은 즉시 음성 경고, 사용자의 음성 질문("화장실이 어느 쪽이야?")에는 Gemma가
장면 컨텍스트를 보고 답변한다. 인터넷 불필요 = 프라이버시 + 오프라인이 핵심 셀링 포인트.

- 팀: 2명 / 시간: 24h / 장비: M1 Air 16GB / 데모: 맥북 들고 라이브, 영어

## 냉정한 타당성 평가 (설계에 반영된 결론)

| 항목 | 판정 | 근거 / 대응 |
|---|---|---|
| YOLO26n 실시간 탐지 | ✅ 실측 확인 | **MPS 58 FPS** (워밍업 후, 이 M1 Air에서 실측). 안 되면 YOLO11n 폴백 |
| Florence-2 실시간 | ❌ 불가 | 생성형 seq2seq 모델. **실측: MPS fp32 0.9초/프레임** (CPU 2.8초, fp16은 빈 출력 버그) → **2.5초 주기 + 질문 시 on-demand**로만 실행. transformers==4.49.0 필수 |
| Gemma를 매 프레임 경유 | ❌ 불가 | 경고가 5초+ 지연 → **경고는 룰베이스 템플릿, Gemma는 Q&A 전용** |
| Gemma 3 4B Q4 (Ollama) | ✅ 실측 확인 | **24 tok/s 실측**. 짧은 응답 강제하면 Q&A 왕복 3~6초 |
| STT/TTS 로컬 | ✅ 가능 | faster-whisper small.en + macOS `say`. wake word는 시간상 제외, **push-to-talk** |
| RAM 총합 | ⚠️ 빠듯 | YOLO(~0.5GB)+Florence(~1.5GB)+Gemma(~3GB)+whisper(~0.5GB) ≈ 6GB. 브라우저 닫으면 OK |
| 포지셔닝 | ⚠️ 주의 | "실시간 내비"가 아니라 "즉각 경고 + 몇 초 내 답하는 음성 어시스턴트"로 발표 |

## 아키텍처

단일 Python 프로세스, 스레드 기반. 중심에 스레드 세이프한 **SceneState** 하나.

```
카메라(OpenCV, 640x480)
 ├─ YOLO26n 루프 (~10FPS, ByteTrack 트래킹) ──┐
 ├─ Florence-2 워커 (2.5s 주기, <OCR_WITH_REGION>) ─┤→ SceneState (JSON)
 │                                            │   objects: [{label, track_id, pos(L/C/R),
 │                                            │              dist(near/med/far), status}]
 │                                            │   texts: [{content, pos, ts}]
 ├─ Alert 엔진 (룰베이스) ← SceneState 변화 감지 → TTS 큐 (즉각, <0.5s)
 └─ Q&A 루프: [Space 누름] → 녹음 → whisper STT → Gemma(Ollama)
       프롬프트 = 시스템(시각보조 역할, 2문장 제한) + SceneState JSON + 질문 → TTS 큐
디버그 오버레이: OpenCV 창에 bbox + OCR 결과 표시 (심사위원이 "시스템이 뭘 보는지" 확인용)
```

두 경로의 분리 (핵심 설계 판단):

| | 트리거 | 처리 | 레이턴시 |
|---|---|---|---|
| 장애물 경고 | YOLO가 위험 감지 (자동) | 룰베이스 + 템플릿 문장 (Gemma 미경유) | <0.5초 |
| Q&A | 사용자 push-to-talk | Gemma + SceneState 컨텍스트 | 3~6초 |

**정보 전달 정책 — "기본은 침묵, 정보는 Pull, 안전만 Push"** (TTS는 직렬 채널이라
말이 많으면 위험 정보가 묻힌다):

| 계층 | 무엇을 | 언제 |
|---|---|---|
| Push (자동) | 안전 위협만 (center + near/approaching) | 즉시, 3~5단어 템플릿, 쿨다운 |
| Announce (1회) | 새 표지판 발견 "Sign detected: EXIT, on your right" | 내용당 최초 1회만 (표지판의 존재를 모르면 질문도 못 하므로) |
| Pull (질문) | 나머지 전부 (SceneState에 조용히 축적) | 물어볼 때만, 질문에 대한 답만 |

Pull의 "질문에 대한 답만"은 시스템 프롬프트의 우선순위 규칙(1. 안전 우선 → 2. 질문만
답변·대타 금지 → 3. 2문장)으로 강제하고 tools/test_gemma.py 8케이스로 회귀 검증.

## 거리 추정 (단안 카메라 휴리스틱)

YOLO에는 거리 개념이 없으므로 세 가지 프록시를 조합한다:

1. **bbox 높이 비율 (1순위, 주 신호)** — 같은 클래스면 가까울수록 bbox가 크다.
   `bbox_h / frame_h` 기준, **클래스별 임계값** (사람/의자/자동차는 실제 크기가 다름):
   ```
   person: >0.6 → near / 0.3~0.6 → medium / <0.3 → far
   (다른 클래스는 H16 리허설에서 1m/2m/4m 실측으로 캘리브레이션)
   ```
2. **bbox 크기 변화율 (트래킹으로 공짜)** — 같은 track_id의 bbox가 커지는 중 = 접근 중.
   status 전이: `new → approaching(증가율>임계) → near(크기 임계 도달) → gone`
   → 멀리 정지한 물체는 경고 안 함, 다가오는 것만 경고 (오탐 억제)
3. **bbox 하단 y좌표 (보조 확인용)** — 바닥 접점이 화면 하단에 가까울수록 가까움.
   단, 맥북 들고 걸으면 카메라 기울기가 변해 불안정 → "하단 15% 진입 = 매우 가까움" 확인용으로만.

**Alert 룰 (의사코드):**
```python
near     = bbox_h / frame_h > NEAR_THRESH[label]
closing  = 같은 track_id의 bbox_h 증가율 > 임계값
direction = bbox 중심 x → left / center / right
경고 발동 = (near or closing) and direction == center
           and track_id별 쿨다운(5s) 통과 and 전역 발화 제한(2s당 1회) 통과
```

프로덕션이라면 단안 깊이추정 모델(Depth Anything V2 small)을 붙이는 게 정석이나
24h 내에는 RAM/복잡도 비용이 커서 제외 (P2, 심사 Q&A 답변용으로만 준비).

## 확정 기술 스택

| 역할 | 선택 | 폴백 |
|---|---|---|
| 객체 탐지 | YOLO26n (ultralytics, MPS, ByteTrack) | YOLO11n |
| OCR/텍스트 | Florence-2-base (transformers) | Apple Vision OCR (pyobjc) — Florence가 MPS에서 깨지면 CPU 실행 먼저 시도 |
| LLM | Gemma 3 4B instruct QAT — `ollama pull gemma3:4b` | gemma3n:e2b (RAM 압박 시) |
| STT | faster-whisper `small.en`, push-to-talk | 텍스트 입력 |
| TTS | macOS `say` (subprocess, 논블로킹) | — |
| 카메라/UI | OpenCV | — |

## 추가 고려사항 (실전 리스크)

1. **카메라 방향 문제 (중요)** — 맥북 내장 캠은 화면 위에서 사용자를 향한다. 전방을 찍으려면
   **화면이 바깥(심사위원 쪽)을 향하게 들어야** 한다. 이건 오히려 장점으로 쓸 수 있음:
   발표자는 음성만 듣고(실사용자와 동일 조건), 심사위원은 화면의 디버그 오버레이로
   "시스템이 뭘 보는지" 실시간 확인. 대안: iPhone Continuity Camera를 목걸이/가슴 거치
   (셋업 리스크 있으니 H16 리허설에서 둘 다 테스트 후 결정).
2. **TTS 발화 충돌** — Gemma 답변 발화 중 경고가 발생하면? 우선순위 큐 + **경고는 현재
   발화 중인 `say` 프로세스를 kill하고 즉시 발화** (안전 > 답변). 답변은 잘린 지점부터 재개하지 않고 폐기.
3. **경고 스팸** — 해커톤 행사장은 사람이 많아 "person ahead"가 쉴 새 없이 울릴 수 있음.
   track_id별 쿨다운(5s) + 전역 발화 제한(2s당 1회) + center만 경고 + **데모 동선을 한적한
   복도로 선정**. 리허설 때 실제 환경에서 임계값 튜닝.
4. **자기 목소리 루프** — TTS 발화 중 마이크 녹음이 겹치면 whisper가 자기 TTS를 받아적음.
   push-to-talk이라 대부분 회피되지만, **녹음 중에는 TTS 큐 일시정지** 처리.
5. **OCR 중복/노이즈** — Florence가 같은 표지판을 2.5초마다 다시 읽음 → 문자열 유사도로
   dedupe, 2글자 미만/기호만 있는 결과 필터. texts는 **30초 후 만료** (지나간 표지판이
   Gemma 컨텍스트를 오염시키지 않도록), Gemma 프롬프트에 "seen Ns ago" 표기.
6. **macOS 권한 프롬프트** — 카메라/마이크 권한을 터미널(Python)이 처음 요청할 때 팝업이 뜸.
   **H1에 미리 트리거**해서 데모 중 팝업 방지.
7. **인터페이스 계약 먼저** — 2인 병렬 작업이므로 H1에 **SceneState JSON 스키마를 확정**하고
   시작 (A는 실제 데이터 생산, B는 가짜 JSON으로 Gemma/음성 개발 → H8 통합이 순조로움).
8. **라이선스 (심사 Q&A 대비)** — ultralytics(YOLO)는 AGPL-3.0(해커톤 OK, 상용화 시 유료),
   Florence-2는 MIT, Gemma는 자체 라이선스(허용적). 해커톤 사용엔 전부 문제없음.

## 24시간 타임라인 (2인 분담)

**A = 비전 담당, B = 음성/LLM 담당**

| 시간 | A (비전) | B (음성/LLM) |
|---|---|---|
| H0–1 | 공통: repo 스켈레톤, venv, **모델 전부 다운로드**(행사장 와이파이 리스크 — 최우선), **SceneState 스키마 확정**, 카메라/마이크 권한 트리거 | 〃 |
| H1–4 | 카메라 + YOLO + 트래킹 + 오버레이 | Ollama 셋업, 가짜 SceneState JSON으로 Gemma 프롬프트 완성, `say` TTS 큐 |
| H4–8 | Florence OCR 워커 + SceneState 통합 | whisper push-to-talk 녹음→STT, STT→Gemma→TTS 연결 |
| H8–12 | **통합**: 실제 SceneState → Gemma 경로 end-to-end | 〃 (같이) |
| H12–16 | Alert 룰 튜닝(거리 임계값 캘리브레이션, 쿨다운, 오탐 억제) | 프롬프트 튜닝, 응답 길이/레이턴시 최적화 |
| H16–20 | **데모 리허설**: 복도 걷기, Exit 표지판 시나리오 실측 + 카메라 방향(맥북 vs iPhone) 결정 → 수정 | 〃 |
| H20–24 | 버퍼 + 발표 준비 + **백업 데모 영상 녹화(필수)** | 〃 |

## 컷라인 (스코프 방어)

- **P0** (이것만 돼도 데모 성립): YOLO 탐지 → 룰베이스 음성 경고 / SceneState + Gemma Q&A(텍스트 입력이라도)
- **P1**: Florence OCR → SceneState 반영 / STT push-to-talk
- **P2** (시간 남으면): 표지판 방향 안내 고도화, Depth Anything 거리 추정, 한국어 지원
- **하지 않음**: wake word, 경로 기억/지도, 모바일 포팅, 파인튜닝

## 리스크 & 대응

1. **Florence-2 MPS 오류/속도** → CPU 실행(base 0.23B라 1~2초, 주기 실행이라 허용) → 그래도 안 되면 Apple Vision OCR
2. **YOLO26이 ultralytics에 없거나 불안정** → YOLO11n 즉시 교체 (API 동일)
3. **RAM 압박/스왑** → Florence를 질문 시에만 실행, gemma3n:e2b로 다운그레이드
4. **M1 Air 서멀 스로틀링** → 데모 직전 재부팅 + 앱 외 전부 종료, 해상도 640x480 유지
5. **행사장 소음으로 STT 실패** → push-to-talk + 마이크 근접 발화, 최후엔 텍스트 입력 폴백
6. **라이브 데모 실패** → H20에 녹화한 백업 영상으로 전환

## Gemma의 역할 — "왜 LLM이 필요한가"에 대한 답 (발표 핵심 포인트)

탐지 결과를 템플릿에 넣는 건 LLM 없이도 된다. Gemma가 증명해야 하는 건 **추론 계층**:

1. **시맨틱 매핑** — "앉을 데 있어?" → chair/bench/couch. 사용자 어휘를 탐지 어휘로 변환.
   데모에서 일부러 COCO 라벨과 다른 단어로 질문할 것.
2. **시간 축 추론** — texts는 30초 누적이므로 "아까 화장실 표지판 봤어?"에
   "12초 전 왼쪽에서 봤다"고 답변 가능. 프레임 단위 비전 모델은 못 하는 것.
3. **의도 해석 + 종합** — "출구 찾아야 해" → EXIT 표지판 위치 + 경로상 장애물을 한 문장으로.
4. **후속 대화** — 직전 2~3턴 유지로 "어느 쪽이라고?" 되묻기 대응.
5. **목표 기억** — "I'm looking for the restroom"이라고 말해두면 Gemma가 동의어 키워드를
   1회 추출(restroom/toilet/wc/...)해 저장하고, 이후 새 표지판이 감지될 때마다 룰베이스
   매칭(LLM 미경유, 0ms)으로 검사해서 매칭 시 자동으로 "Found it — a sign for RESTROOM,
   on your left" 발화. LLM은 의미 확장에 1회만, 핫패스는 룰 — 아키텍처 원칙의 응용 사례.

시나리오별 커버리지: 장애물 ✅ / 보행자 ✅ (인원수까지) / 신호등 ⚠️ (traffic light 클래스
+ bbox 색상 판정 P2, 실내 데모는 인쇄물) / 계단 ❌ 선제 경고 불가, on-demand 캡션 질문만 △.

## 발표 Q&A 대비 — 알려진 한계 (숨기지 말고 선제적으로 인정)

1. **계단·턱 감지 불가.** COCO 클래스 밖이고, depth를 붙여도 하강 계단은 "depth가 멀어짐 =
   길이 뚫림"으로 읽혀 원리적으로 위험. 전용 데이터/스테레오/LiDAR 영역 → "known object
   classes 한정, 향후 depth 센서 확장" 프레이밍.
2. **발밑 사각지대.** 웹캠 화각 특성상 발 앞 ~1m는 안 보임. 데모 장애물은 무릎 높이 이상 배치.
3. **개념 증명(PoC)이지 제품 아님.** 실사용은 웨어러블 + 진동 피드백 필요.
   "M1 Air는 개발 플랫폼, 아키텍처는 모바일 NPU로 이식 가능"이 정확한 포지셔닝.
4. **"왜 Gemma에 이미지를 직접 안 넣었나?" (예상 질문, 킬러 답변 준비됨)**
   → 실측했다: gemma3:4b 멀티모달로 프레임 투입 시 M1 Air에서 왕복 10~20초,
   그리고 정중앙 장애물 앞에서 "길이 깨끗하다"고 환각했다 (합성 장면 테스트).
   그래서 LLM에는 검증 가능한 탐지 결과(SceneState)만 주는 구조로 설계했다.
5. **거리는 근사치.** bbox 크기 휴리스틱이라 near/medium/far 수준. 미터 단위 아님.
6. **갈림길·옆길 자동 알림 불가.** 검토·실측까지 했음: Florence open-vocab detection은
   대상이 없어도 항상 탐지 결과를 반환해서(계단 없는 사진에서 staircase 오탐 실측) 부적합.
   원리적으로 맞는 방법은 depth 컬럼 휴리스틱(좌/우 depth 급증 = 옆 통로)뿐이며 P2 실험
   항목. "검토했고, 오탐이 사용자 신뢰를 깨기 때문에 뺐다"가 답변.

## 검증 방법

- H4: 웹캠 앞 사람/의자 인식 + bbox 오버레이 확인
- H8: "Where is the exit?" 텍스트 입력 → Gemma가 SceneState 기반 정답 응답
- H12: 카메라 → 경고 발화까지 <1초, 질문 → 답변 발화까지 <7초 실측
- H16: 거리 임계값 캘리브레이션(사람 1m/2m/4m bbox 높이 실측) + Exit 표지판 인쇄물로 데모 시나리오 3회 연속 성공
- 데모 시나리오(영어): ① 복도 초입에서 "I'm looking for the restroom" → "안 보인다" 답변과
  함께 목표 armed ② 걸으며 장애물 경고 시연 ③ RESTROOM 표지판이 시야에 들어오는 순간
  **자동으로** "Found it — a sign for RESTROOM, on your left" (질문 없이 선제 알림 = 하이라이트)
  ④ 후속 질문 "Which way should I go?" → Gemma가 표지판 위치 기반 답변
