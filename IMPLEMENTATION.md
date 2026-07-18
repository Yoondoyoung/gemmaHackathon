# 구현 가이드 (개발 모델용 — 이 문서만 보고 구현 가능하도록 작성됨)

이 문서는 아키텍처 결정이 모두 끝난 상태의 **구현 지시서**다.
여기 적힌 결정을 재검토하거나 "더 나은 방법"으로 바꾸지 말 것. 이유가 궁금하면 PLAN.md 참고.

## 0. 절대 규칙 (위반 금지)

1. **장애물 경고 경로에 Gemma(LLM)를 절대 넣지 않는다.** 경고는 룰베이스 + 고정 템플릿 문장. LLM은 사용자 질문 응답에만 사용.
2. **Florence-2를 매 프레임 실행하지 않는다.** 2.5초 주기 + 질문 직후 1회만.
3. **asyncio 금지, threading만 사용.** 단일 프로세스.
4. **`cv2.imshow`/`cv2.waitKey`는 반드시 메인 스레드에서만 호출** (macOS 제약. 워커 스레드에서 호출하면 크래시).
5. requirements.txt에 없는 새 의존성을 추가하지 않는다.
6. 테스트 프레임워크/추상 클래스/플러그인 구조 등 확장성 설계 금지. 24시간 해커톤 코드다 — 각 모듈 하단의 `if __name__ == "__main__":` 단독 실행 블록이 테스트의 전부.
7. SceneState JSON 스키마(§2)는 두 개발자의 계약이다. 필드 추가/변경 금지 (팀 합의 시에만).

## 1. 저장소 구조 (이대로 생성)

```
src/
  __init__.py
  config.py            # 모든 상수/임계값 (매직넘버는 전부 여기로)
  scene_state.py       # 공유 상태 (A 소유)
  vision/
    __init__.py
    camera.py          # 카메라 캡처 스레드 (A)
    yolo_worker.py     # YOLO 탐지 루프 (A)
    florence_worker.py # OCR 주기 실행 (A)
  alerts.py            # 룰베이스 경고 엔진 (A)
  llm/
    __init__.py
    gemma_client.py    # Ollama HTTP 호출 (B)
  audio/
    __init__.py
    tts.py             # 발화 큐 (B)
    stt.py             # push-to-talk 녹음 + whisper (B)
  main.py              # 스레드 조립 + 오버레이 + 키 입력 (공동)
prompts/system_prompt.txt   # 이미 있음
tools/fake_scene.json       # 이미 있음 — B는 통합 전까지 이걸로 개발
tools/smoke_test.py         # 이미 있음
```

## 2. SceneState JSON 스키마 (계약 — 변경 금지)

```json
{
  "timestamp": 1720000000.0,
  "objects": [
    {"track_id": 3, "label": "person", "pos": "center", "dist": "near",
     "status": "approaching", "bbox_h_ratio": 0.62}
  ],
  "texts": [
    {"content": "EXIT", "pos": "right", "age_sec": 4.2}
  ]
}
```

- `pos`: bbox 중심 x를 화면 3등분 → `left | center | right`
- `dist`: bbox 높이비율 기준 → `near | medium | far` (임계값은 config.NEAR_THRESH/MED_THRESH)
- `status`: `new`(첫 등장) → `approaching`(bbox 커지는 중) → `seen`(안정) → 사라지면 객체 제거
- `texts`: 30초 지난 항목은 스냅샷에서 제외. 같은 내용(대소문자 무시, 유사도)은 age만 갱신

## 3. config.py

```python
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

YOLO_WEIGHTS = "yolo26n.pt"          # 로드 실패 시 yolo11n.pt 자동 폴백
YOLO_CONF = 0.4
YOLO_DEVICE = "mps"

# 경고 대상 클래스만 (COCO 라벨) — 이외 클래스는 SceneState에도 안 넣음 (노이즈 억제)
TRACK_LABELS = {"person", "chair", "bicycle", "car", "dog", "backpack",
                "suitcase", "bench", "potted plant", "couch", "dining table"}

# 거리 임계값: bbox_h / frame_h. 클래스별, 없으면 default. H16 리허설에서 실측 튜닝.
NEAR_THRESH = {"person": 0.60, "chair": 0.45, "default": 0.50}
MED_THRESH  = {"person": 0.30, "chair": 0.25, "default": 0.28}
CLOSING_RATE = 0.15                  # bbox_h_ratio 초당 증가율이 이 이상이면 approaching

ALERT_COOLDOWN_SEC = 5.0             # 같은 track_id 재경고 금지 시간
ALERT_GLOBAL_INTERVAL = 2.0          # 전체 경고 최소 간격
TEXT_TTL_SEC = 30.0                  # OCR 텍스트 수명
ANNOUNCE_NEW_SIGNS = True            # 새 표지판 최초 1회 음성 알림
GOAL_ENABLED = True                  # 목표 기억 (표지판 자동 매칭 알림)
FLORENCE_PERIOD_SEC = 2.5
FLORENCE_MODEL = "microsoft/Florence-2-base"

OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL = "gemma3:4b"
GEMMA_MAX_TOKENS = 80
GEMMA_TIMEOUT_SEC = 30

WHISPER_MODEL = "small.en"
SAMPLE_RATE = 16000
TTS_VOICE = "Samantha"
```

## 4. 모듈별 구현 스펙

### 4.1 vision/camera.py (A)

```python
class Camera:
    """캡처 전용 스레드. 항상 '최신 프레임 1장'만 유지 (오래된 프레임은 버림)."""
    def __init__(self, source=CAM_INDEX)    # int면 웹캠, str이면 비디오 파일 경로
    def start(self) -> None            # cv2.VideoCapture(source), 데몬 스레드로 read 루프
    def latest(self) -> np.ndarray | None   # lock 걸고 복사본 반환
    def stop(self) -> None
```

- 워커들이 각자 `latest()`를 당겨가는 pull 구조. 큐/버퍼링 금지 (지연 누적 방지).
- **비디오 주입 모드**: `python -m src.main --video clips/hallway.mp4` → 같은 클립으로
  임계값을 바꿔가며 반복 테스트하는 재현 가능한 하네스. 비디오 모드에서는 원본 FPS에 맞춰
  sleep (안 그러면 영상이 수 배속으로 소진됨). 테스트 클립은 직접 촬영(폰 가슴 높이,
  복도/사람 접근/표지판) + 유튜브 1인칭 walking 영상. 연구용 대규모 데이터셋
  (Ego4D 등)은 라이선스/용량 문제로 사용 금지.

### 4.2 scene_state.py (A)

```python
@dataclass
class Event:                # AlertEngine 입력
    kind: str               # "new_near" | "approaching" | "entered_near"
    label: str
    pos: str
    track_id: int

class SceneState:
    def update_objects(self, detections) -> list[Event]
        # detections: [{"track_id", "label", "bbox_h_ratio", "pos"}] — YOLO 루프가 프레임마다 호출
        # 내부에 track_id별 이력(deque, 최근 1초) 유지 → 증가율 계산 → status/dist 판정
        # 상태 전이가 발생한 객체만 Event로 반환:
        #   처음 등장했는데 이미 near → "new_near"
        #   증가율 > CLOSING_RATE → "approaching" (전이 순간 1회만)
        #   medium→near 진입 → "entered_near"
        # 5프레임 연속 미검출 track_id는 제거
    def update_texts(self, items) -> list[str]
        # items: [{"content", "pos"}] — Florence 워커가 호출
        # 기존 텍스트와 대소문자 무시 비교로 중복이면 timestamp만 갱신, 아니면 추가
        # 반환: 처음 보는 텍스트의 1회성 알림 문장들 (ANNOUNCE_NEW_SIGNS=True일 때)
        #   예: "Sign detected: EXIT, on your right" → 호출자가 speaker.say(문장, priority=1)
        #   같은 내용은 TTL 내 재알림 금지. 정보 정책: 표지판의 존재를 모르면 질문도 못 하므로
        #   존재만 1회 push, 내용 상세는 pull(질문)로.
    def snapshot_json(self) -> str
        # 스키마(§2)대로 직렬화. TTL 지난 텍스트 제외. age_sec 계산해서 포함.
        # texts는 최신순 최대 6개까지만 포함 (실측: 텍스트 8개짜리 장면에서 Gemma 응답이
        # 5초→7초대로 늘어남 — 프롬프트가 길어지면 프리필이 느려진다).
    # 모든 메서드는 내부 threading.Lock으로 보호
```

### 4.3 vision/yolo_worker.py (A)

```python
def load_model():
    from ultralytics import YOLO
    try:    return YOLO("yolo26n.pt")
    except Exception: return YOLO("yolo11n.pt")   # API 완전 동일, 조용히 폴백

def run_loop(camera, scene, alert_engine, speaker, stop_flag):
    model = load_model()
    while not stop_flag.is_set():
        frame = camera.latest()
        if frame is None: continue
        results = model.track(frame, persist=True, device=YOLO_DEVICE,
                              conf=YOLO_CONF, verbose=False, tracker="bytetrack.yaml")
        boxes = results[0].boxes
        # boxes.id (track id, None 가능 — None이면 그 박스는 스킵),
        # boxes.cls, boxes.xyxy 사용. label이 TRACK_LABELS 밖이면 스킵.
        # pos = 중심 x / FRAME_W → 1/3 단위, bbox_h_ratio = (y2-y1)/FRAME_H
        events = scene.update_objects(detections)
        for sentence in alert_engine.process(events):
            speaker.say(sentence, priority=0)
        # 오버레이용으로 마지막 results[0].plot() 이미지를 공유 변수에 저장 (main이 그림)
```

- `model.track(persist=True)`가 ultralytics의 프레임 단위 트래킹 API다. 직접 트래커 구현 금지.
- 별도 sleep 불필요 — 추론 자체가 페이스 조절 (M1에서 약 10~20 FPS).

### 4.4 vision/florence_worker.py (A)

```python
# 로드 (MPS + fp32 — 실측: MPS fp32 0.9s/frame, CPU 2.8s, fp16은 빈 출력이라 금지):
model = AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL, trust_remote_code=True,
                                             torch_dtype=torch.float32).to("mps")
# MPS에서 오류가 나면 그때만 .to("cpu")로 폴백하고 FLORENCE_PERIOD_SEC를 5.0으로 늘릴 것
processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)

# 루프: FLORENCE_PERIOD_SEC마다 camera.latest() 1장 → OCR → scene.update_texts()
inputs = processor(text="<OCR_WITH_REGION>", images=pil_img, return_tensors="pt")
gen = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                     max_new_tokens=128, num_beams=1, do_sample=False)
raw = processor.batch_decode(gen, skip_special_tokens=False)[0]
parsed = processor.post_process_generation(raw, task="<OCR_WITH_REGION>",
                                           image_size=(w, h))
# parsed["<OCR_WITH_REGION>"] = {"quad_boxes": [...], "labels": ["EXIT", ...]}
# 필터: 라벨 앞뒤 특수토큰(</s> 등) strip, 2글자 미만이거나 영숫자가 없으면 버림
# pos: quad_box x좌표 평균 → left/center/right
```

- `num_beams=1` 필수 (beam search 켜면 2~3배 느려짐).
- Florence 로드가 transformers 버전 문제로 실패하면: `pip install "transformers==4.49.0"`.
- 그래도 실패하면 이 워커만 비활성화하고 진행 (P1 기능. P0인 YOLO 경로를 막지 말 것).

### 4.5 alerts.py (A)

```python
class AlertEngine:
    def process(self, events: list[Event]) -> list[str]:
        # 필터 순서:
        # 1. event.pos != "center" 이면 버림 (좌우는 경고 안 함)
        # 2. track_id별 마지막 경고 후 ALERT_COOLDOWN_SEC 미경과면 버림
        # 3. 전역: 마지막 경고 후 ALERT_GLOBAL_INTERVAL 미경과면 버림
        # 4. 한 번의 process 호출에서 최대 1건만 통과 (가장 near인 것 우선)
        # 템플릿 (f-string, LLM 금지):
        #   new_near / entered_near → f"{label} ahead, close"
        #   approaching             → f"{label} approaching"
```

### 4.6 llm/gemma_client.py (B)

```python
def ask_streaming(question: str, scene_json: str, on_sentence) -> str:
    # stream=True로 받아서 문장 경계('.', '!', '?')마다 on_sentence(문장) 콜백 호출.
    # → 첫 문장이 완성되는 순간 TTS가 발화 시작 = 체감 레이턴시 절반.
    payload = {
        "model": GEMMA_MODEL,
        "system": open("prompts/system_prompt.txt").read(),
        "prompt": f"Current scene:\n{scene_json}\n\nUser question: {question}",
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": GEMMA_MAX_TOKENS},
    }
    # requests.post(..., stream=True) → iter_lines() → 각 줄 json의 "response" 조각 누적
    # 버퍼에서 문장 완성 시 on_sentence(sentence) 후 버퍼 비움. 전체 텍스트 반환.
    # 예외 시(연결 실패/타임아웃) on_sentence("Sorry, I couldn't process that.") — 크래시 금지
```

- **금지: 프레임 이미지를 Gemma에 직접 넣지 말 것.** 실측 결과 10~20초 걸리고,
  장애물이 정중앙에 있는데 "길이 깨끗하다"고 환각했다 (안전상 최악의 실패 모드).
  Gemma에는 SceneState JSON(검증된 탐지 사실)만 준다.
- **후속 대화**: 모듈 내부에 직전 2~3개의 (질문, 답변) 쌍을 리스트로 유지하고 prompt 앞에
  "Previous exchange:" 로 붙임 ("어느 쪽이라고?" 되묻기 대응). 그 이상의 대화 기억은 금지.
- **목표 추출**: `extract_goal(question) -> list[str] | None` —
  `prompts/goal_prompt.txt`를 system으로, temperature 0.0, num_predict 40.
  응답이 "NONE"이면 None, 아니면 쉼표 분리 키워드 리스트.
  검증된 프로토타입이 `tools/test_goal.py`에 있음 (11케이스 PASS) — 그대로 가져다 쓸 것.
- "내 앞에 뭐가 보여?" 류 장면 묘사 질문 대응: main이 질문 텍스트에
  describe/see/look/front 키워드가 있으면 florence_worker에 `<MORE_DETAILED_CAPTION>`
  1회 실행(+0.8초)을 요청해서 결과를 scene_json에 `"caption": "..."` 필드로 덧붙임.
  (이 필드는 스냅샷에 캡션이 있을 때만 존재 — 스키마 §2의 유일한 선택적 확장)

- 단독 테스트: `python -m src.llm.gemma_client` → fake_scene.json 읽어서
  "Where is the restroom?" 물었을 때 RESTROOM/left가 언급되면 통과.

### 4.7 audio/tts.py (B)

```python
class Speaker:
    """단일 워커 스레드 + PriorityQueue[(priority, seq, text)].
    priority 0 = 경고, 1 = Gemma 답변."""
    def say(self, text, priority):
        # priority 0이고 현재 priority 1 발화가 진행 중이면:
        #   현재 say 프로세스 kill + 큐에 남은 priority 1 항목 전부 폐기 (안전 > 답변)
        # 발화는 subprocess.Popen(["say", "-v", TTS_VOICE, text]) — Popen 핸들 보관 (kill용)
    def pause(self) / resume(self)
        # 녹음 중 자기 목소리 루프 방지. pause 중 들어온 priority 0은 resume 후 발화,
        # priority 1은 폐기.
```

### 4.8 audio/stt.py (B)

```python
class Recorder:
    def start(self):  # sd.InputStream(samplerate=16000, channels=1, dtype="float32") 콜백으로 청크 누적
    def stop_and_transcribe(self) -> str:
        # np.concatenate(청크들) → WhisperModel(WHISPER_MODEL, device="cpu",
        # compute_type="int8").transcribe(audio) → 세그먼트 텍스트 join
        # 1초 미만 녹음이거나 결과가 빈 문자열이면 "" 반환
```

- WhisperModel은 모듈 로드 시 1회만 생성 (매번 만들면 3초씩 낭비).

### 4.9 main.py (공동 — 통합 시점 H8에 작성)

```python
# 메인 스레드가 UI를 갖는다 (규칙 4). 흐름:
# 1. Camera, SceneState, AlertEngine, Speaker 생성
# 2. 데몬 스레드 시작: yolo_worker.run_loop, florence_worker.run_loop
# 3. 메인 루프:
#    overlay = yolo가 저장한 plot 이미지 (없으면 raw frame)
#    상단에 scene.snapshot_json()의 texts 표시, 녹음 중이면 "REC" 표시
#    cv2.imshow("assist", overlay)
#    key = cv2.waitKey(1)
#    SPACE: 녹음 토글 —
#      시작: speaker.pause(); recorder.start()
#      종료: q = recorder.stop_and_transcribe(); speaker.resume()
#            별도 스레드에서: answer = gemma.ask(q, scene.snapshot_json())
#                            speaker.say(answer, priority=1)
#            (QA 스레드 실행 중 재진입 방지 — busy 플래그)
#    't': 터미널 input()으로 질문 받기 (STT 폴백 데모용)
#    'q': stop_flag set 후 종료
#
# 목표 기억 (GOAL_ENABLED, M7):
#  - QA 스레드에서 답변 발화를 큐에 넣은 "후에" extract_goal(질문) 실행 (응답 지연 방지).
#    결과가 있으면 active_goal = {"keywords": [...]} 로 교체 (동시 목표는 1개만).
#  - update_texts가 반환한 새 표지판마다: any(kw in content.lower() for kw in keywords)
#    → 매칭 시 speaker.say(f"Found it — a sign for {content}, {pos_문구}", priority=1)
#    후 active_goal = None (1회 알림 후 해제). 매칭은 LLM 미경유 — 절대 규칙 1/2 준수.
```

## 5. 구현 순서 (마일스톤 — 반드시 이 순서로, 각각 완료 검증 후 다음으로)

| # | 내용 | 담당 | 완료 기준 (실행해서 확인) |
|---|---|---|---|
| M1 | config + camera + yolo_worker + 미니 오버레이 | A | 화면에 본인 얼굴 bbox + track_id 표시, 10 FPS 이상 |
| M2 | tts.py + gemma_client.py (fake_scene 사용) | B | "Where is the restroom?" → 좌측 언급 답변이 음성으로 나옴 |
| M3 | scene_state + alerts 연결 | A | 카메라에 다가가면 1회 "person ahead, close" 발화, 스팸 없음 |
| M4 | stt.py push-to-talk | B | 스페이스 토글로 말한 문장이 텍스트로 출력 |
| M5 | main.py 통합 (fake_scene → 실제 snapshot 교체) | 공동 | 데모 시나리오 2개 end-to-end 통과 |
| M6 | florence_worker 연결 | A | EXIT 인쇄물 비추면 texts에 등장, Gemma가 인용 |
| M7 | 목표 기억 (extract_goal + 표지판 매칭) | B | "I'm looking for the restroom" 말한 뒤 RESTROOM 인쇄물 비추면 자동으로 "Found it" 발화 |

- **M5까지가 P0.** M6(Florence)이 안 풀리면 과감히 스킵하고 M5 상태로 데모.
- **P2 (M6까지 끝나고 시간이 남을 때만, 순서대로):**
  1. 오버레이에 성능 지표 표시 (FPS, OCR age, Q&A 왕복 시간) — 심사 어필용
  2. Depth Anything V2 small (`transformers pipeline("depth-estimation")`, 의존성 추가 아님)로
     화면 중앙 depth 급접근 시 클래스 무관 "obstacle ahead" 경고 — 임계값 튜닝이 까다로우니
     기존 경고 룰을 건드리지 말고 별도 룰로 추가만 할 것.
     (같은 depth 맵으로 좌/우 컬럼 depth 급증 = 옆 통로/갈림길 감지도 시도 가능하나
      난도 더 높음 — obstacle 룰이 안정된 뒤에만)
  3. 신호등 색상 판정: YOLO가 traffic light를 잡으면 해당 bbox 크롭 → HSV로 빨강/초록
     우세 색 판정 (룰베이스, LLM 금지) → SceneState의 해당 객체에 `"light": "red"|"green"`
     추가. 데모는 실외 불가하므로 신호등 이미지 인쇄물/화면으로 시연

## 6. 알려진 함정 (여기서 시간 날리지 말 것)

| 함정 | 대응 |
|---|---|
| Florence-2 fp16에서 출력이 빈 리스트 | fp16 금지, fp32 사용 (실측 확인된 버그) |
| Florence `<OPEN_VOCABULARY_DETECTION>`으로 문/통로/계단 감지하고 싶은 유혹 | 금지. phrase grounding이라 **대상이 없어도 항상 박스를 반환** (실측: 계단 없는 사진에서 staircase 탐지됨). 신뢰도 점수도 없음. 경고 경로에 쓰면 오탐 알림 남발 |
| Florence 로드 시 `forced_bos_token_id`/`flash_attn` 에러 | transformers가 4.49.0인지 확인 (4.50+ 깨짐) |
| `boxes.id`가 None | 트래킹 미확정 박스 — 해당 박스만 스킵 (크래시 주의) |
| cv2 창이 안 뜨거나 크래시 | imshow를 워커 스레드에서 부르고 있는 것. 메인 스레드로 옮겨라 |
| 경고가 쉴 새 없이 울림 | 룰을 고치기 전에 config의 쿨다운/임계값부터 조정 |
| whisper가 TTS 음성을 받아적음 | 녹음 시작 전 speaker.pause() 호출 누락 확인 |
| say가 안 끊김 | subprocess.run(블로킹) 쓰고 있는 것 — Popen으로 바꾸고 핸들 kill |
| Gemma 첫 호출만 10초+ | 모델 콜드로드. main 시작 시 워밍업 호출 1회 넣기 |
| Whisper 전사가 5초+ 걸림 | config의 WHISPER_MODEL을 "base.en"으로 다운그레이드 (정확도 소폭 하락, 짧은 질문엔 충분) |
| 전체적으로 느려짐/스왑 | Activity Monitor에서 메모리 압박 확인 → 브라우저 종료, Florence 주기 5초로 |

## 7. 검증 명령

```bash
source .venv/bin/activate
python tools/smoke_test.py            # 환경 전체 (8항목 PASS)
python tools/test_gemma.py            # Gemma 추론 회귀 테스트 (7케이스, 픽스처 기반)
python tools/test_pipeline.py tools/images/bus.jpg "Is it safe to walk?"
                                      # 실이미지 → YOLO+Florence → SceneState → Gemma E2E
                                      # (src/ 구현 결과가 이 도구의 SceneState와 같아야 함)
python tools/test_video.py clips/hallway.mp4
                                      # 영상 시간 축: 트래킹 + 접근 판정 + 경고 쿨다운 시뮬레이션
                                      # (src/alerts.py는 이 도구의 룰과 동일해야 함 — 기준 구현.
                                      #  실측: 6초 접근 영상에서 경고 2건 발화 / 15건 쿨다운 억제 = 정상)
python -m src.vision.yolo_worker      # M1: 오버레이 창
python -m src.llm.gemma_client        # M2: fake_scene Q&A
python -m src.audio.stt               # M4: push-to-talk
python -m src.main                    # M5/M6: 통합 데모
python -m src.main --video clips/hallway.mp4   # 비전 경로 재현 테스트
```

- **테스트 분리 원칙**: Gemma는 픽셀을 안 보고 SceneState JSON만 본다. 따라서
  추론 품질은 `tools/test_gemma.py`(장면 픽스처 + 질문 + 기대 키워드)로 영상 없이 테스트하고,
  비전 경로(프레임→SceneState 정확도)는 `--video` 클립으로 테스트한다. 접합부는 리허설.
- **시스템 프롬프트를 수정하면 반드시 test_gemma.py를 재실행**해서 회귀 확인
  (실측: 규칙 3줄 추가로 거리 환각·위치 오류가 사라짐 — 프롬프트는 깨지기 쉽다).
  새 실패 유형을 발견하면 CASES에 케이스를 추가하고 튜닝할 것.
