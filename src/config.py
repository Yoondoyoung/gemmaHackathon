"""전역 상수 — 모든 매직넘버는 여기로 (IMPLEMENTATION.md §3)."""

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

YOLO_WEIGHTS = "yolo26n.pt"          # 로드 실패 시 YOLO_FALLBACK
YOLO_FALLBACK = "yolo11n.pt"
YOLO_CONF = 0.4
YOLO_DEVICE = "mps"
YOLO_MAX_FPS = 30                    # 60fps 풀가동은 GPU 경합(프레임드랍 체감 원인)과
                                     # 팬리스 M1 발열만 키운다 — 실측 진단

# 경고/추적 대상 클래스 (COCO) — 이외는 SceneState에도 안 넣음 (노이즈 억제)
# 음식·스포츠 용품 등 Q&A 잡음만 키우는 클래스는 제외.
TRACK_LABELS = {
    # people & animals
    "person", "dog", "cat",
    # vehicles & traffic
    "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "stop sign", "fire hydrant", "parking meter",
    # furniture & indoor obstacles / landmarks
    "chair", "couch", "bench", "bed", "dining table", "toilet",
    "potted plant", "refrigerator", "oven", "microwave", "sink",
    # personal belongings (분실물·Q&A)
    "backpack", "handbag", "suitcase", "umbrella", "cell phone",
    "laptop", "keyboard", "mouse", "remote", "book", "bottle", "cup",
    # misc
    "skateboard", "clock", "vase",
}

# 거리 임계값: bbox_h / frame_h. 리허설에서 실측 캘리브레이션.
# 대형 물체(테이블/소파/차량)는 멀어도 bbox가 커서 임계값을 높게 잡는다.
NEAR_THRESH = {"person": 0.60, "chair": 0.50, "bicycle": 0.55,
               "dining table": 0.80, "couch": 0.80, "bench": 0.70, "bed": 0.85,
               "car": 0.70, "bus": 0.85, "truck": 0.85, "train": 0.85,
               "refrigerator": 0.80, "oven": 0.55, "sink": 0.50,
               "toilet": 0.50, "laptop": 0.35, "bottle": 0.30, "cup": 0.25,
               "cell phone": 0.20, "book": 0.30, "clock": 0.30, "vase": 0.40,
               "umbrella": 0.55, "skateboard": 0.35,
               "fire hydrant": 0.45, "parking meter": 0.45, "default": 0.55}
MED_THRESH = {"person": 0.30, "chair": 0.28,
              "dining table": 0.50, "couch": 0.50, "bed": 0.55, "car": 0.40,
              "bus": 0.55, "truck": 0.55, "train": 0.55,
              "refrigerator": 0.50, "toilet": 0.28,
              "laptop": 0.18, "bottle": 0.15, "cup": 0.12, "cell phone": 0.10,
              "umbrella": 0.30, "skateboard": 0.18,
              "fire hydrant": 0.25, "parking meter": 0.25, "default": 0.30}
NEAR_BOTTOM_MIN = 0.75               # near 판정 추가 조건: bbox 하단이 프레임 하단 25% 안
                                     # (멀리 있는 대형 물체는 화면 중간에 떠 있음)

# 미터 단위 깊이 (Depth Anything V2 metric-indoor, 실측 192ms/frame @MPS)
# 깊이 맵이 있으면 bbox 휴리스틱 대신 이걸로 near/medium/far 판정 + 경고에 미터 포함
DEPTH_ENABLED = True
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_PERIOD_SEC = 1.5
DEPTH_NEAR_M = 2.5                   # 이내면 near — 보행속도 1.2m/s 기준 ~2초 여유
DEPTH_MED_M = 5.0                    # 이내면 medium
CLOSING_RATE = 0.15                  # bbox_h_ratio 초당 증가율 → approaching
DEPTH_CLOSING_MPS = 0.5              # depth 감소율(m/s) — 정면에서 이 속도로 접근 중이면 경고
DEPTH_CLOSING_MAX_M = 5.0            # 이 거리 안에서 접근 중일 때만 (너무 먼 접근은 무시)

# 신호등 색 판정 (룰베이스 HSV — LLM/Florence 금지: 안전 판정 + 즉시성 필요)
LIGHT_MIN_FRACTION = 0.03            # 크롭에서 점등 색 픽셀 비율이 이 이상일 때만 판정
LIGHT_MIN_BOX_PX = 12                # 이보다 작은 박스는 판정 안 함 (노이즈)
GONE_AFTER_MISSES = 5                # 연속 미검출 프레임 수 → 객체 제거

# 의자 점유: person–chair IoU 또는 person 하단 중심이 chair 안이면 occupied
CHAIR_OCCUPIED_IOU = 0.15
CHAIR_PERSON_BOTTOM_IN_CHAIR = True

ALERT_COOLDOWN_SEC = 5.0             # 같은 track_id 재경고 금지
ALERT_GLOBAL_INTERVAL = 2.0          # 전체 경고 최소 간격

TEXT_TTL_SEC = 30.0                  # OCR 텍스트 수명
MAX_TEXTS_IN_SNAPSHOT = 6            # 스냅샷 텍스트 상한 (프리필 지연 방지)
ANNOUNCE_NEW_SIGNS = True            # 새 표지판 최초 1회 알림
ANNOUNCE_MAX_WORDS = 3               # 이보다 긴 텍스트는 저장만 (OCR 잡음 낭독 방지)
ANNOUNCE_MAX_CHARS = 20
ANNOUNCE_MIN_INTERVAL = 4.0          # 표지판 알림 전역 최소 간격
# 알림 가치 판정: 내비게이션 어휘에 있거나, 화면에서 충분히 크게 보이는 텍스트만 발화.
# (작은 글자 = 모니터 브랜드/제품 라벨 등 → 저장만, Q&A·목표 매칭에는 사용)
NAV_SIGN_WORDS = {"exit", "restroom", "toilet", "toilets", "wc", "men", "women",
                  "ladies", "gents", "gate", "elevator", "lift", "stairs",
                  "escalator", "entrance", "emergency", "information", "info",
                  "cafeteria", "cafe", "parking", "reception", "registration",
                  "push", "pull", "caution", "danger", "wet", "floor"}
SIGN_MIN_H_RATIO = 0.05              # 텍스트 높이/프레임 높이 — 이 이상이면 '눈에 띄는 표지판'
TEXT_SIMILARITY = 0.75               # OCR 지터 dedupe 임계 (difflib ratio)
TEXT_REANNOUNCE_GAP = 10.0           # 시야에서 이 시간 이상 사라졌다 재등장하면 재알림
GOAL_ENABLED = True                  # 목표 기억 (표지판 자동 매칭)

FLORENCE_MODEL = "microsoft/Florence-2-base"
FLORENCE_PERIOD_SEC = 3.5            # GPU 창 빈도 (짧을수록 OCR 빠른 대신 프레임드랍 잦음)
FLORENCE_MAX_TOKENS = 64             # 표지판 용도 충분 — 길수록 GPU 점유 창이 길어짐

OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL = "gemma4:e2b"
GEMMA_MAX_TOKENS = 80
GEMMA_TIMEOUT_SEC = 45               # 이미지 전송 시 프리필이 늘어나 30s는 빠듯 (실측 여유)
GEMMA_KEEP_ALIVE = "30m"             # 유휴 언로드 방지 — 언로드되면 다음 질문에 리로드 5~10초
# gemma4는 thinking 모델 — True면 num_predict가 생각에 소진되어 답이 빈 문자열이 됨
GEMMA_THINK = False
# b 질문 시 장면 JSON + 카메라 프레임을 함께 전송 (정확도↑, 지연 약간↑)
GEMMA_SEND_IMAGE = True
GEMMA_IMAGE_MAX_W = 480          # 전송 전 리사이즈 (원본 640→지연·토큰 절감)
GEMMA_IMAGE_JPEG_Q = 70

WHISPER_MODEL = "small.en"
SAMPLE_RATE = 16000
TTS_VOICE = "Samantha"

SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
GOAL_PROMPT_PATH = "prompts/goal_prompt.txt"
GEMMA_FRAME_DUMP_DIR = "gemma_frames"   # b 질문 시 Gemma에 전달된 장면 스냅샷 저장

POS_SPOKEN = {"left": "on your left", "center": "ahead of you",
              "right": "on your right"}
