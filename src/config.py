"""전역 상수 — 모든 매직넘버는 여기로 (IMPLEMENTATION.md §3)."""

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

YOLO_WEIGHTS = "yolo26n.pt"          # 로드 실패 시 YOLO_FALLBACK
YOLO_FALLBACK = "yolo11n.pt"
YOLO_CONF = 0.4
YOLO_DEVICE = "mps"

# 경고/추적 대상 클래스 (COCO) — 이외는 SceneState에도 안 넣음 (노이즈 억제)
TRACK_LABELS = {"person", "chair", "bicycle", "car", "dog", "backpack",
                "suitcase", "bench", "potted plant", "couch", "dining table",
                "bus", "truck", "motorcycle", "traffic light", "stop sign"}

# 거리 임계값: bbox_h / frame_h. 리허설에서 실측 캘리브레이션.
NEAR_THRESH = {"person": 0.60, "chair": 0.45, "default": 0.50}
MED_THRESH = {"person": 0.30, "chair": 0.25, "default": 0.28}
CLOSING_RATE = 0.15                  # bbox_h_ratio 초당 증가율 → approaching
GONE_AFTER_MISSES = 5                # 연속 미검출 프레임 수 → 객체 제거

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
FLORENCE_PERIOD_SEC = 2.5

OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL = "gemma3:4b"
GEMMA_MAX_TOKENS = 80
GEMMA_TIMEOUT_SEC = 30

WHISPER_MODEL = "small.en"
SAMPLE_RATE = 16000
TTS_VOICE = "Samantha"

SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
GOAL_PROMPT_PATH = "prompts/goal_prompt.txt"

POS_SPOKEN = {"left": "on your left", "center": "ahead of you",
              "right": "on your right"}
