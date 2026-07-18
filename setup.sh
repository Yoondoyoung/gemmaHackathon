#!/bin/bash
# 해커톤 사전 셋업 — 두 팀원 모두 해커톤 시작 전에 집에서 실행할 것.
# 목적: 행사장 와이파이에 의존하지 않도록 모든 모델/의존성을 미리 받아둔다.
set -e
cd "$(dirname "$0")"

echo "=== [1/6] Python venv + 의존성 ==="
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "=== [2/6] Ollama 설치 + Gemma 다운로드 (~3GB) ==="
if ! command -v ollama &>/dev/null; then
  brew install ollama
fi
# ollama 서버가 안 떠있으면 백그라운드로 시작
if ! curl -s localhost:11434 &>/dev/null; then
  (ollama serve &>/dev/null &)
  sleep 3
fi
ollama pull gemma3:4b

echo "=== [3/6] YOLO 가중치 다운로드 ==="
python - <<'EOF'
from ultralytics import YOLO
try:
    YOLO("yolo26n.pt")
    print("YOLO26n OK")
except Exception as e:
    print(f"YOLO26 실패 ({e}) -> YOLO11n 폴백 다운로드")
    YOLO("yolo11n.pt")
    print("YOLO11n OK")
EOF

echo "=== [4/6] Florence-2-base 다운로드 (~500MB) ==="
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("microsoft/Florence-2-base")
print("Florence-2-base OK")
EOF

echo "=== [5/6] Whisper small.en 다운로드 ==="
python - <<'EOF'
from faster_whisper import WhisperModel
WhisperModel("small.en", device="cpu", compute_type="int8")
print("Whisper small.en OK")
EOF

echo "=== [6/6] 스모크 테스트 ==="
python tools/smoke_test.py

echo ""
echo "셋업 완료. smoke test 결과에서 FAIL 항목이 있으면 MEETING.md의 트러블슈팅 참고."
