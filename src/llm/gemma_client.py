"""Gemma(Ollama) 클라이언트 — 사용자 Q&A 전용 (경고 경로 사용 금지, 하드 룰 1).

ask_streaming: 문장 경계마다 콜백 → 첫 문장 완성 즉시 TTS 시작 (체감 레이턴시 절반).
extract_goal: 발화에서 목표 키워드 추출 (표지판 자동 매칭용, 답변 발화 후 백그라운드 호출).
장면은 SceneState JSON이 기준이고, 옵션으로 카메라 프레임(base64 JPEG)을 함께 보낸다.
"""
import base64
import json
import pathlib
from collections import deque

import cv2
import requests

from src import config

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_history = deque(maxlen=3)   # 후속 대화용 (질문, 답변) — 그 이상 기억 금지


def encode_frame(frame) -> str | None:
    """카메라 BGR 프레임 → Ollama images용 base64 JPEG. 실패 시 None."""
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        max_w = config.GEMMA_IMAGE_MAX_W
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)))
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.GEMMA_IMAGE_JPEG_Q])
        if not ok:
            return None
        return base64.b64encode(buf).decode("ascii")
    except Exception:
        return None


def _post(payload, stream=False):
    return requests.post(config.OLLAMA_URL, json=payload, stream=stream,
                         timeout=config.GEMMA_TIMEOUT_SEC)


def ask_streaming(question: str, scene_json: str, on_sentence,
                  image_b64: str | None = None) -> str:
    prev = "".join(f"Previous exchange:\nUser: {q}\nYou: {a}\n\n"
                   for q, a in _history)
    payload = {
        "model": config.GEMMA_MODEL,
        "system": (_ROOT / config.SYSTEM_PROMPT_PATH).read_text(),
        "prompt": f"{prev}Current scene:\n{scene_json}\n\nUser question: {question}",
        "stream": True,
        "think": config.GEMMA_THINK,
        "options": {"temperature": 0.2, "num_predict": config.GEMMA_MAX_TOKENS},
    }
    if image_b64:
        payload["images"] = [image_b64]
    full, buf = "", ""
    try:
        with _post(payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                piece = json.loads(line).get("response", "")
                full += piece
                buf += piece
                while True:
                    idxs = [i for i in (buf.find(c) for c in ".!?") if i != -1]
                    if not idxs:
                        break
                    cut = min(idxs) + 1
                    sent, buf = buf[:cut].strip(), buf[cut:]
                    if sent:
                        on_sentence(sent)
    except Exception as e:
        print(f"[gemma] 요청 실패: {type(e).__name__}: {e}")
        on_sentence("Sorry, I couldn't process that.")
        return ""
    if buf.strip():
        on_sentence(buf.strip())
    if not full.strip():
        print("[gemma] 빈 응답 (thinking 토큰 소진 가능 — config.GEMMA_THINK 확인)")
        on_sentence("Sorry, I couldn't process that.")
        return ""
    _history.append((question, full.strip()))
    return full.strip()


def extract_goal(question: str):
    """목표가 있으면 소문자 키워드 리스트, 없으면 None."""
    try:
        r = _post({
            "model": config.GEMMA_MODEL,
            "system": (_ROOT / config.GOAL_PROMPT_PATH).read_text(),
            "prompt": question,
            "stream": False,
            "think": config.GEMMA_THINK,
            "options": {"temperature": 0.0, "num_predict": 40},
        })
        r.raise_for_status()
        out = r.json()["response"].strip()
    except Exception:
        return None
    if out.upper().startswith("NONE"):
        return None
    return [k.strip().lower() for k in out.split(",") if k.strip()] or None


def warmup():
    """콜드로드 방지 — main 시작 시 백그라운드 호출."""
    try:
        _post({"model": config.GEMMA_MODEL, "prompt": "OK", "stream": False,
               "options": {"num_predict": 2}})
    except Exception:
        pass


if __name__ == "__main__":
    scene = json.dumps({"timestamp": 0, "objects": [
        {"track_id": 1, "label": "person", "pos": "center", "dist": "medium",
         "status": "approaching", "bbox_h_ratio": 0.45}],
        "texts": [{"content": "RESTROOM ←", "pos": "left", "age_sec": 6.0}]})
    print("Q: Where is the restroom?")
    ask_streaming("Where is the restroom?", scene,
                  lambda s: print(f"  문장: {s}"))
    print("goal:", extract_goal("I'm looking for the restroom"))
    print("goal(없어야 함):", extract_goal("Is it safe here?"))

    chairs = json.dumps({"timestamp": 0, "objects": [
        {"track_id": 10, "label": "chair", "pos": "left", "dist": "near",
         "status": "seen", "bbox_h_ratio": 0.5, "occupied": False},
        {"track_id": 11, "label": "chair", "pos": "center", "dist": "near",
         "status": "seen", "bbox_h_ratio": 0.48, "occupied": True},
        {"track_id": 12, "label": "chair", "pos": "right", "dist": "medium",
         "status": "seen", "bbox_h_ratio": 0.3, "occupied": False},
        {"track_id": 13, "label": "person", "pos": "center", "dist": "near",
         "status": "seen", "bbox_h_ratio": 0.55}],
        "texts": []})
    print("Q: Where can I find an empty seat?")
    ask_streaming("Where can I find an empty seat?", chairs,
                  lambda s: print(f"  문장: {s}"))
