"""공유 장면 상태. JSON 스키마는 IMPLEMENTATION.md §2 계약 — 필드 변경 금지.

YOLO 루프가 update_objects()로 객체를, Florence 워커가 update_texts()로 텍스트를
채우고, Gemma 경로는 snapshot_json()으로 읽는다. 모든 메서드는 Lock으로 보호.
"""
import json
import threading
import time
from collections import deque
from dataclasses import dataclass

from src import config


@dataclass
class Event:
    kind: str        # new_near | entered_near | approaching
    label: str
    pos: str         # left | center | right
    track_id: int


def _dist_of(label: str, h_ratio: float) -> str:
    if h_ratio >= config.NEAR_THRESH.get(label, config.NEAR_THRESH["default"]):
        return "near"
    if h_ratio >= config.MED_THRESH.get(label, config.MED_THRESH["default"]):
        return "medium"
    return "far"


class SceneState:
    def __init__(self):
        self._lock = threading.Lock()
        self._objects = {}      # tid -> {label,pos,dist,status,bbox_h_ratio,misses}
        self._history = {}      # tid -> deque[(t, h_ratio)] 최근 이력 (접근율 계산)
        self._texts = []        # [{content,pos,ts}]
        self._caption = None    # 장면 묘사 질문 시에만 일시 설정

    def update_objects(self, detections) -> list:
        """detections: [{"track_id","label","pos","bbox_h_ratio"}] — 매 프레임 호출.
        상태 전이가 발생한 객체만 Event로 반환."""
        now = time.time()
        events = []
        with self._lock:
            seen = set()
            for d in detections:
                tid = d["track_id"]
                seen.add(tid)
                hist = self._history.setdefault(tid, deque(maxlen=60))
                hist.append((now, d["bbox_h_ratio"]))
                old = next(((t, r) for t, r in hist if now - t >= 0.8), None)
                closing = old is not None and \
                    (d["bbox_h_ratio"] - old[1]) / (now - old[0]) > config.CLOSING_RATE
                dist = _dist_of(d["label"], d["bbox_h_ratio"])
                prev = self._objects.get(tid)
                if prev is None:
                    status = "new"
                    if dist == "near":
                        events.append(Event("new_near", d["label"], d["pos"], tid))
                else:
                    if dist == "near" and prev["dist"] != "near":
                        events.append(Event("entered_near", d["label"], d["pos"], tid))
                    if closing and prev["status"] != "approaching":
                        events.append(Event("approaching", d["label"], d["pos"], tid))
                    status = "approaching" if closing else "seen"
                self._objects[tid] = {"label": d["label"], "pos": d["pos"],
                                      "bbox_h_ratio": d["bbox_h_ratio"],
                                      "dist": dist, "status": status, "misses": 0}
            for tid in list(self._objects):
                if tid not in seen:
                    self._objects[tid]["misses"] += 1
                    if self._objects[tid]["misses"] >= config.GONE_AFTER_MISSES:
                        del self._objects[tid]
                        self._history.pop(tid, None)
        return events

    def update_texts(self, items) -> list:
        """items: [{"content","pos"}] — Florence 워커가 호출.
        처음 보는 텍스트만 [{"content","pos"}]로 반환 (알림/목표 매칭용)."""
        import difflib
        now = time.time()
        fresh = []
        with self._lock:
            for it in items:
                key = it["content"].strip().lower()
                # OCR 지터 흡수: 유사한 텍스트("M1SIL"↔"M1SOL")는 같은 표지판으로 취급
                ex = next((t for t in self._texts
                           if difflib.SequenceMatcher(
                               None, t["content"].strip().lower(), key
                           ).ratio() >= config.TEXT_SIMILARITY), None)
                if ex:
                    ex["ts"], ex["pos"] = now, it["pos"]
                else:
                    self._texts.append({"content": it["content"],
                                        "pos": it["pos"], "ts": now})
                    fresh.append(dict(it))
            self._texts = [t for t in self._texts
                           if now - t["ts"] <= config.TEXT_TTL_SEC]
        return fresh

    def set_caption(self, caption):
        with self._lock:
            self._caption = caption

    def snapshot_json(self) -> str:
        now = time.time()
        with self._lock:
            objs = [{"track_id": tid, "label": o["label"], "pos": o["pos"],
                     "dist": o["dist"], "status": o["status"],
                     "bbox_h_ratio": round(o["bbox_h_ratio"], 2)}
                    for tid, o in self._objects.items()]
            texts = sorted((t for t in self._texts
                            if now - t["ts"] <= config.TEXT_TTL_SEC),
                           key=lambda t: t["ts"], reverse=True)
            texts = texts[:config.MAX_TEXTS_IN_SNAPSHOT]
            out = {"timestamp": now, "objects": objs,
                   "texts": [{"content": t["content"], "pos": t["pos"],
                              "age_sec": round(now - t["ts"], 1)} for t in texts]}
            if self._caption:
                out["caption"] = self._caption
        return json.dumps(out)


if __name__ == "__main__":
    # 접근 시뮬레이션: person bbox가 0.2 -> 0.7로 커지면 approaching/entered_near 발생해야 함
    scene = SceneState()
    fired = []
    for i in range(30):
        r = 0.2 + 0.5 * i / 29
        evs = scene.update_objects([{"track_id": 1, "label": "person",
                                     "pos": "center", "bbox_h_ratio": r}])
        fired += [(i, e.kind) for e in evs]
        time.sleep(0.1)
    scene.update_texts([{"content": "EXIT", "pos": "right"}])
    fresh = scene.update_texts([{"content": "EXIT", "pos": "right"},
                                {"content": "WC", "pos": "left"}])
    print("events:", fired)
    print("fresh(중복 EXIT 제외돼야 함):", fresh)
    print("snapshot:", scene.snapshot_json())
    assert any(k == "approaching" for _, k in fired), "approaching 미발생"
    assert any(k == "entered_near" for _, k in fired), "entered_near 미발생"
    assert fresh == [{"content": "WC", "pos": "left"}], "텍스트 dedupe 실패"
    print("PASS")
