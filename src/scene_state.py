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
    depth_m: float = None   # 미터 단위 실측 깊이 (depth 워커 활성 시)


def _dist_of(label: str, h_ratio: float, bottom: float = 1.0,
             depth_m: float = None) -> str:
    """depth_m(미터, Depth Anything)이 있으면 그걸로 판정 — 가장 정확.
    없으면 bbox 휴리스틱 폴백: 크기 + 바닥 접점(멀리 있는 대형 물체는 화면 중간에 뜸)."""
    if depth_m is not None:
        if depth_m <= config.DEPTH_NEAR_M:
            return "near"
        return "medium" if depth_m <= config.DEPTH_MED_M else "far"
    if h_ratio >= config.NEAR_THRESH.get(label, config.NEAR_THRESH["default"]) \
            and bottom >= config.NEAR_BOTTOM_MIN:
        return "near"
    if h_ratio >= config.MED_THRESH.get(label, config.MED_THRESH["default"]):
        return "medium"
    return "far"


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _person_bottom_in_chair(person_bbox, chair_bbox) -> bool:
    px1, py1, px2, py2 = person_bbox
    cx1, cy1, cx2, cy2 = chair_bbox
    bx, by = (px1 + px2) / 2, py2
    return cx1 <= bx <= cx2 and cy1 <= by <= cy2


def _chair_occupied(chair_bbox, person_bboxes) -> bool:
    for pb in person_bboxes:
        if _iou(chair_bbox, pb) >= config.CHAIR_OCCUPIED_IOU:
            return True
        if config.CHAIR_PERSON_BOTTOM_IN_CHAIR and \
                _person_bottom_in_chair(pb, chair_bbox):
            return True
    return False


def _compute_occupancy(objects: dict) -> None:
    """chair 객체에 occupied 플래그 설정 (내부 bbox 사용). in-place."""
    persons = [o["bbox"] for o in objects.values()
               if o["label"] == "person" and o.get("bbox")]
    for o in objects.values():
        if o["label"] != "chair":
            o.pop("occupied", None)
            continue
        if not o.get("bbox"):
            o["occupied"] = False
            continue
        o["occupied"] = _chair_occupied(o["bbox"], persons)


class SceneState:
    def __init__(self):
        self._lock = threading.Lock()
        self._objects = {}      # tid -> {label,pos,dist,status,bbox_h_ratio,misses,bbox?,occupied?}
        self._history = {}      # tid -> deque[(t, h_ratio)] 최근 이력 (접근율 계산)
        self._texts = []        # [{content,pos,ts}]
        self._caption = None    # 장면 묘사 질문 시에만 일시 설정

    def update_objects(self, detections) -> list:
        """detections: [{"track_id","label","pos","bbox_h_ratio", optional "bbox"}].
        상태 전이가 발생한 객체만 Event로 반환."""
        now = time.time()
        events = []
        with self._lock:
            seen = set()
            for d in detections:
                tid = d["track_id"]
                seen.add(tid)
                hist = self._history.setdefault(tid, deque(maxlen=120))
                hist.append((now, d["bbox_h_ratio"], d.get("depth_m")))
                # 접근 판정 1: bbox 크기 증가율 (0.8초 창)
                old = next(((t, r) for t, r, _ in hist if now - t >= 0.8), None)
                closing = old is not None and \
                    (d["bbox_h_ratio"] - old[1]) / (now - old[0]) > config.CLOSING_RATE
                # 접근 판정 2: depth 감소율 (2초 창 — depth는 1.5초 주기 갱신이라 창을 길게)
                # 실측 근거: 정면 접근자는 2.5m 안에 들어오기 전에 옆으로 비켜가므로
                # '가깝다'가 아니라 '거리가 줄고 있다'를 봐야 정면 접근을 조기에 잡는다.
                dm = d.get("depth_m")
                if not closing and dm is not None and dm <= config.DEPTH_CLOSING_MAX_M:
                    d_old = next(((t, dep) for t, _, dep in hist
                                  if dep is not None and now - t >= 2.0), None)
                    closing = d_old is not None and \
                        (d_old[1] - dm) / (now - d_old[0]) > config.DEPTH_CLOSING_MPS
                depth_m = d.get("depth_m")
                dist = _dist_of(d["label"], d["bbox_h_ratio"],
                                d.get("bottom", 1.0), depth_m)
                prev = self._objects.get(tid)
                if prev is None:
                    status = "new"
                    if dist == "near":
                        events.append(Event("new_near", d["label"], d["pos"],
                                            tid, depth_m))
                else:
                    if dist == "near" and prev["dist"] != "near":
                        events.append(Event("entered_near", d["label"], d["pos"],
                                            tid, depth_m))
                    if closing and prev["status"] != "approaching":
                        events.append(Event("approaching", d["label"], d["pos"],
                                            tid, depth_m))
                    status = "approaching" if closing else "seen"
                entry = {"label": d["label"], "pos": d["pos"],
                         "bbox_h_ratio": d["bbox_h_ratio"],
                         "depth_m": depth_m,
                         "dist": dist, "status": status, "misses": 0}
                if "bbox" in d:
                    entry["bbox"] = d["bbox"]
                self._objects[tid] = entry
            for tid in list(self._objects):
                if tid not in seen:
                    self._objects[tid]["misses"] += 1
                    if self._objects[tid]["misses"] >= config.GONE_AFTER_MISSES:
                        del self._objects[tid]
                        self._history.pop(tid, None)
            _compute_occupancy(self._objects)
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
                    # 시야에서 한동안 사라졌다 재등장한 표지판은 다시 '새 표지판'으로 알림
                    if now - ex["ts"] > config.TEXT_REANNOUNCE_GAP:
                        fresh.append(dict(it))
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
            objs = []
            for tid, o in self._objects.items():
                item = {"track_id": tid, "label": o["label"], "pos": o["pos"],
                        "dist": o["dist"], "status": o["status"],
                        "bbox_h_ratio": round(o["bbox_h_ratio"], 2)}
                if o.get("depth_m") is not None:
                    item["depth_m"] = round(o["depth_m"], 1)   # 스키마 선택 필드 (팀 합의)
                if o["label"] == "chair" and "occupied" in o:
                    item["occupied"] = o["occupied"]
                objs.append(item)
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

    # occupancy: person–chair 겹침 → occupied
    scene2 = SceneState()
    scene2.update_objects([
        {"track_id": 10, "label": "chair", "pos": "left", "bbox_h_ratio": 0.4,
         "bbox": [100, 200, 200, 350]},
        {"track_id": 11, "label": "person", "pos": "left", "bbox_h_ratio": 0.5,
         "bbox": [110, 150, 190, 340]},
    ])
    snap2 = json.loads(scene2.snapshot_json())
    chairs2 = {o["track_id"]: o for o in snap2["objects"] if o["label"] == "chair"}
    assert chairs2[10]["occupied"] is True, "겹친 chair가 empty로 나옴"
    assert "bbox" not in chairs2[10], "snapshot에 bbox가 노출됨"
    assert all("occupied" not in o for o in snap2["objects"] if o["label"] != "chair")

    # occupancy: chair만 → empty
    scene3 = SceneState()
    scene3.update_objects([
        {"track_id": 20, "label": "chair", "pos": "center", "bbox_h_ratio": 0.35,
         "bbox": [250, 200, 350, 340]},
    ])
    snap3 = json.loads(scene3.snapshot_json())
    assert snap3["objects"][0]["occupied"] is False, "빈 chair가 occupied로 나옴"

    # occupancy: 혼재 — left empty, center occupied, right empty
    scene4 = SceneState()
    scene4.update_objects([
        {"track_id": 30, "label": "chair", "pos": "left", "bbox_h_ratio": 0.4,
         "bbox": [50, 200, 140, 340]},
        {"track_id": 31, "label": "chair", "pos": "center", "bbox_h_ratio": 0.4,
         "bbox": [250, 200, 340, 340]},
        {"track_id": 32, "label": "chair", "pos": "right", "bbox_h_ratio": 0.4,
         "bbox": [450, 200, 540, 340]},
        {"track_id": 33, "label": "person", "pos": "center", "bbox_h_ratio": 0.5,
         "bbox": [260, 140, 330, 330]},
    ])
    snap4 = json.loads(scene4.snapshot_json())
    by_id = {o["track_id"]: o for o in snap4["objects"] if o["label"] == "chair"}
    assert by_id[30]["occupied"] is False and by_id[32]["occupied"] is False
    assert by_id[31]["occupied"] is True, "앉아있는 center chair가 empty"
    print("occupancy PASS")
    print("PASS")
