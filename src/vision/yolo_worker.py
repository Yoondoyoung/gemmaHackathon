"""YOLO 탐지 루프 (~10+ FPS): 프레임 → 트래킹 → SceneState → 룰베이스 경고."""
import time

import cv2

from src import config
from src.scene_state import _dist_of


def _bbox_depth(depth_entry, x1, y1, x2, y2, w, h):
    """깊이 맵에서 bbox 중앙 50% 영역의 중앙값(미터). 깊이 맵 없으면 None."""
    if depth_entry is None:
        return None
    import numpy as np
    dmap, _ = depth_entry
    dh, dw = dmap.shape
    qx, qy = (x2 - x1) / 4, (y2 - y1) / 4          # 가장자리 25% 제외 (배경 오염 방지)
    ix1, ix2 = int((x1 + qx) * dw / w), int((x2 - qx) * dw / w)
    iy1, iy2 = int((y1 + qy) * dh / h), int((y2 - qy) * dh / h)
    region = dmap[max(iy1, 0):max(iy2, 1), max(ix1, 0):max(ix2, 1)]
    return float(np.median(region)) if region.size else None


def load_model():
    from ultralytics import YOLO
    try:
        return YOLO(config.YOLO_WEIGHTS)
    except Exception:
        return YOLO(config.YOLO_FALLBACK)


def run_loop(camera, scene, alert_engine, speaker, stop_flag, shared):
    """shared: {"overlay": np.ndarray, "fps": float} — main이 그리기용으로 읽음."""
    model = load_model()
    while not stop_flag.is_set():
        frame = camera.latest()
        if frame is None:
            time.sleep(0.05)
            continue
        t0 = time.time()
        res = model.track(frame, persist=True, device=config.YOLO_DEVICE,
                          conf=config.YOLO_CONF, verbose=False,
                          tracker="bytetrack.yaml")[0]
        h, w = res.orig_shape
        depth_entry = shared.get("depth")
        detections = []
        for box in res.boxes:
            if box.id is None:          # 트래킹 미확정 박스는 스킵
                continue
            label = res.names[int(box.cls)]
            if label not in config.TRACK_LABELS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            pos = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
            detections.append({"track_id": int(box.id), "label": label,
                               "pos": pos, "bbox_h_ratio": (y2 - y1) / h,
                               "bbox": [x1, y1, x2, y2],
                               "bottom": y2 / h,
                               "depth_m": _bbox_depth(depth_entry, x1, y1, x2, y2, w, h),
                               "_xy": (int(x1), int(y1))})
        for sentence in alert_engine.process(scene.update_objects(detections)):
            speaker.say(sentence, priority=0)
        overlay = res.plot()
        for d in detections:      # 거리 판정 표시 (리허설 캘리브레이션용)
            dist = _dist_of(d["label"], d["bbox_h_ratio"], d["bottom"], d["depth_m"])
            tag = f"{dist} {d['depth_m']:.1f}m" if d["depth_m"] is not None \
                else f"{dist} h={d['bbox_h_ratio']:.2f}"
            cv2.putText(overlay, tag, (d["_xy"][0], max(d["_xy"][1] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
        shared["overlay"] = overlay
        shared["fps"] = 1.0 / max(time.time() - t0, 1e-6)


if __name__ == "__main__":
    # 단독 테스트: 비디오/웹캠에서 탐지+경고가 도는지 (경고는 print로)
    import sys
    import threading
    from src.alerts import AlertEngine
    from src.scene_state import SceneState
    from src.vision.camera import Camera

    class PrintSpeaker:
        def say(self, text, priority=1):
            print(f"[ALERT p{priority}] {text}")

    cam = Camera(sys.argv[1] if len(sys.argv) > 1 else config.CAM_INDEX)
    cam.start()
    scene, shared, stop = SceneState(), {}, threading.Event()
    t = threading.Thread(target=run_loop, daemon=True,
                         args=(cam, scene, AlertEngine(), PrintSpeaker(), stop, shared))
    t.start()
    t0 = time.time()
    while time.time() - t0 < 20 and not cam.ended:
        time.sleep(1)
        print(f"  t={time.time()-t0:4.1f}s fps={shared.get('fps', 0):5.1f} "
              f"scene={scene.snapshot_json()[:120]}")
    stop.set()
    cam.stop()
