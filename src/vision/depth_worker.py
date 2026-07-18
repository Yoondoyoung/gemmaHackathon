"""미터 단위 깊이 워커 — Depth Anything V2 (metric indoor small, MPS 192ms/frame 실측).

1초 주기로 최신 프레임의 깊이 맵(미터)을 shared["depth"]에 저장한다.
YOLO 루프가 bbox 중앙 영역의 중앙값을 읽어 객체 거리로 사용.
로드 실패 시 조용히 종료 — 거리 판정은 bbox 휴리스틱으로 자동 폴백.
"""
import time

import cv2
from PIL import Image

from src import config


def run_loop(camera, shared, stop_flag):
    try:
        import torch
        from transformers import pipeline
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe = pipeline("depth-estimation", model=config.DEPTH_MODEL,
                        device=device)
    except Exception as e:
        print(f"[depth] 로드 실패 — bbox 휴리스틱으로 폴백: {e}")
        return
    print("[depth] 준비 완료")
    while not stop_flag.is_set():
        t0 = time.time()
        if shared.get("llm_busy"):     # Gemma 응답 중엔 GPU 양보
            stop_flag.wait(0.5)
            continue
        frame = camera.latest()
        if frame is not None:
            try:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                out = pipe(img)
                depth_m = out["predicted_depth"].squeeze().float().numpy()
                shared["depth"] = (depth_m, frame.shape[:2])  # (맵, (H, W))
            except Exception as e:
                print(f"[depth] 오류 (계속 진행): {e}")
        stop_flag.wait(max(0.1, config.DEPTH_PERIOD_SEC - (time.time() - t0)))
