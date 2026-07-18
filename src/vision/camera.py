"""카메라 캡처 스레드. 항상 '최신 프레임 1장'만 유지 (버퍼링 금지 — 지연 누적 방지).

source가 int면 웹캠, str이면 비디오 파일 (원본 FPS에 맞춰 재생 — 테스트 하네스용).
"""
import threading
import time

import cv2

from src import config


class Camera:
    def __init__(self, source=config.CAM_INDEX):
        self.source = source
        self.is_video = isinstance(source, str)
        self.ended = False           # 비디오 모드에서 재생 종료
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"카메라/비디오 열기 실패: {self.source}")
        if not self.is_video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        def loop():
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.is_video:
                        self.ended = True
                        break
                    time.sleep(0.05)
                    continue
                with self._lock:
                    self._frame = frame
                if self.is_video:
                    time.sleep(1.0 / fps)
            cap.release()

        threading.Thread(target=loop, daemon=True).start()

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()


if __name__ == "__main__":
    import sys
    cam = Camera(sys.argv[1] if len(sys.argv) > 1 else config.CAM_INDEX)
    cam.start()
    t0, n = time.time(), 0
    while time.time() - t0 < 3 and not cam.ended:
        if cam.latest() is not None:
            n += 1
        time.sleep(0.05)
    cam.stop()
    print(f"3초간 latest() 성공 {n}회, ended={cam.ended}")
