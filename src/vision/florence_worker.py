"""Florence-2 워커 — 2.5초 주기 OCR + on-demand 캡션. 매 프레임 실행 금지 (하드 룰 2).

MPS + fp32 고정 (fp16은 빈 출력 버그, 실측 확인). transformers==4.49.0 필수.
"""
import time

import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from src import config


class Florence:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            config.FLORENCE_MODEL, trust_remote_code=True,
            torch_dtype=torch.float32).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            config.FLORENCE_MODEL, trust_remote_code=True)

    def _run(self, image, task):
        inputs = self.processor(text=task, images=image,
                                return_tensors="pt").to(self.device)
        gen = self.model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=128, num_beams=1, do_sample=False,
            early_stopping=False)   # num_beams=1과 충돌하는 기본값 경고 억제
        raw = self.processor.batch_decode(gen, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(
            raw, task=task, image_size=image.size)[task]

    @staticmethod
    def _to_pil(frame_bgr):
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def ocr(self, frame_bgr) -> list:
        """[{"content","pos"}] — 2글자 미만/영숫자 없는 조각은 필터."""
        img = self._to_pil(frame_bgr)
        parsed = self._run(img, "<OCR_WITH_REGION>")
        items = []
        for label, quad in zip(parsed["labels"], parsed["quad_boxes"]):
            content = label.replace("</s>", "").strip()
            if len(content) < 2 or not any(c.isalnum() for c in content):
                continue
            cx = sum(quad[0::2]) / 4
            w = img.size[0]
            pos = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
            items.append({"content": content, "pos": pos})
        return items

    def caption(self, frame_bgr) -> str:
        """장면 묘사 질문 시 on-demand 1회만 (주기 실행 금지)."""
        return str(self._run(self._to_pil(frame_bgr),
                             "<MORE_DETAILED_CAPTION>")).strip()


def run_loop(camera, scene, on_new_texts, stop_flag, florence):
    """FLORENCE_PERIOD_SEC마다 최신 프레임 1장 OCR → 처음 보는 텍스트는 콜백."""
    while not stop_flag.is_set():
        frame = camera.latest()
        if frame is not None:
            try:
                fresh = scene.update_texts(florence.ocr(frame))
                if fresh:
                    on_new_texts(fresh)
            except Exception as e:
                print(f"[florence] 오류 (계속 진행): {e}")
        stop_flag.wait(config.FLORENCE_PERIOD_SEC)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "_prep/tools/images/bus.jpg"
    fl = Florence()
    frame = cv2.imread(path)
    t0 = time.time()
    print(f"OCR: {fl.ocr(frame)} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    print(f"caption: {fl.caption(frame)!r} ({time.time()-t0:.1f}s)")
