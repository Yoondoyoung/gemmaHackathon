"""통합 실행. cv2.imshow/waitKey는 반드시 이 메인 스레드에서만 (하드 룰 3).

키: SPACE=push-to-talk 토글, t=텍스트 질문(STT 폴백), q=종료
실행: python -m src.main [--video clips/x.mp4] [--no-florence] [--mute]
"""
import argparse
import os
import threading
import time
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # say 서브프로세스 fork 경고 방지
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import cv2

from src import config
from src.alerts import AlertEngine
from src.audio.stt import Recorder, warmup as whisper_warmup
from src.audio.tts import Speaker
from src.llm import gemma_client
from src.scene_state import SceneState
from src.vision import yolo_worker
from src.vision.camera import Camera

DESCRIBE_WORDS = ("describe", "what do you see", "around me", "in front of")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="웹캠 대신 비디오 파일 (테스트 하네스)")
    ap.add_argument("--no-florence", action="store_true", help="OCR 비활성 (P0만)")
    ap.add_argument("--mute", action="store_true", help="TTS를 print로만")
    args = ap.parse_args()

    camera = Camera(args.video if args.video else config.CAM_INDEX)
    camera.start()
    scene = SceneState()
    speaker = Speaker(enabled=not args.mute)
    recorder = Recorder()
    stop = threading.Event()
    shared = {}
    state = {"goal": None, "busy": False, "last_qa": ""}

    # 워밍업 (콜드로드로 첫 질문만 10초+ 걸리는 것 방지)
    threading.Thread(target=gemma_client.warmup, daemon=True).start()
    threading.Thread(target=whisper_warmup, daemon=True).start()

    threading.Thread(target=yolo_worker.run_loop, daemon=True,
                     args=(camera, scene, AlertEngine(), speaker, stop, shared)
                     ).start()

    last_announce = [0.0]

    def on_new_texts(fresh):
        """처음 보는 표지판: 1회 알림 + 목표 매칭 (룰베이스, LLM 미경유).
        긴/깨진 텍스트는 SceneState에 저장만 하고 낭독하지 않는다 (목표 매칭은 전부 검사)."""
        for it in fresh:
            spoken_pos = config.POS_SPOKEN[it["pos"]]
            goal = state["goal"]
            if goal and any(kw in it["content"].lower() for kw in goal):
                speaker.say(f"Found it — a sign for {it['content']}, {spoken_pos}", 1)
                state["goal"] = None
                continue
            if not config.ANNOUNCE_NEW_SIGNS:
                continue
            if len(it["content"].split()) > config.ANNOUNCE_MAX_WORDS \
                    or len(it["content"]) > config.ANNOUNCE_MAX_CHARS:
                continue
            if time.time() - last_announce[0] < config.ANNOUNCE_MIN_INTERVAL:
                continue
            last_announce[0] = time.time()
            speaker.say(f"Sign detected: {it['content']}, {spoken_pos}", 1)

    florence = None
    if not args.no_florence:
        def start_florence():
            nonlocal florence
            from src.vision.florence_worker import Florence, run_loop
            try:
                fl = Florence()
            except Exception as e:
                print(f"[florence] 로드 실패 — OCR 없이 진행 (P0 유지): {e}")
                return
            florence = fl
            print("[florence] 준비 완료")
            run_loop(camera, scene, on_new_texts, stop, fl)
        threading.Thread(target=start_florence, daemon=True).start()

    def handle_question(question):
        try:
            if not question:
                speaker.say("Sorry, I didn't catch that.", 1)
                return
            print(f"[Q] {question}")
            if florence and any(k in question.lower() for k in DESCRIBE_WORDS):
                frame = camera.latest()          # 장면 묘사류만 캡션 1회 (on-demand)
                if frame is not None:
                    try:
                        scene.set_caption(florence.caption(frame))
                    except Exception:
                        pass
            t0 = time.time()
            answer = gemma_client.ask_streaming(
                question, scene.snapshot_json(), lambda s: speaker.say(s, 1))
            scene.set_caption(None)
            state["last_qa"] = f"{time.time()-t0:.1f}s"
            if config.GOAL_ENABLED:
                kws = gemma_client.extract_goal(question)
                if kws:
                    state["goal"] = kws
                    print(f"[goal 설정] {kws}")
        finally:
            state["busy"] = False

    def ask_async(question):
        if state["busy"]:
            return
        state["busy"] = True
        threading.Thread(target=handle_question, args=(question,),
                         daemon=True).start()

    last_space = [0.0]
    print("SPACE=음성 질문 토글 | t=텍스트 질문 | q=종료")
    while True:
        overlay = shared.get("overlay")
        if overlay is None:
            overlay = camera.latest()
        if overlay is None:
            time.sleep(0.05)               # 카메라 준비 전 busy loop 방지
        else:
            y = 20
            for line in (f"fps={shared.get('fps', 0):.0f}  "
                         f"goal={state['goal'] or '-'}  qa={state['last_qa']}",
                         "REC..." if recorder.recording else ""):
                if line:
                    cv2.putText(overlay, line, (8, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    y += 22
            cv2.imshow("assist", overlay)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and time.time() - last_space[0] > 0.4:  # 키 반복 디바운스
            last_space[0] = time.time()
            if not recorder.recording:
                speaker.pause()                 # 자기 TTS 녹음 방지
                recorder.start()
                print("[REC] 녹음 중 — SPACE로 종료")
            else:
                print("[REC] 전사 중...")
                def finish():                   # 전사(2~4초)는 UI를 멈추지 않도록 별도 스레드
                    question = recorder.stop_and_transcribe()
                    speaker.resume()
                    ask_async(question)
                threading.Thread(target=finish, daemon=True).start()
        if key == ord("t"):
            ask_async(input("질문 입력: ").strip())
        if camera.ended:                        # 비디오 모드: 재생 끝나면 종료
            print("[video] 재생 종료")
            time.sleep(1)
            break

    stop.set()
    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
