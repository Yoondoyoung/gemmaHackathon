"""통합 실행. cv2.imshow/waitKey는 반드시 이 메인 스레드에서만 (하드 룰 3).

키: SPACE=목표 키워드만 설정(PTT), b=일반 질문 음성(PTT), t=텍스트 폴백, q=종료
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
EMPTY_SEAT_WORDS = ("empty seat", "free chair", "vacant seat", "vacant chair",
                    "where can i sit", "find a seat", "find an empty",
                    "빈 자리", "빈자리", "빈 의자")


def _is_empty_seat_question(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in EMPTY_SEAT_WORDS)


def worth_announcing(item) -> bool:
    """내비게이션 어휘이거나 화면에서 크게 보이는 텍스트만 발화 가치 있음.
    (브랜드 로고/제품 라벨 같은 작은 잡글자는 저장만 — Q&A·목표 매칭에는 사용됨)"""
    content = item["content"]
    if len(content.split()) > config.ANNOUNCE_MAX_WORDS \
            or len(content) > config.ANNOUNCE_MAX_CHARS:
        return False
    words = {w.strip(".,:;!?<>→←-").lower() for w in content.split()}
    if words & config.NAV_SIGN_WORDS:
        return True
    return item.get("h_ratio", 0) >= config.SIGN_MIN_H_RATIO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="웹캠 대신 비디오 파일 (테스트 하네스)")
    ap.add_argument("--no-florence", action="store_true", help="OCR 비활성 (P0만)")
    ap.add_argument("--no-depth", action="store_true", help="미터 깊이 비활성")
    ap.add_argument("--mute", action="store_true", help="TTS를 print로만")
    args = ap.parse_args()

    camera = Camera(args.video if args.video else config.CAM_INDEX)
    camera.start()
    scene = SceneState()
    speaker = Speaker(enabled=not args.mute)
    recorder = Recorder()
    stop = threading.Event()
    shared = {}
    # rec_mode: None | "goal"(SPACE) | "ask"(B)
    state = {"goal": None, "busy": False, "last_qa": "", "rec_mode": None}

    # 워밍업 (콜드로드로 첫 질문만 10초+ 걸리는 것 방지)
    threading.Thread(target=gemma_client.warmup, daemon=True).start()
    threading.Thread(target=whisper_warmup, daemon=True).start()

    threading.Thread(target=yolo_worker.run_loop, daemon=True,
                     args=(camera, scene, AlertEngine(), speaker, stop, shared)
                     ).start()

    if config.DEPTH_ENABLED and not args.no_depth:
        from src.vision import depth_worker
        threading.Thread(target=depth_worker.run_loop, daemon=True,
                         args=(camera, shared, stop)).start()

    last_announce = [0.0]

    def on_new_texts(fresh):
        """처음 보는 표지판: 1회 알림 + 목표 매칭 (룰베이스, LLM 미경유).
        긴/깨진 텍스트는 SceneState에 저장만 하고 낭독하지 않는다 (목표 매칭은 전부 검사)."""
        import re
        for it in fresh:
            spoken_pos = config.POS_SPOKEN[it["pos"]]
            goal = state["goal"]
            # 단어 경계 매칭 — "men"이 EQUIPMENT에 걸리는 오탐 방지
            if goal and any(re.search(rf"\b{re.escape(kw)}\b", it["content"].lower())
                            for kw in goal):
                speaker.say(f"Found it — a sign for {it['content']}, {spoken_pos}", 1)
                state["goal"] = None
                continue
            if not config.ANNOUNCE_NEW_SIGNS or not worth_announcing(it):
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

    def handle_question(question, set_goal=False):
        try:
            if not question:
                speaker.say("Sorry, I didn't catch that.", 1)
                return
            print(f"[Q] {question}" + (" (goal)" if set_goal else " (ask)"))
            t0 = time.time()

            # SPACE(목표): 장면 Q&A 없이 키워드만 설정 → 이후 Florence 표지판 매칭
            if set_goal:
                if not config.GOAL_ENABLED:
                    speaker.say("Goal tracking is off.", 1)
                    return
                kws = gemma_client.extract_goal(question)
                if kws:
                    state["goal"] = kws
                    print(f"[goal 설정] {kws}")
                    speaker.say(f"Looking for {kws[0]}.", 1)
                else:
                    speaker.say("I couldn't tell what you're looking for.", 1)
                state["last_qa"] = f"{time.time()-t0:.1f}s"
                return

            # B(일반 질문): 장면 스냅샷으로 Gemma 답변 (extract_goal 없음)
            if florence and any(k in question.lower() for k in DESCRIBE_WORDS):
                frame = camera.latest()
                if frame is not None:
                    try:
                        scene.set_caption(florence.caption(frame))
                    except Exception:
                        pass
            snap = scene.snapshot_json()
            if _is_empty_seat_question(question):
                answer = gemma_client.ask_streaming(question, snap, lambda _s: None)
                speaker.say(answer.strip() if answer.strip()
                            else "Sorry, I couldn't process that.", 1)
            else:
                gemma_client.ask_streaming(
                    question, snap, lambda s: speaker.say(s, 1))
            scene.set_caption(None)
            state["last_qa"] = f"{time.time()-t0:.1f}s"
        finally:
            state["busy"] = False

    def ask_async(question, set_goal=False):
        if state["busy"]:
            return
        state["busy"] = True
        threading.Thread(target=handle_question, args=(question, set_goal),
                         daemon=True).start()

    def toggle_rec(mode):
        """mode: 'goal'(SPACE) | 'ask'(B). 녹음 중에는 시작한 키로만 종료."""
        if recorder.recording:
            if state["rec_mode"] != mode:
                return
            print("[REC] 전사 중...")
            set_goal = mode == "goal"
            state["rec_mode"] = None

            def finish():
                question = recorder.stop_and_transcribe()
                speaker.resume()
                ask_async(question, set_goal=set_goal)
            threading.Thread(target=finish, daemon=True).start()
            return
        if state["busy"]:
            return
        speaker.pause()
        recorder.start()
        state["rec_mode"] = mode
        end_key = "SPACE" if mode == "goal" else "b"
        print(f"[REC] 녹음 중 ({mode}) — {end_key}로 종료")

    last_key = [0.0]
    print("SPACE=목표 음성 | b=일반 질문 음성 | t=텍스트 | q=종료")
    while True:
        overlay = shared.get("overlay")
        if overlay is None:
            overlay = camera.latest()
        if overlay is None:
            time.sleep(0.05)               # 카메라 준비 전 busy loop 방지
        else:
            y = 20
            rec = ""
            if recorder.recording:
                rec = f"REC({state['rec_mode']})..."
            for line in (f"fps={shared.get('fps', 0):.0f}  "
                         f"goal={state['goal'] or '-'}  qa={state['last_qa']}",
                         rec):
                if line:
                    cv2.putText(overlay, line, (8, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    y += 22
            cv2.imshow("assist", overlay)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key in (ord(" "), ord("b")) and time.time() - last_key[0] > 0.4:
            last_key[0] = time.time()
            toggle_rec("goal" if key == ord(" ") else "ask")
        if key == ord("t"):
            ask_async(input("질문 입력: ").strip(), set_goal=False)
        if camera.ended:                        # 비디오 모드: 재생 끝나면 종료
            print("[video] 재생 종료")
            time.sleep(1)
            break

    stop.set()
    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
