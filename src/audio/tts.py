"""발화 큐. priority 0=경고(현재 답변 발화를 끊고 즉시), 1=Gemma 답변/알림.

macOS `say`를 Popen으로 실행 — 핸들을 보관해야 경고가 끼어들 때 kill 가능.
녹음 중에는 pause() — 자기 TTS를 whisper가 받아적는 루프 방지.
"""
import itertools
import queue
import subprocess
import threading
import time

from src import config


class Speaker:
    def __init__(self, enabled=True):
        self.enabled = enabled          # False면 print만 (자동 테스트용)
        self._q = queue.PriorityQueue() # (priority, seq, text)
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._current = None            # (priority, Popen)
        self._paused = False
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text, priority=1):
        if self._paused and priority == 1:
            return                       # 녹음 중 답변류는 폐기
        if priority == 0:
            with self._lock:             # 경고: 진행 중인 답변 발화를 끊는다
                if self._current and self._current[0] > 0:
                    self._current[1].kill()
            self._drop_answers()
        self._q.put((priority, next(self._seq), text))

    def _drop_answers(self):
        keep = []
        try:
            while True:
                item = self._q.get_nowait()
                if item[0] == 0:
                    keep.append(item)
        except queue.Empty:
            pass
        for item in keep:
            self._q.put(item)

    def pause(self):
        """녹음 시작 시 호출. 진행 중인 답변 발화는 끊고, 경고는 큐에 남는다."""
        self._paused = True
        with self._lock:
            if self._current and self._current[0] > 0:
                self._current[1].kill()

    def resume(self):
        self._paused = False

    def _worker(self):
        while True:
            if self._paused:
                time.sleep(0.1)
                continue
            try:
                prio, _, text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            print(f"[TTS p{prio}] {text}")
            if not self.enabled:
                continue
            proc = subprocess.Popen(["say", "-v", config.TTS_VOICE, text])
            with self._lock:
                self._current = (prio, proc)
            proc.wait()
            with self._lock:
                self._current = None


if __name__ == "__main__":
    sp = Speaker(enabled=True)
    sp.say("This is a long answer sentence that should be interrupted.", 1)
    time.sleep(1.0)
    sp.say("person ahead", 0)   # 답변을 끊고 나와야 함
    time.sleep(4)
    print("PASS (경고가 답변을 끊고 발화됐으면 정상)")
