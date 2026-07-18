"""push-to-talk 녹음 + whisper 전사. WhisperModel은 프로세스에서 1회만 로드."""
import numpy as np
import sounddevice as sd

from src import config

_model = None


def warmup():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu",
                              compute_type="int8")
    return _model


class Recorder:
    def __init__(self):
        self._chunks = []
        self._stream = None

    @property
    def recording(self):
        return self._stream is not None

    def start(self):
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
            callback=lambda data, *_: self._chunks.append(data.copy()))
        self._stream.start()

    def stop_and_transcribe(self) -> str:
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._chunks:
            return ""
        audio = np.concatenate(self._chunks)[:, 0]
        if len(audio) < config.SAMPLE_RATE:    # 1초 미만은 오조작으로 간주
            return ""
        # vad_filter: 무음/잡음 구간에서 whisper가 문장을 지어내는 환각 방지 (실측)
        segments, _ = warmup().transcribe(audio, vad_filter=True,
                                          condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


if __name__ == "__main__":
    import time
    print("3초 녹음 시작 — 마이크에 말하세요...")
    rec = Recorder()
    rec.start()
    time.sleep(3.5)
    text = rec.stop_and_transcribe()
    print(f"전사 결과: {text!r}")
