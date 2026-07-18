"""룰베이스 경고 엔진 — LLM 절대 금지 (하드 룰 1). 템플릿 문장만 반환."""
import time

from src import config


class AlertEngine:
    def __init__(self):
        self._last_by_label = {}   # 라벨 기준 쿨다운 — 트래킹 ID 재할당(churn)에도 스팸 방지
        self._last_global = -1e9

    def process(self, events) -> list:
        """상태 전이 Event 목록 → 발화할 경고 문장 (한 번에 최대 1건)."""
        now = time.time()
        cands = [e for e in events if e.pos == "center"]
        # near 진입이 approaching보다 우선
        cands.sort(key=lambda e: 0 if e.kind in ("new_near", "entered_near") else 1)
        for e in cands:
            if now - self._last_by_label.get(e.label, -1e9) < config.ALERT_COOLDOWN_SEC:
                continue
            if now - self._last_global < config.ALERT_GLOBAL_INTERVAL:
                continue
            self._last_by_label[e.label] = now
            self._last_global = now
            if e.kind in ("new_near", "entered_near"):
                return [f"{e.label} ahead, close"]
            return [f"{e.label} approaching"]
        return []


if __name__ == "__main__":
    from src.scene_state import Event
    eng = AlertEngine()
    e = Event("entered_near", "person", "center", 1)
    print("1차:", eng.process([e]))                    # 발화
    print("즉시 재시도(쿨다운):", eng.process([e]))       # 억제
    print("좌측(무시):", eng.process([Event("entered_near", "chair", "left", 2)]))
    assert eng.process([e]) == []
    print("PASS")
