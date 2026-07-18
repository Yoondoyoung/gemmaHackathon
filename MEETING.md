# 킥오프 미팅 아젠다 (해커톤 시작 전, ~1시간)

목표: 아래 7개 결정을 끝내고, 양쪽 다 `setup.sh` + 스모크 테스트 통과를 확인한 상태로 해커톤에 입장한다.
각 항목에 **제안(기본값)**이 있으니, 이견 없으면 그대로 채택하고 넘어갈 것.

---

## 결정 1. SceneState JSON 스키마 (가장 중요 — 두 사람의 인터페이스 계약)

**제안:**

```json
{
  "timestamp": 1720000000.0,
  "objects": [
    {
      "track_id": 3,
      "label": "person",
      "pos": "center",          // left | center | right  (bbox 중심 x, 화면 3등분)
      "dist": "near",           // near | medium | far    (bbox 높이 비율, 클래스별 임계값)
      "status": "approaching",  // new | approaching | seen | gone
      "bbox_h_ratio": 0.62
    }
  ],
  "texts": [
    {
      "content": "EXIT",
      "pos": "right",           // OCR region 중심 x 기준
      "age_sec": 4.2            // 마지막으로 읽힌 후 경과 시간 (30초 지나면 삭제)
    }
  ]
}
```

- 이 스키마가 확정되면: A는 이걸 채우는 코드를, B는 이걸 소비하는 코드를 **독립적으로** 짠다.
- B는 통합 전까지 `tools/fake_scene.json` 같은 가짜 데이터로 개발.
- 확정 후 변경하려면 반드시 상호 합의 (일방 변경 금지).

## 결정 2. 역할 분담

**제안:**

| 담당 | 범위 | 파일 |
|---|---|---|
| A (비전) | 카메라, YOLO+트래킹, Florence OCR, SceneState 생산, Alert 룰, 디버그 오버레이 | `src/vision/*`, `src/scene_state.py`, `src/alerts.py` |
| B (음성/LLM) | Ollama/Gemma 클라이언트, 프롬프트, STT(push-to-talk), TTS 큐, 우선순위/인터럽트 | `src/llm/*`, `src/audio/*` |
| 공동 | `src/main.py` (스레드 조립), 통합(H8), 리허설, 발표 | |

→ 누가 A/B 할지 미팅에서 결정: ___________

## 결정 3. 모듈 인터페이스 (함수 시그니처 수준 계약)

**제안:**

```python
# scene_state.py (A 구현, B 소비)
class SceneState:
    def update_objects(self, detections: list) -> list[Event]  # YOLO 루프가 호출
    def update_texts(self, ocr_results: list) -> None          # Florence 워커가 호출
    def snapshot_json(self) -> str                             # Gemma 프롬프트용

# alerts.py (A)
class AlertEngine:
    def process(self, events: list[Event]) -> list[str]        # 발화할 경고 문장 반환

# llm/gemma_client.py (B)
def ask(question: str, scene_json: str) -> str                 # 블로킹, 3~6초

# audio/tts.py (B)
class Speaker:
    def say(self, text: str, priority: int) -> None   # 0=경고(현재 발화 kill), 1=답변
    def pause(self) / resume(self)                    # 녹음 중 일시정지용

# audio/stt.py (B)
def record_and_transcribe() -> str                    # push-to-talk 토글 사이 녹음
```

## 결정 4. Gemma 프롬프트

**제안:** `prompts/system_prompt.txt` 초안 검토 (2문장 제한, 사용자 시점 방향, 기술용어 금지).
미팅에서 소리 내어 읽어보고 데모 시나리오 질문 2개에 대한 기대 응답을 합의할 것.

## 결정 5. 데모 시나리오 + 준비물

**제안 시나리오 (영어, 총 ~3분):**
1. 오프닝: "완전 오프라인" 강조 — **와이파이 끄고 시작** (심사위원 앞에서 토글)
2. 복도를 걸으며 장애물 경고 시연 (의자/사람 배치)
3. 표지판 앞에서: "I need to find the restroom, do you see any signs?" → Gemma가 OCR 인용 답변
4. 클로징: 디버그 오버레이 화면 보여주며 아키텍처 한 장 설명

**준비물 (미팅에서 담당자 지정):**
- [ ] "EXIT →" / "RESTROOM ←" 표지판 A4 인쇄물 (크고 굵은 산세리프, 2장씩 여분)
- [ ] 충전기 + 보조 배터리
- [ ] (옵션) 외장/유선 마이크 — 행사장 소음 대비
- [ ] (옵션) iPhone + 목걸이 거치대 — Continuity Camera 대안용
- [ ] 백업 데모 영상은 H20에 현장 녹화

## 결정 6. Git 규칙

**제안:** main 직커밋 (24h 해커톤에 PR 리뷰는 사치). 대신:
- 결정 2의 파일 소유권을 지키면 충돌이 구조적으로 안 남
- `main.py` 수정은 상대에게 말하고 할 것
- 최소 2시간마다 push (맥북 사망 대비)

## 결정 7. 카메라 방향

**제안:** 기본은 맥북을 화면이 바깥(전방)을 향하게 들기 — 심사위원이 디버그 오버레이를 보게 되는 연출 효과.
iPhone Continuity Camera는 H16 리허설에서 비교 후 결정. 지금은 결정 보류만 합의.

---

## 미팅 전 각자 숙제 (해커톤 전날까지)

1. `git clone` 후 `./setup.sh` 실행 (모델 ~4GB 다운로드 포함, 집 와이파이에서)
2. `python tools/smoke_test.py` 전 항목 PASS 스크린샷을 서로 공유
3. 카메라/마이크 권한 팝업이 이때 뜸 — 허용해둘 것 (데모 중 팝업 방지)
4. PLAN.md 정독
5. **테스트 클립 촬영** (폰을 가슴 높이, 각 1~2분): 복도 보행 / 사람 접근 / 의자 지나침 /
   EXIT 표지판 접근 / (가능하면) 횡단보도. `clips/`에 넣기 — `--video` 주입 테스트용.
   대규모 연구 데이터셋(Ego4D 등)은 라이선스·용량 문제로 쓰지 않기로 결정됨

## 스모크 테스트 트러블슈팅

| 증상 | 대응 |
|---|---|
| Florence-2 실패 (`forced_bos_token_id` / `flash_attn` 에러) | `pip install "transformers==4.49.0"` 후 재시도 (4.50+에서 깨짐, 실측 확인) |
| Florence-2 OCR 결과가 빈 리스트 | fp16으로 로드한 것 — fp32로 (실측 확인된 버그) |
| YOLO26 가중치 없음 | 자동으로 yolo11n 폴백됨 — 그대로 진행 (API 동일) |
| Ollama connection refused | `ollama serve` 별도 터미널에서 실행 후 재시도 |
| 카메라/마이크 FAIL | 시스템 설정 > 개인정보 보호 > 카메라/마이크에서 터미널 허용 |
| Gemma < 10 tok/s | 다른 앱 종료 후 재측정, 그래도 느리면 `ollama pull gemma3n:e2b`로 교체 검토 |
