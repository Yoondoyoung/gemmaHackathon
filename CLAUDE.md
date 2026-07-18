# 프로젝트: 온디바이스 시각 보조 어시스턴트 (24h 해커톤)

**구현 작업 전 반드시 IMPLEMENTATION.md를 읽어라.** 그 문서가 유일한 구현 지시서다.
아키텍처/설계 배경은 PLAN.md, 팀 결정사항은 MEETING.md.

## 하드 룰 (재검토·개선 제안 금지)

1. 장애물 경고 경로에 LLM(Gemma) 금지 — 룰베이스 템플릿만. Gemma는 사용자 Q&A 전용.
2. Florence-2는 2.5초 주기 + on-demand만. 매 프레임 실행 금지. MPS + fp32 (fp16 금지 — 빈 출력 버그, 실측 확인).
3. threading만 사용 (asyncio 금지). `cv2.imshow`는 메인 스레드에서만.
4. requirements.txt 외 의존성 추가 금지.
5. SceneState JSON 스키마(IMPLEMENTATION.md §2)는 계약 — 필드 변경 금지.
6. 확장성 설계 금지 (추상 클래스, 플러그인, 테스트 프레임워크 등). 각 모듈의
   `__main__` 블록이 테스트의 전부. 24시간 해커톤 코드다.
7. 마일스톤 순서(M1→M6)를 지키고, 각 마일스톤의 완료 기준을 실행으로 확인한 뒤 다음으로.
8. 막히면 IMPLEMENTATION.md §6 함정 목록부터 확인.

## 환경

- M1 Air 16GB, Python 3.11, venv는 `.venv/`
- 실행 전: `source .venv/bin/activate`
- Ollama 필요: `ollama serve` 실행 중이어야 함 (모델: gemma3:4b)
- 환경 검증: `python _prep/tools/smoke_test.py`
- `_prep/`은 해커톤 이전 준비물(제출 불가) — 참고만 하고 수정/제출 금지. 새 코드는 전부 `src/`, `prompts/`에.
