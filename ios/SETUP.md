# iOS 앱 빌드 가이드 (GemmaVision)

완성된 Xcode 프로젝트가 `ios/GemmaVision/`에 있습니다. 새로 만들 것 없이 클론 → 열기 → 실행이면 됩니다.

## 1. 클론 후 열기
```bash
git clone <이 저장소 URL>
cd pahacker && git checkout doyoung   # 작업 브랜치
open ios/GemmaVision/GemmaVision.xcodeproj
```
- YOLO26 모델(`yolo26n.mlpackage`, 2.6MB)은 저장소에 포함되어 있어 별도 작업 불필요
- LiteRT-LM(`Vendor/LiteRT-LM`)은 로컬 SPM 패키지로 이미 프로젝트에 연결되어 있음.
  단 실제 프레임워크 바이너리는 Package.swift가 구글 릴리스에서 원격으로 받아오므로
  **Xcode를 처음 열 때 인터넷 연결 필요** (Resolve Package Versions 자동 진행, 1~2분)

## 2. Gemma 4 E2B 모델 파일 받기 (필수, ~2.4GB — 깃허브에는 없음)

깃허브 파일 크기 제한 때문에 이 파일만 별도로 받아야 합니다.

**행사장에서 가장 빠른 방법 — AirDrop:** 원본 보유 맥북에서
`ios/GemmaVision/GemmaVision/gemma-4-E2B-it.litertlm`을 AirDrop/케이블로 받아서
Xcode 프로젝트 내 같은 경로(`GemmaVision/GemmaVision/` 폴더)에 넣고 Xcode에 드래그
(Copy items if needed + 타깃 멤버십 체크).

**인터넷으로 새로 받는 경우:**
```bash
source .venv/bin/activate   # 없으면 ../setup.sh 먼저 실행
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('litert-community/gemma-4-E2B-it-litert-lm', \
'gemma-4-E2B-it.litertlm', local_dir='ios/GemmaVision/GemmaVision')"
```
(Apache 2.0, 게이팅 없음. `-web`이나 `_qualcomm...` 등 변형 파일은 iOS용이 아니니
접미사 없는 base 파일만 받을 것)

## 3. 서명
1. Xcode → GemmaVision 타깃 → Signing & Capabilities → **Team을 본인 Apple ID로 변경**
   (Bundle Identifier가 겹치면 뒤에 아무 문자열 추가, 예: `...gemmavision.본인이니셜`)
2. 카메라/마이크/음성인식 권한 문구는 이미 빌드 설정에 포함되어 있어 추가 작업 불필요

## 4. 실기기 실행
1. iPhone 연결 → 상단 기기 선택 → ⌘R
2. 폰에서: 설정 → 일반 → VPN 및 기기 관리 → 개발자 앱 신뢰
3. 첫 실행 시 카메라/마이크/음성인식 권한 팝업 허용

## 검증 체크리스트
- [ ] 실행 직후 "Vision assist started, hold the button to ask" 발화
- [ ] 잠시 후 "Assistant ready on GPU" (또는 CPU) — Gemma 모델 로드 완료 신호
- [ ] 화면에 탐지 박스 표시 (초록 = 일반, **빨강 = 경고 대상**, 라벨에 거리)
- [ ] 사람/의자 등에 가까이 다가가면 "person ahead, N meters"(LiDAR 기기) 또는
      "person ahead, close"(비 LiDAR) — 1회, 5초 쿨다운
- [ ] EXIT 등 표지판 인쇄물 비추면 ~3초 내 "Sign detected: EXIT, ahead of you"
- [ ] 화면 하단 **Hold to talk** 버튼을 누른 채 질문 → 떼면 Gemma가 현재 장면
      (+프레임 이미지) 기반으로 답변, 문장 단위로 스트리밍 발화
- [ ] (횡단보도 있으면) 신호등 근처에서 "The light is red/green"
- [ ] LiDAR 기기(iPhone 12 Pro 이상)는 오버레이 라벨이 `2.3m` 형태 실측 미터,
      아니면 `h=0.42` 휴리스틱 — 코드 수정 없이 기기에 따라 자동 전환

## 영상 파일로 오프라인 테스트 (카메라 없이)
1. `../clips/*.mp4`(nyc2.mp4, cross1.mp4 등) 중 하나를 프로젝트에 드래그
2. `VisionAssistApp.swift`에서 `ContentView()`를 주석 처리하고
   `VideoTestView(clip: "nyc2.mp4")`로 교체 후 실행 (시뮬레이터에서도 가능)
3. 영상엔 LiDAR가 없어 거리는 휴리스틱(`h=`)으로 표시됨 — 탐지/경고/OCR 로직
   검증용. 실기기 라이브에서 박스가 어긋나면 이 모드로 먼저 로직 정상 여부를 분리 확인

## 트러블슈팅
| 증상 | 원인/대응 |
|---|---|
| 실행 중 `MLIR pass manager failed`로 크래시 | 일부 기기에서 CoreML GPU 컴파일 assertion. `Pipeline.swift`의 `computeUnits`가 이미 `.cpuAndNeuralEngine`으로 회피 처리됨 — 그래도 나면 `.cpuOnly`로 |
| Gemma 로드 실패/한없이 loading | `.litertlm` 파일이 번들에 없음 (2단계 확인) |
| SPM resolve 실패 | 오프라인 상태 — Vendor/LiteRT-LM이 최초 1회 인터넷에서 바이너리를 받아야 함 |
| 카메라 안 열림 | 시뮬레이터에는 카메라가 없음 — 실기기 필요 (영상 테스트 모드는 시뮬레이터 가능) |

## 아키텍처 메모 (발표 대비)
탐지는 Neural Engine(YOLO26 CoreML), OCR·음성은 iOS 네이티브(Vision/AVSpeech),
구조물(벽/문/창)은 **ARKit Scene Geometry + LiDAR 메쉬 분류**(SegFormer 등
별도 세그 모델 없음 — NPU/전용 하드웨어), 거리는 `sceneDepth` 실측(폴백:
bbox 휴리스틱), LLM은 LiteRT-LM 위 Gemma 4 E2B(온디바이스). Pro 기기에서는
카메라가 `ARSession`으로 통합되고, 비-LiDAR는 기존 `AVCapture` 경로로
자동 폴백. 경고는 계속 룰베이스만(정면·좁은 시야각, 문/창 ≤1.4m, 벽 ≤1.1m,
구조물 쿨다운 10초). Mac 데모와 동일한 SceneState 원칙(검증된 탐지
사실만 LLM에 투입)을 Swift로 이식했습니다.

## ARKit 메쉬 검증 (LiDAR Pro)
- [ ] 상단 배지 `ARKit mesh + YOLO` 표시
- [ ] 우상단 **Mesh ON** — 분류 컬러 오버레이 (floor=녹, wall=주황, door=파랑, window=청록)
- [ ] 상태줄에 `mesh: floor center 3.2m · door …` 형태가 수 초 내 등장
- [ ] 열린 복도에서 `"Path clear ahead, about N meters"` (12초 쿨다운)
- [ ] 문/창에 다가가면 `"door ahead, N meters"` / `"window ahead, N meters"`
- [ ] 복도 측면 벽에는 거의 울리지 않고, 막다른 벽(~1.8m)에만 wall 경고
