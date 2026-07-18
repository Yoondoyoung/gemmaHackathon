// 탐지 + 경고 + 표지판 파이프라인 — src/config.py, src/alerts.py, src/main.py의
// 검증된 임계값·룰을 그대로 이식. OCR은 Florence 대신 iOS Vision(네이티브, ~100ms).
import Combine
import CoreImage
import CoreML
import CoreVideo
import Foundation
import simd
import UIKit
import Vision

enum Config {
    // src/config.py TRACK_LABELS와 동기 — 음식·스포츠 잡음 클래스는 제외
    static let trackLabels: Set<String> = [
        "person", "dog", "cat",
        "bicycle", "car", "motorcycle", "bus", "truck", "train",
        "traffic light", "stop sign", "fire hydrant", "parking meter",
        "chair", "couch", "bench", "bed", "dining table", "toilet",
        "potted plant", "refrigerator", "oven", "microwave", "sink",
        "backpack", "handbag", "suitcase", "umbrella", "cell phone",
        "laptop", "keyboard", "mouse", "remote", "book", "bottle", "cup",
        "skateboard", "clock", "vase", "backpack"]
    static let nearThresh: [String: CGFloat] = [
        "person": 0.60, "chair": 0.50, "bicycle": 0.55, "dining table": 0.80,
        "couch": 0.80, "bench": 0.70, "bed": 0.85, "car": 0.70, "bus": 0.85,
        "truck": 0.85, "train": 0.85, "refrigerator": 0.80,
        "oven": 0.55, "sink": 0.50, "toilet": 0.50, "laptop": 0.35,
        "bottle": 0.30, "cup": 0.25, "cell phone": 0.20, "book": 0.30,
        "clock": 0.30, "vase": 0.40, "umbrella": 0.55, "skateboard": 0.35,
        "fire hydrant": 0.45, "parking meter": 0.45, "backpack": 0.45]
    static let defaultNear: CGFloat = 0.55
    static let nearBottomMaxY: CGFloat = 0.25    // Vision 좌표(원점 좌하단): minY < 0.25 = 바닥 접점
    static let nearMeters: Double = 2.5          // LiDAR 실측 기준 (Mac판 depth 튜닝값)
    static let mediumMeters: Double = 5.0
    /// ARKit 구조물 경고 — 실내 옆벽 스팸 방지: 벽은 거의 코앞+정면만
    static let structureNearMeters: Double = 1.4   // 문/창 경고 거리
    static let wallAlertMeters: Double = 0.9       // 막다른 벽만 (옆벽 제외는 스캔 단계에서)
    static let structureAlertCooldown: TimeInterval = 12  // 구조물 라벨 재경고 간격
    /// 바닥이 이 거리 이상 이어지고 앞에 막힘이 없으면 "path clear" 안내
    static let pathClearMinFloorM: Double = 1.5
    static let pathClearCooldown: TimeInterval = 12
    /// 갈림길 선제 발화 여부 — 바닥 메쉬 기반 감지는 방 입구/가구 틈/넓은 복도를
    /// 오탐해 정확도가 낮음. 오보는 시각장애 사용자를 벽으로 안내하므로 Push는 끔.
    /// (Gemma 스냅샷의 fork 힌트는 유지 — 질문 시 이미지와 교차검증되는 Pull 경로)
    static let announceForks = false
    /// 갈림길/측면 통로 안내 (같은 형태 재알림 쿨다운)
    static let forkCooldown: TimeInterval = 18
    /// 저장된 갈림길 웨이포인트에 이 거리 안으로 들어오면 "지나침" 안내
    static let forkPassRadiusM: Float = 1.4
    static let forkPassCooldown: TimeInterval = 25
    static let forkWaypointMax = 12
    /// YOLO 물체를 ARKit 월드 좌표에 찍어 "다시 가기" 안내용 (GPS 아님)
    static let objectMemoryMaxAgeSec: TimeInterval = 180
    static let objectMemoryMax = 30
    static let objectMemoryMinDepthM: Double = 0.5
    static let objectMemoryMaxDepthM: Double = 6.0
    static let findBackPhrases: [String] = [
        "where is my", "where's my", "where is the", "where's the",
        "find my", "find the", "take me back", "go back to", "guide me to",
        "how do i get back", "lead me to"]
    static let confMin: Float = 0.45              // nano 기본 — 낮으면 후반 오탐↑
    /// 클래스별 conf (미지정은 confMin). backpack는 낮추고, 의자/병 등 오탐 클래스는 올림.
    static let confByLabel: [String: Float] = [
        "backpack": 0.28, "handbag": 0.30, "suitcase": 0.32,
        "cell phone": 0.35, "laptop": 0.35, "bottle": 0.55, "cup": 0.55,
        "book": 0.55, "remote": 0.55, "vase": 0.55, "clock": 0.50,
        "chair": 0.50, "potted plant": 0.50, "mouse": 0.55, "keyboard": 0.50]
    static let yoloMaxDets = 12                   // conf 상위만 — 패딩/잡음 슬롯 차단
    static let yoloMinBoxArea: CGFloat = 0.004    // 화면 대비 너무 작은 박스 무시
    static let alertCooldown: TimeInterval = 5    // YOLO 라벨별 (ID churn 대응)
    static let alertGlobalGap: TimeInterval = 2
    static let ocrPeriod: TimeInterval = 2.0      // 16 Pro 열/메모리 여유 (이전이 1.2)
    /// 30fps는 발열·드랍으로 후반 품질 저하. 데모 3분이면 15fps면 충분.
    static let yoloMinInterval: TimeInterval = 1.0 / 15.0
    static let ocrMinConfidence: Float = 0.4      // 빛번짐으로 저하된 읽기도 살림 (nav/목표 필터가 잡음 차단)
    static let navWords: Set<String> = [
        "exit", "restroom", "toilet", "toilets", "wc", "men", "women",
        "ladies", "gents", "gate", "elevator", "lift", "stairs", "escalator",
        "entrance", "emergency", "information", "info", "cafeteria", "cafe",
        "parking", "reception", "push", "pull", "caution", "danger", "wet", "floor"]
    static let signMinHeight: CGFloat = 0.1      // 프레임 대비 텍스트 높이
    static let signRearmGap: TimeInterval = 10    // 사라졌다 재등장 시 재알림

    // 목표(목적지) 기억: 발화에 목적지가 있으면 유사어 집합을 목표로 잡고,
    // 이후 표지판이 매칭되면 "Found it" 알림 (Mac판 extract_goal 이식, 온디바이스 테이블).
    static let destinationSynonyms: [String: [String]] = [
        "restroom": ["restroom", "toilet", "toilets", "bathroom", "washroom",
                     "wc", "men", "women", "ladies", "gents", "lavatory"],
        "exit": ["exit", "exits", "way out"],
        "elevator": ["elevator", "lift", "elevators"],
        "stairs": ["stairs", "stairway", "staircase"],
        "escalator": ["escalator", "escalators"],
        "cafeteria": ["cafeteria", "cafe", "canteen", "dining", "food"],
        "entrance": ["entrance", "entry"],
        "information desk": ["information", "info", "reception", "help"],
        "parking": ["parking", "garage"],
        "gate": ["gate", "gates"],
    ]
    // 목적지 노출이 없어도 이 문구 뒤 단어를 단일 키워드 목표로 (테이블 밖 목적지)
    static let goalIntentPhrases: [String] = [
        "go to", "get to", "take me to", "looking for", "find the", "find a",
        "find me", "where is", "where's", "need the", "need a", "want to go to"]

    // 에피소드 기억: 지나온 표지판·물체를 몇 분간 누적 → "아까 지나쳤어?" 류 질문에만 사용
    static let episodeMaxAgeSec: TimeInterval = 300     // 5분 창
    static let episodeMaxCount = 40                     // 로그 상한
    static let episodeInPromptMax = 15                  // 프롬프트에 넣는 최대 개수
    static let episodeObjectCooldown: TimeInterval = 8  // 같은 라벨 물체 재기록 간격
    // 회상 질문 판별 — 이 문구가 있으면 현재 장면 대신(추가로) 에피소드 로그를 Gemma에 전달
    static let recallPhrases: [String] = [
        "did we", "did i", "did you see", "have we", "earlier", "a moment ago",
        "back there", "we passed", "passed a", "passed the", "was there",
        "behind us", "just passed", "go back", "on the way"]
    static let announceGap: TimeInterval = 4
    /// YOLO26 end2end 입력 해상도 (CoreML imgsz).
    static let yoloInput: CGFloat = 640
    /// COCO-80 — yolo26n.mlpackage metadata `names`와 동일.
    static let cocoNames: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"]
}

private struct Det {
    let label: String
    let confidence: Float
    let box: CGRect   // Vision 정규화, 원점 좌하단
}

/// 화면 오버레이용 탐지 박스 (디버그 — 심사위원이 "뭘 보는지" 확인).
struct DetBox: Identifiable {
    let id: Int
    let visionBox: CGRect   // Vision 정규화, 원점 좌하단
    let text: String
    let alert: Bool         // center + near → 빨강, 그 외 초록
}

final class Pipeline: ObservableObject {
    @Published var statusLine = "camera starting…"
    @Published var lastSpoken = ""
    @Published var boxes: [DetBox] = []   // 화면 박스 오버레이
    @Published var activeGoal = ""        // 현재 목표(목적지) 표시용, 없으면 ""
    // 입력 방향: 라이브 카메라(가로 센서, 세로 파지)=.right / 영상 파일(이미 정립)=.up
    var inputOrientation: CGImagePropertyOrientation = .right
    private let goalLock = NSLock()
    private var goalKeywords: Set<String> = []   // handleTexts(백그라운드)에서 매칭
    // 에피소드 기억 (지나온 것들) — 백그라운드에서 기록, 질문 시 읽음
    private let episodeLock = NSLock()
    private var episodes: [(t: Date, what: String, pos: String)] = []
    private var lastObjectEpisode: [String: Date] = [:]

    private var model: VNCoreMLModel?
    private var lastAlertByLabel: [String: Date] = [:]
    private var lastGlobalAlert = Date.distantPast
    private var lastOCRAt = Date.distantPast
    private var lastYoloAt = Date.distantPast
    private var signRecent: [String: (seen: Date, pos: String, content: String)] = [:]
    private var lastAnnounceAt = Date.distantPast
    private var processing = false
    private let queue = DispatchQueue(label: "pipeline.queue", qos: .userInitiated)
    // Gemma 스냅샷용 미니 SceneState (Mac판 스키마 축소 이식)
    // sceneLock: 쓰기는 pipeline.queue, 읽기는 PTT 순간 메인 스레드(snapshotJSON) —
    // Dictionary/Array 동시 접근은 크래시 벡터라 반드시 락.
    private let sceneLock = NSLock()
    private var lastObjects: [(label: String, pos: String, dist: String)] = []
    // ARKit Scene Geometry (벽/문/창/갈림길) — YOLO 보완, 룰베이스 경고만
    private let structureLock = NSLock()
    private var lastStructures: [StructureHit] = []
    private var structureStatus = ""
    private var lastForkPos: String?   // rising-edge 감지용 (both/left/right)
    // ARKit 월드 좌표 갈림길 기억 (GPS 아님 — 실내 온디바이스 궤적)
    private struct ForkWaypoint {
        let position: SIMD3<Float>
        let side: String          // left | right | both
        var announcedPass: Bool
    }
    private var forkWaypoints: [ForkWaypoint] = []
    // 물체·표지판 공간 기억 — AR 월드 좌표 (GPS 아님)
    private struct ObjectMemory {
        let label: String       // "backpack" 또는 표지판 문구 "EXIT"
        let position: SIMD3<Float>
        let seenAt: Date
        let isSign: Bool
    }
    private let memoryLock = NSLock()
    private var objectMemories: [ObjectMemory] = []
    private var lastPose: (pos: SIMD3<Float>, forward: SIMD3<Float>)?
    private var lastObjectMemoryAt: [String: Date] = [:]
    // Q&A용 최신 프레임 (JPEG). 탐지 루프에서 스로틀 갱신.
    private let jpegLock = NSLock()
    private var latestJPEG: Data?
    private var lastJPEGAt = Date.distantPast
    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    init() {
        // Xcode가 yolo26n.mlpackage에서 생성한 클래스 사용 (프로젝트에 모델 추가 필수)
        let cfg = MLModelConfiguration()
        // MPSGraph `MLIR pass manager failed` assertion (catch 불가, 프로세스 사망):
        // - 시뮬레이터: GPU/ANE 경로 전부 위험 → cpuOnly
        // - 실기기: ANE 서브타입 미인식(0x8030) 기기에서 .all이 GPU 컴파일로
        //   폴백하다 같은 assertion으로 사망 → GPU만 제외한 .cpuAndNeuralEngine.
        #if targetEnvironment(simulator)
        cfg.computeUnits = .cpuOnly
        #else
        cfg.computeUnits = .cpuAndNeuralEngine
        #endif
        if let ml = try? yolo26n(configuration: cfg).model {
            model = try? VNCoreMLModel(for: ml)
        }
    }

    /// Gemma 멀티모달 입력용 — 세로 보정된 최신 프레임 JPEG (없으면 nil).
    func frameJPEG() -> Data? {
        jpegLock.lock()
        defer { jpegLock.unlock() }
        return latestJPEG
    }

    func process(_ pixelBuffer: CVPixelBuffer, depth: CVPixelBuffer?,
                 pose: (SIMD3<Float>, SIMD3<Float>)? = nil) {
        if let pose { lastPose = (pose.0, pose.1) }
        guard !processing, let model else { return }
        let now = Date()
        // YOLO 스로틀 — AR 프레임(60fps)마다 돌리면 중반부터 발열·드랍
        guard now.timeIntervalSince(lastYoloAt) >= Config.yoloMinInterval else { return }
        lastYoloAt = now
        processing = true
        // depth/pixel은 비동기 전에 retain (다음 프레임에 버퍼 재사용될 수 있음)
        queue.async { [self] in
            defer { processing = false }
            refreshJPEGIfNeeded(pixelBuffer)
            let detect = VNCoreMLRequest(model: model)
            detect.imageCropAndScaleOption = .scaleFill
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                orientation: inputOrientation)
            try? handler.perform([detect])
            let dets = Self.decodeYOLO26(detect.results)
            handleDetections(dets, depth: depth)

            if Date().timeIntervalSince(lastOCRAt) > Config.ocrPeriod {
                lastOCRAt = Date()
                let ocr = Self.makeOCRRequest()
                let ciImage = Self.deglared(pixelBuffer)
                let ocrHandler = VNImageRequestHandler(ciImage: ciImage,
                                                       orientation: inputOrientation)
                try? ocrHandler.perform([ocr])
                handleTexts(ocr.results ?? [], depth: depth)
            }
        }
    }

    /// ARKit 메쉬 — 벽/문/창 경고 + 갈림길 + 지나침 + 직진 안내 (LLM 금지).
    /// 경로는 GPS가 아니라 `update`의 ARKit 월드 좌표를 사용.
    /// AR 델리게이트 큐에서 오지만, lastAlertByLabel 등 공유 상태가 YOLO 경로
    /// (pipeline.queue)와 겹치므로 같은 큐로 직렬화 (Dictionary 동시 변경 방지).
    func processStructures(_ update: StructureUpdate) {
        queue.async { [self] in processStructuresSerialized(update) }
    }

    private func processStructuresSerialized(_ update: StructureUpdate) {
        let hits = update.hits
        let hint = hits.prefix(5).map {
            String(format: "%@ %@ %.1fm", $0.label, $0.pos, $0.meters)
        }.joined(separator: " · ")
        structureLock.lock()
        lastStructures = hits
        structureStatus = hint
        structureLock.unlock()

        lastPose = (update.cameraPosition, update.cameraForward)
        // 저장된 갈림길 근처를 지나가면 안내
        if announceForkPassIfNeeded(at: update.cameraPosition) { return }

        guard !SpeechOut.shared.suppressingWarnings else { return }

        // 1) 장애물 경고 — 정면 + 가까운 거리만
        let candidates = hits.filter { hit in
            guard hit.pos == "center" else { return false }
            switch hit.label {
            case "floor", "fork": return false
            case "wall": return hit.meters <= Config.wallAlertMeters
            case "door", "window": return hit.meters <= Config.structureNearMeters
            default: return false
            }
        }.sorted { $0.meters < $1.meters }

        for hit in candidates {
            let now = Date()
            if now.timeIntervalSince(lastAlertByLabel[hit.label] ?? .distantPast)
                < Config.structureAlertCooldown { continue }
            if now.timeIntervalSince(lastGlobalAlert) < Config.alertGlobalGap { continue }
            lastAlertByLabel[hit.label] = now
            lastGlobalAlert = now
            let rounded = max(1, Int(hit.meters.rounded()))
            speak("\(hit.label) ahead, \(rounded) meter\(rounded > 1 ? "s" : "")",
                  priority: 0)
            logObjectPassed(hit.label, pos: hit.pos)
            return
        }

        // 2) 갈림길 — 확정된 fork가 새로 나타날 때만 + 월드 좌표에 웨이포인트 저장
        if Config.announceForks, let fork = hits.first(where: { $0.label == "fork" }) {
            let prev = lastForkPos
            lastForkPos = fork.pos
            let appeared = prev != fork.pos
            if appeared {
                let now = Date()
                let key = "fork_\(fork.pos)"
                if now.timeIntervalSince(lastAlertByLabel[key] ?? .distantPast)
                    >= Config.forkCooldown,
                   now.timeIntervalSince(lastGlobalAlert) >= Config.alertGlobalGap {
                    lastAlertByLabel[key] = now
                    lastGlobalAlert = now
                    rememberForkWaypoint(fork: fork, camera: update.cameraPosition,
                                         forward: update.cameraForward)
                    let phrase: String
                    switch fork.pos {
                    case "both":
                        phrase = "Fork ahead — paths on your left and right"
                    case "left":
                        phrase = "Opening on your left"
                    case "right":
                        phrase = "Opening on your right"
                    default:
                        phrase = "Path splits ahead"
                    }
                    speak(phrase, priority: 1)
                    logEpisode("fork \(fork.pos)",
                               pos: fork.pos == "both" ? "center" : fork.pos)
                    return
                }
            }
        } else {
            lastForkPos = nil
        }

        // 3) 직진 가능
        guard let floor = hits.first(where: { $0.label == "floor" && $0.pos == "center" }),
              floor.meters >= Config.pathClearMinFloorM else { return }
        let blocked = hits.contains {
            $0.pos == "center"
                && ($0.label == "wall" || $0.label == "door" || $0.label == "window")
                && $0.meters < min(floor.meters, Config.structureNearMeters)
        }
        guard !blocked else { return }
        let now = Date()
        if now.timeIntervalSince(lastAlertByLabel["path_clear"] ?? .distantPast)
            < Config.pathClearCooldown { return }
        if now.timeIntervalSince(lastGlobalAlert) < Config.alertGlobalGap { return }
        lastAlertByLabel["path_clear"] = now
        lastGlobalAlert = now
        let clearM = max(1, Int(min(floor.meters, Config.mediumMeters).rounded()))
        speak("Path clear ahead, about \(clearM) meter\(clearM > 1 ? "s" : "")",
              priority: 1)
        logEpisode("clear path", pos: "center")
    }

    /// 갈림길을 카메라 앞쪽 월드 좌표에 찍어 둔다 (세션 내 온디바이스 지도).
    private func rememberForkWaypoint(fork: StructureHit, camera: SIMD3<Float>,
                                      forward: SIMD3<Float>) {
        let ahead = Float(min(fork.meters * 0.55, 2.5))
        var pos = camera + forward * ahead
        // 측면이면 약간 옆으로 치우쳐 저장
        let right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        if fork.pos == "left" { pos -= right * 0.8 }
        if fork.pos == "right" { pos += right * 0.8 }
        // 가까운 기존 포인트와 합치기
        for i in forkWaypoints.indices {
            if simd_distance(forkWaypoints[i].position, pos) < 1.2 {
                forkWaypoints[i] = ForkWaypoint(position: pos, side: fork.pos,
                                                announcedPass: false)
                return
            }
        }
        forkWaypoints.append(ForkWaypoint(position: pos, side: fork.pos,
                                          announcedPass: false))
        if forkWaypoints.count > Config.forkWaypointMax {
            forkWaypoints.removeFirst(forkWaypoints.count - Config.forkWaypointMax)
        }
    }

    /// 예전에 찍은 갈림길 근처를 지나가면 한 번 안내. true면 이 틱에서 발화함.
    @discardableResult
    private func announceForkPassIfNeeded(at camera: SIMD3<Float>) -> Bool {
        guard Config.announceForks else { return false }
        guard !SpeechOut.shared.suppressingWarnings else { return false }
        let now = Date()
        if now.timeIntervalSince(lastAlertByLabel["fork_pass"] ?? .distantPast)
            < Config.forkPassCooldown { return false }
        if now.timeIntervalSince(lastGlobalAlert) < Config.alertGlobalGap { return false }

        for i in forkWaypoints.indices {
            let wp = forkWaypoints[i]
            guard !wp.announcedPass else { continue }
            let d = simd_distance(camera, wp.position)
            guard d <= Config.forkPassRadiusM else { continue }
            forkWaypoints[i].announcedPass = true
            lastAlertByLabel["fork_pass"] = now
            lastGlobalAlert = now
            let phrase: String
            switch wp.side {
            case "both":
                phrase = "You are at a junction — paths left and right"
            case "left":
                phrase = "You are passing a turn on your left"
            case "right":
                phrase = "You are passing a turn on your right"
            default:
                phrase = "You are passing a junction"
            }
            speak(phrase, priority: 1)
            logEpisode("passed fork \(wp.side)", pos: "center")
            return true
        }
        return false
    }

    private static func makeOCRRequest() -> VNRecognizeTextRequest {
        let r = VNRecognizeTextRequest()
        r.recognitionLevel = .accurate
        r.recognitionLanguages = ["en-US"]
        r.usesLanguageCorrection = false   // 표지판은 산문이 아님 (EXIT/게이트번호) — 보정이 오히려 왜곡
        r.minimumTextHeight = 0.02         // 먼 표지판도 시도
        return r
    }

    /// 빛번짐(bloom) 완화: 하이라이트를 눌러 날아간 밝은 글자 복원 + 대비로 획 선명화.
    /// 완전 포화(순백)된 LED는 복원 불가 — 그건 노출 문제라 소프트웨어 한계.
    private static func deglared(_ pixelBuffer: CVPixelBuffer) -> CIImage {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        return ci
            .applyingFilter("CIHighlightShadowAdjust",
                            parameters: ["inputHighlightAmount": 0.25,
                                         "inputShadowAmount": 0.3])
            .applyingFilter("CIColorControls",
                            parameters: [kCIInputContrastKey: 1.2,
                                         kCIInputBrightnessKey: -0.08,
                                         kCIInputSaturationKey: 0.6])
    }

    /// YOLO26 CoreML end2end 출력 `[1, 300, 6]` = `[x1,y1,x2,y2,conf,cls]` (640 픽셀, 좌상단).
    private static func decodeYOLO26(_ results: [VNObservation]?) -> [Det] {
        guard let features = results as? [VNCoreMLFeatureValueObservation],
              let array = features.first?.featureValue.multiArrayValue,
              array.shape.count >= 3 else { return [] }
        let n = array.shape[1].intValue
        let imgsz = Config.yoloInput
        var out: [Det] = []
        out.reserveCapacity(min(n, 64))
        for i in 0..<n {
            let conf = array[[0, i, 4] as [NSNumber]].floatValue
            let cls = Int(array[[0, i, 5] as [NSNumber]].floatValue)
            guard cls >= 0, cls < Config.cocoNames.count else { continue }
            let label = Config.cocoNames[cls]
            guard Config.trackLabels.contains(label) else { continue }
            let need = Config.confByLabel[label] ?? Config.confMin
            guard conf >= need else { continue }
            let x1 = CGFloat(array[[0, i, 0] as [NSNumber]].floatValue) / imgsz
            let y1 = CGFloat(array[[0, i, 1] as [NSNumber]].floatValue) / imgsz
            let x2 = CGFloat(array[[0, i, 2] as [NSNumber]].floatValue) / imgsz
            let y2 = CGFloat(array[[0, i, 3] as [NSNumber]].floatValue) / imgsz
            // YOLO 좌상단 → Vision 좌하단
            let box = CGRect(x: x1, y: 1 - y2, width: max(0, x2 - x1), height: max(0, y2 - y1))
            guard box.width * box.height >= Config.yoloMinBoxArea else { continue }
            out.append(Det(label: label, confidence: conf, box: box))
        }
        // conf 상위만 — 300슬롯 패딩/저신뢰 잔여가 후반에 쌓여 보이는 오탐 완화
        return Array(out.sorted { $0.confidence > $1.confidence }.prefix(Config.yoloMaxDets))
    }

    // MARK: - 장애물 경고 (룰베이스 — LLM 금지 원칙 유지)

    private func handleDetections(_ detections: [Det], depth: CVPixelBuffer?) {
        var summary: [String] = []
        var objects: [(String, String, String)] = []
        var newBoxes: [DetBox] = []
        for (idx, det) in detections.enumerated() {
            let box = det.box
            let pos = position(ofMidX: box.midX)
            let meters = depth.flatMap { medianDepth($0, box: box) }
            let near: Bool
            let dist: String
            if let m = meters {                  // LiDAR 실측 (Mac판 depth 판정 이식)
                near = m <= Config.nearMeters
                dist = near ? "near" : (m <= Config.mediumMeters ? "medium" : "far")
            } else {                             // 휴리스틱 폴백: 크기 + 바닥 접점
                let thresh = Config.nearThresh[det.label] ?? Config.defaultNear
                near = box.height >= thresh && box.minY < Config.nearBottomMaxY
                dist = near ? "near" : (box.height >= thresh * 0.5 ? "medium" : "far")
            }
            objects.append((det.label, pos, dist))
            let tag = meters.map { String(format: "%.1fm", $0) }
                ?? String(format: "h=%.2f", box.height)
            summary.append("\(det.label) \(pos) \(tag)")
            newBoxes.append(DetBox(id: idx, visionBox: box,
                                   text: "\(det.label) \(tag)",
                                   alert: near && pos == "center"))
            // 에피소드 기억은 경고보다 느슨하게 — near뿐 아니라 medium(수 m 내)도,
            // 어느 쪽(좌/중/우)이든 '지나친 것'으로 기록 (경고는 여전히 near+center만).
            if dist != "far" {
                logObjectPassed(det.label, pos: pos)
                // 월드 좌표 기억 — LiDAR 우선, 없으면 박스 높이로 대략 추정
                // (깊이 없으면 "Where's my backpack?"가 항상 실패하던 구멍)
                let depthM = meters ?? min(Config.objectMemoryMaxDepthM,
                    max(Config.objectMemoryMinDepthM, 0.55 / max(0.05, Double(box.height))))
                rememberObject(label: det.label, screenPos: pos, depthM: depthM)
            }
            guard near, pos == "center" else { continue }
            // Q&A 답변 재생 중엔 경고를 내지 않는다 (쿨다운도 소진 안 함 —
            // 답변 끝난 뒤 여전히 가까우면 즉시 다시 경고).
            if SpeechOut.shared.suppressingWarnings { continue }
            let now = Date()
            if now.timeIntervalSince(lastAlertByLabel[det.label] ?? .distantPast)
                < Config.alertCooldown { continue }
            if now.timeIntervalSince(lastGlobalAlert) < Config.alertGlobalGap { continue }
            lastAlertByLabel[det.label] = now
            lastGlobalAlert = now
            if let m = meters {
                let rounded = max(1, Int(m.rounded()))
                speak("\(det.label) ahead, \(rounded) meter\(rounded > 1 ? "s" : "")",
                      priority: 0)
            } else {
                speak("\(det.label) ahead, close", priority: 0)
            }
        }
        sceneLock.lock()
        lastObjects = objects
        sceneLock.unlock()
        structureLock.lock()
        let mesh = structureStatus
        structureLock.unlock()
        var line = summary.isEmpty ? "clear" : summary.joined(separator: " | ")
        if !mesh.isEmpty { line += " || mesh: \(mesh)" }
        DispatchQueue.main.async {
            self.statusLine = line
            self.boxes = newBoxes
        }
    }

    /// LiDAR 깊이 맵에서 bbox 중앙 50% 영역의 중앙값(미터).
    /// 좌표 변환: Vision(세로 표시 공간, 원점 좌하단) → 센서 버퍼(가로, 원점 좌상단)
    /// orientation .right 기준: x_buf = (1 - y_vis) * W, y_buf = (1 - x_vis) * H
    private func medianDepth(_ depth: CVPixelBuffer, box: CGRect) -> Double? {
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let w = CVPixelBufferGetWidth(depth)
        let h = CVPixelBufferGetHeight(depth)
        let rowBytes = CVPixelBufferGetBytesPerRow(depth)
        let inner = box.insetBy(dx: box.width / 4, dy: box.height / 4)
        var samples: [Double] = []
        for i in 0..<8 {
            for j in 0..<8 {
                let u = inner.minX + inner.width * CGFloat(i) / 7
                let v = inner.minY + inner.height * CGFloat(j) / 7
                let xb = min(max(Int((1 - v) * CGFloat(w)), 0), w - 1)
                let yb = min(max(Int((1 - u) * CGFloat(h)), 0), h - 1)
                let value = base.advanced(by: yb * rowBytes + xb * 4)
                    .assumingMemoryBound(to: Float32.self).pointee
                if value.isFinite && value > 0 { samples.append(Double(value)) }
            }
        }
        guard samples.count >= 8 else { return nil }   // LiDAR 범위 밖(>5m 등)이면 폴백
        return samples.sorted()[samples.count / 2]
    }

    private func refreshJPEGIfNeeded(_ pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastJPEGAt) >= 1.0 else { return }
        lastJPEGAt = now
        // Gemma vision용 — 질문 직전에만 최신 프레임이면 충분. 자주 인코딩하면 발열↑
        guard let data = Self.jpegData(from: pixelBuffer, maxSide: 384,
                                       orientation: inputOrientation) else { return }
        jpegLock.lock()
        latestJPEG = data
        jpegLock.unlock()
    }

    /// 센서 버퍼 → 정립 방향으로 회전 후 JPEG.
    private static func jpegData(from pixelBuffer: CVPixelBuffer, maxSide: CGFloat,
                                 orientation: CGImagePropertyOrientation) -> Data? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation)
        let w = ci.extent.width
        let h = ci.extent.height
        guard w > 0, h > 0 else { return nil }
        let scale = min(1, maxSide / max(w, h))
        let scaled = scale < 1
            ? ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            : ci
        guard let cg = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.7)
    }

    /// Gemma 프롬프트용 장면 스냅샷 (Mac판 SceneState.snapshot_json 축소 이식)
    func snapshotJSON(includeHistory: Bool = false) -> String {
        sceneLock.lock()
        let objectsCopy = lastObjects
        let signsCopy = Array(signRecent.values)
        sceneLock.unlock()
        let objs = objectsCopy.map {
            "{\"label\": \"\($0.label)\", \"pos\": \"\($0.pos)\", \"dist\": \"\($0.dist)\"}"
        }
        let now = Date()
        let texts = signsCopy
            .filter { now.timeIntervalSince($0.seen) < 30 }
            .sorted { $0.seen > $1.seen }
            .prefix(6)
            .map {
                "{\"content\": \"\($0.content)\", \"pos\": \"\($0.pos)\", " +
                "\"age_sec\": \(Int(now.timeIntervalSince($0.seen)))}"
            }
        structureLock.lock()
        let structures = lastStructures
        structureLock.unlock()
        let structs = structures.prefix(4).map { h -> String in
            let dist: String
            if h.meters <= Config.nearMeters { dist = "near" }
            else if h.meters <= Config.mediumMeters { dist = "medium" }
            else { dist = "far" }
            return String(format:
                "{\"label\": \"%@\", \"pos\": \"%@\", \"dist\": \"%@\", \"depth_m\": %.1f}",
                h.label, h.pos, dist, h.meters)
        }
        // 정면 바닥이 이어지고 가까운 벽/문/창이 없으면 이동 가능
        let floorM = structures.first { $0.label == "floor" && $0.pos == "center" }?.meters
        let blocked = structures.contains {
            $0.pos == "center"
                && ($0.label == "wall" || $0.label == "door" || $0.label == "window")
                && $0.meters < Config.structureNearMeters
        }
        let pathClear = (floorM ?? 0) >= Config.pathClearMinFloorM && !blocked
        var json = "{\"objects\": [\(objs.joined(separator: ", "))], " +
                   "\"texts\": [\(texts.joined(separator: ", "))], " +
                   "\"structures\": [\(structs.joined(separator: ", "))], " +
                   "\"path_clear\": \(pathClear)" +
                   (floorM.map { String(format: ", \"path_clear_m\": %.1f", $0) } ?? "")
        if includeHistory {   // 회상 질문에만 — 지나온 것들의 기록
            json += ", \"recent_history\": \(recentHistoryJSON())"
        }
        return json + "}"
    }

    // MARK: - 표지판 (Vision OCR — 알림 가치 필터는 Mac판 worth_announcing 이식)

    private func handleTexts(_ observations: [VNRecognizedTextObservation],
                             depth: CVPixelBuffer?) {
        for obs in observations {
            guard let candidate = obs.topCandidates(1).first,
                  candidate.confidence > Config.ocrMinConfidence else { continue }
            let content = candidate.string.trimmingCharacters(in: .whitespaces)
            guard content.count >= 2 else { continue }
            let words = Set(content.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty })
            let pos = position(ofMidX: obs.boundingBox.midX)
            // 목표 매칭은 일반 알림 필터보다 먼저 (목표는 무조건 알림)
            if matchGoal(words: words, content: content, pos: pos) {
                rememberSign(content: content, screenPos: pos, box: obs.boundingBox,
                             depth: depth)
                logEpisode("\(content) sign", pos: pos)   // Found it도 회상 로그에 남김
                continue
            }
            let isNav = !words.isDisjoint(with: Config.navWords)
            let isBig = obs.boundingBox.height >= Config.signMinHeight
            guard isNav || isBig else { continue }              // 의미있는 것만
            guard content.split(separator: " ").count <= 3, content.count <= 20,
                  content.contains(where: { $0.isLetter })
                    || content.allSatisfy({ $0.isNumber }) else { continue }
            let key = content.lowercased()
            let now = Date()
            sceneLock.lock()
            let seenBefore = signRecent[key]?.seen
            signRecent[key] = (now, pos, content)
            sceneLock.unlock()
            // 공간 기억은 알림 여부와 무관하게 (되돌아가기용)
            rememberSign(content: content, screenPos: pos, box: obs.boundingBox,
                         depth: depth)
            if let seen = seenBefore,
               now.timeIntervalSince(seen) < Config.signRearmGap { continue }
            logEpisode("\(content) sign", pos: pos)
            if now.timeIntervalSince(lastAnnounceAt) < Config.announceGap { continue }
            lastAnnounceAt = now
            speak("Sign detected: \(content), \(spoken(position(ofMidX: obs.boundingBox.midX)))",
                  priority: 1)
        }
    }

    // MARK: - 에피소드 기억 (지나온 것들)

    private func logEpisode(_ what: String, pos: String) {
        let now = Date()
        episodeLock.lock()
        episodes.append((now, what, pos))
        episodes.removeAll { now.timeIntervalSince($0.t) > Config.episodeMaxAgeSec }
        if episodes.count > Config.episodeMaxCount {
            episodes.removeFirst(episodes.count - Config.episodeMaxCount)
        }
        episodeLock.unlock()
    }

    /// 가까이 지나친 물체를 라벨별 쿨다운으로 기록 ("passed a chair on your left").
    private func logObjectPassed(_ label: String, pos: String) {
        let now = Date()
        episodeLock.lock()
        let recent = lastObjectEpisode[label]
        let ok = recent == nil || now.timeIntervalSince(recent!) > Config.episodeObjectCooldown
        if ok { lastObjectEpisode[label] = now }
        episodeLock.unlock()
        if ok { logEpisode("a \(label)", pos: pos) }
    }

    /// YOLO 탐지 → ARKit 월드 좌표 메모리 (GPS 대신).
    private func rememberObject(label: String, screenPos: String, depthM: Double) {
        storeMemory(label: label, screenPos: screenPos, depthM: depthM, isSign: false)
    }

    /// 표지판 OCR → 월드 좌표 메모리 (EXIT/restroom 등 되돌아가기).
    private func rememberSign(content: String, screenPos: String, box: CGRect,
                              depth: CVPixelBuffer?) {
        let depthM: Double
        if let depth, let m = medianDepth(depth, box: box) {
            depthM = m
        } else {
            // 깊이 없으면 글자 크기로 대략 추정 (클수록 가까움)
            let h = max(0.02, Double(box.height))
            depthM = min(Config.objectMemoryMaxDepthM,
                         max(Config.objectMemoryMinDepthM, 0.45 / h))
        }
        storeMemory(label: content, screenPos: screenPos, depthM: depthM, isSign: true)
    }

    private func storeMemory(label: String, screenPos: String, depthM: Double,
                             isSign: Bool) {
        guard depthM >= Config.objectMemoryMinDepthM,
              depthM <= Config.objectMemoryMaxDepthM,
              let pose = lastPose else { return }
        let key = (isSign ? "sign:" : "obj:") + label.lowercased()
        let now = Date()
        if let last = lastObjectMemoryAt[key],
           now.timeIntervalSince(last) < 2.0 { return }
        lastObjectMemoryAt[key] = now

        let right = simd_normalize(simd_cross(pose.forward, SIMD3<Float>(0, 1, 0)))
        let lateral: Float
        switch screenPos {
        case "left": lateral = -0.55
        case "right": lateral = 0.55
        default: lateral = 0
        }
        let world = pose.pos + pose.forward * Float(depthM) + right * lateral

        memoryLock.lock()
        objectMemories.removeAll {
            now.timeIntervalSince($0.seenAt) > Config.objectMemoryMaxAgeSec
        }
        objectMemories.removeAll {
            $0.label.lowercased() == label.lowercased() && $0.isSign == isSign
        }
        objectMemories.append(ObjectMemory(label: label, position: world,
                                           seenAt: now, isSign: isSign))
        if objectMemories.count > Config.objectMemoryMax {
            objectMemories.removeFirst(objectMemories.count - Config.objectMemoryMax)
        }
        memoryLock.unlock()
    }

    /// 회상 질문이면 true → 프롬프트에 recent_history 포함.
    static func isRecallQuestion(_ q: String) -> Bool {
        let lower = q.lowercased()
        return Config.recallPhrases.contains { lower.contains($0) }
    }

    /// "where's my backpack / take me back to my bag" → 공간 안내 문장 (룰베이스).
    static func isFindBackQuestion(_ q: String) -> Bool {
        let lower = q.lowercased()
        return Config.findBackPhrases.contains { lower.contains($0) }
    }

    /// "Did I pass the restroom / my backpack?" → 에피소드·공간기억으로 예/아니오 (Gemma 없이).
    /// 구체적 대상이 질문에 없으면 nil → Gemma 회상 경로로 넘김.
    func answerRecall(for question: String) -> String? {
        let lower = question.lowercased()
        guard Self.questionHasRecallSubject(lower) else { return nil }

        let now = Date()
        // 1) 에피소드 로그 (표지판 알림·지나친 물체)
        episodeLock.lock()
        let epMatch = episodes
            .filter { now.timeIntervalSince($0.t) <= Config.episodeMaxAgeSec }
            .filter { Self.subjectMatches(question: lower, subject: $0.what) }
            .max(by: { $0.t < $1.t })
        episodeLock.unlock()
        if let ep = epMatch {
            let age = Int(now.timeIntervalSince(ep.t).rounded())
            let noun = Self.spokenNoun(from: ep.what)
            return "Yes — we passed \(noun) \(spoken(ep.pos)) \(Self.agePhrase(age))."
        }

        // 2) AR 공간 기억 (알림 없이도 remember된 표지판/물체)
        memoryLock.lock()
        objectMemories.removeAll {
            now.timeIntervalSince($0.seenAt) > Config.objectMemoryMaxAgeSec
        }
        let memMatch = objectMemories
            .filter { memoryMatches(question: lower, mem: $0) }
            .max(by: { $0.seenAt < $1.seenAt })
        memoryLock.unlock()
        if let mem = memMatch {
            let age = Int(now.timeIntervalSince(mem.seenAt).rounded())
            let noun = mem.isSign ? "a \(mem.label) sign" : "a \(mem.label)"
            return "Yes — I saw \(noun) \(Self.agePhrase(age))."
        }

        // 대상은 알아들었는데 기록이 없음
        if let named = Self.namedRecallSubject(in: lower) {
            return "No, I don't remember passing \(named) recently."
        }
        return "I don't remember that yet."
    }

    /// 질문에서 찾을 라벨을 뽑아, 기억된 월드 좌표 기준으로 돌아갈 방향을 말함.
    func guideBack(for question: String) -> String? {
        // nil 반환 = ContentView가 다음 경로(회상→목표→Gemma)로 폴스루.
        // 문자열을 반환하면 라우팅이 여기서 끝나므로, '안내할 수 있을 때만' 문자열.
        guard let pose = lastPose else {
            return nil   // 포즈 없음(비 LiDAR 기기 포함) → 목표 설정/Gemma로
        }
        let lower = question.lowercased()
        memoryLock.lock()
        let now = Date()
        objectMemories.removeAll {
            now.timeIntervalSince($0.seenAt) > Config.objectMemoryMaxAgeSec
        }
        let memories = objectMemories
        memoryLock.unlock()

        // 질문에 포함된 물체/표지판 중 가장 최근 기억
        let match = memories
            .filter { memoryMatches(question: lower, mem: $0) }
            .max(by: { $0.seenAt < $1.seenAt })
        guard let mem = match else {
            return nil   // 저장된 기억 없음 — "Where is the restroom" 등은 목표 설정으로 폴스루
        }

        let age = Int(now.timeIntervalSince(mem.seenAt).rounded())
        let delta = mem.position - pose.pos
        var flat = SIMD3<Float>(delta.x, 0, delta.z)
        let dist = simd_length(flat)
        let noun = mem.isSign ? "\(mem.label) sign" : mem.label
        guard dist > 0.15 else {
            return mem.isSign
                ? "The \(noun) should be right here, within about a step."
                : "Your \(noun) should be right here, within about a step."
        }
        flat /= dist

        let forward = pose.forward
        let right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        let ahead = simd_dot(flat, forward)
        let side = simd_dot(flat, right)
        let meters = max(1, Int(dist.rounded()))
        let agePhrase = Self.agePhrase(age)

        let turn: String
        if ahead < -0.35 {
            turn = side < -0.25 ? "behind you on your left"
                : side > 0.25 ? "behind you on your right"
                : "behind you"
        } else if abs(side) > abs(ahead) {
            turn = side < 0 ? "on your left" : "on your right"
        } else {
            turn = side < -0.25 ? "ahead on your left"
                : side > 0.25 ? "ahead on your right"
                : "ahead of you"
        }
        if mem.isSign {
            return "The \(noun) is about \(meters) meter\(meters > 1 ? "s" : "") \(turn). I saw it \(agePhrase)."
        }
        return "Your \(noun) is about \(meters) meter\(meters > 1 ? "s" : "") \(turn). I saw it \(agePhrase)."
    }

    private func memoryMatches(question q: String, mem: ObjectMemory) -> Bool {
        let label = mem.label.lowercased()
        if Self.subjectMatches(question: q, subject: label) { return true }
        if mem.isSign {
            // "the sign"만 물으면 nav 표지판이면 매칭
            if q.contains("sign"),
               label.components(separatedBy: CharacterSet.alphanumerics.inverted)
                .contains(where: { Config.navWords.contains($0) }) {
                return true
            }
        }
        return synonymMatch(q, label: label)
    }

    /// "RESTROOM sign" / "WC" ↔ 질문 "restroom" / "bathroom" 등 동의어 포함 매칭.
    private static func subjectMatches(question q: String, subject: String) -> Bool {
        let sub = subject.lowercased()
        let subTokens = Set(sub.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 })
        let qTokens = Set(q.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 })
        if !subTokens.isDisjoint(with: qTokens) { return true }
        if q.contains(sub), sub.count >= 3 { return true }

        // destinationSynonyms: restroom ↔ toilet/wc/bathroom …
        for (_, syns) in Config.destinationSynonyms {
            let set = Set(syns)
            let qHit = !qTokens.isDisjoint(with: set) || syns.contains(where: { q.contains($0) })
            let sHit = !subTokens.isDisjoint(with: set) || syns.contains(where: { sub.contains($0) })
            if qHit && sHit { return true }
        }
        // YOLO 물체 별칭
        let objectAliases: [String: [String]] = [
            "backpack": ["bag", "backpack", "rucksack"],
            "handbag": ["purse", "handbag", "bag"],
            "cell phone": ["phone", "iphone", "cellphone"],
            "suitcase": ["suitcase", "luggage"],
        ]
        for (canon, syns) in objectAliases {
            let set = Set(syns + [canon])
            if !qTokens.isDisjoint(with: set), !subTokens.isDisjoint(with: set) { return true }
            if syns.contains(where: { q.contains($0) }),
               sub.contains(canon) || syns.contains(where: { sub.contains($0) }) {
                return true
            }
        }
        return false
    }

    /// 회상 질문에 구체적 대상(목적지/물체)이 있는지.
    private static func questionHasRecallSubject(_ q: String) -> Bool {
        namedRecallSubject(in: q) != nil
            || Config.cocoNames.contains(where: { q.contains($0) })
            || ["bag", "phone", "purse", "luggage", "sign"].contains(where: { q.contains($0) })
    }

    private static func namedRecallSubject(in q: String) -> String? {
        for (name, syns) in Config.destinationSynonyms {
            if syns.contains(where: { q.contains($0) }) { return "the \(name)" }
        }
        for name in Config.cocoNames where q.contains(name) {
            return name == "backpack" || name == "handbag" || name == "suitcase"
                ? "your \(name)" : "a \(name)"
        }
        if q.contains("bag") { return "your bag" }
        if q.contains("phone") { return "your phone" }
        if q.contains("sign") { return "that sign" }
        return nil
    }

    private static func spokenNoun(from what: String) -> String {
        let w = what.trimmingCharacters(in: .whitespacesAndNewlines)
        if w.hasPrefix("a ") || w.hasPrefix("the ") { return w }
        if w.lowercased().contains("sign") { return "the \(w)" }
        return w
    }

    private static func agePhrase(_ age: Int) -> String {
        if age < 15 { return "just now" }
        if age < 45 { return "a moment ago" }
        if age < 90 { return "about a minute ago" }
        return "a couple minutes ago"
    }

    private func synonymMatch(_ q: String, label: String) -> Bool {
        Self.subjectMatches(question: q, subject: label)
    }

    private func recentHistoryJSON() -> String {
        let now = Date()
        episodeLock.lock()
        let evs = episodes
            .filter { now.timeIntervalSince($0.t) <= Config.episodeMaxAgeSec }
            .suffix(Config.episodeInPromptMax)
            .reversed()   // 최근 것부터
        episodeLock.unlock()
        let items = evs.map {
            "{\"what\": \"\($0.what)\", \"pos\": \"\($0.pos)\", " +
            "\"age_sec\": \(Int(now.timeIntervalSince($0.t)))}"
        }
        return "[\(items.joined(separator: ", "))]"
    }

    // MARK: - 목표(목적지) 기억

    /// 발화에서 목적지를 뽑아 (표시명, 유사어 집합) 반환. 목적지가 없으면 nil → 일반 Q&A.
    static func extractGoal(from utterance: String) -> (spoken: String, keywords: Set<String>)? {
        let lower = utterance.lowercased()
        let tokens = Set(lower.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty })
        // 1) 유사어 테이블 매칭 (가장 신뢰도 높음)
        for (name, syns) in Config.destinationSynonyms where !tokens.isDisjoint(with: Set(syns)) {
            return (name, Set(syns))
        }
        // 2) 의도 문구 뒤 명사를 단일 키워드로 (테이블 밖 목적지)
        for phrase in Config.goalIntentPhrases {
            guard let r = lower.range(of: phrase) else { continue }
            let after = lower[r.upperBound...]
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty && !["the", "a", "an"].contains($0) }
            if let noun = after.first, noun.count >= 3 {
                return (noun, [noun])
            }
        }
        return nil
    }

    func setGoal(spoken: String, keywords: Set<String>) {
        goalLock.lock(); goalKeywords = keywords; goalLock.unlock()
        DispatchQueue.main.async { self.activeGoal = spoken }
    }

    func clearGoal() {
        goalLock.lock(); goalKeywords = []; goalLock.unlock()
        DispatchQueue.main.async { self.activeGoal = "" }
    }

    /// 표지판 토큰이 목표 유사어와 겹치면 "Found it" 발화 후 목표 해제. 매칭 시 true.
    private func matchGoal(words: Set<String>, content: String, pos: String) -> Bool {
        goalLock.lock()
        let hit = !goalKeywords.isEmpty && !words.isDisjoint(with: goalKeywords)
        goalLock.unlock()
        guard hit else { return false }
        clearGoal()
        speak("Found it — a sign for \(content), \(spoken(pos))", priority: 1)
        return true
    }

    // MARK: - helpers

    private func position(ofMidX x: CGFloat) -> String {
        x < 1.0 / 3 ? "left" : (x < 2.0 / 3 ? "center" : "right")
    }

    private func spoken(_ pos: String) -> String {
        ["left": "on your left", "center": "ahead of you",
         "right": "on your right"][pos] ?? pos
    }

    private func speak(_ text: String, priority: Int) {
        SpeechOut.shared.say(text, priority: priority)
        DispatchQueue.main.async { self.lastSpoken = text }
    }
}
