// 탐지 + 경고 + 표지판 파이프라인 — src/config.py, src/alerts.py, src/main.py의
// 검증된 임계값·룰을 그대로 이식. OCR은 Florence 대신 iOS Vision(네이티브, ~100ms).
import Combine
import CoreImage
import CoreML
import CoreVideo
import Foundation
import UIKit
import Vision

enum Config {
    // src/config.py TRACK_LABELS와 동기 — 음식·스포츠 잡음 클래스는 제외
    static let trackLabels: Set<String> = [
        "person", "dog", "cat",
        "bicycle", "car", "motorcycle", "bus", "truck", "train",
        "traffic light", "stop sign", "fire hydrant", "parking meter",
        "chair", "couch", "bench", "bed", "dining table", "toilet",
        "potted plant", "tv", "refrigerator", "oven", "microwave", "sink",
        "backpack", "handbag", "suitcase", "umbrella", "cell phone",
        "laptop", "keyboard", "mouse", "remote", "book", "bottle", "cup",
        "skateboard", "clock", "vase"]
    static let nearThresh: [String: CGFloat] = [
        "person": 0.60, "chair": 0.50, "bicycle": 0.55, "dining table": 0.80,
        "couch": 0.80, "bench": 0.70, "bed": 0.85, "car": 0.70, "bus": 0.85,
        "truck": 0.85, "train": 0.85, "tv": 0.55, "refrigerator": 0.80,
        "oven": 0.55, "sink": 0.50, "toilet": 0.50, "laptop": 0.35,
        "bottle": 0.30, "cup": 0.25, "cell phone": 0.20, "book": 0.30,
        "clock": 0.30, "vase": 0.40, "umbrella": 0.55, "skateboard": 0.35,
        "fire hydrant": 0.45, "parking meter": 0.45]
    static let defaultNear: CGFloat = 0.55
    static let nearBottomMaxY: CGFloat = 0.25    // Vision 좌표(원점 좌하단): minY < 0.25 = 바닥 접점
    static let nearMeters: Double = 2.5          // LiDAR 실측 기준 (Mac판 depth 튜닝값)
    static let mediumMeters: Double = 5.0
    /// ARKit 구조물(벽/문/창) — 벽은 더 가까울 때만 (복도 측면 스팸 방지)
    static let structureNearMeters: Double = 2.5
    static let wallAlertMeters: Double = 1.8
    static let confMin: Float = 0.4
    static let alertCooldown: TimeInterval = 5    // 라벨별 (ID churn 대응, Mac판 실측 반영)
    static let alertGlobalGap: TimeInterval = 2
    static let ocrPeriod: TimeInterval = 1.2      // 시도 빈도↑ — 전광판 깜빡임/글레어 사이 깨끗한 프레임 확보
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
    private var signRecent: [String: (seen: Date, pos: String, content: String)] = [:]
    private var lastAnnounceAt = Date.distantPast
    private var processing = false
    private let queue = DispatchQueue(label: "pipeline.queue")
    // Gemma 스냅샷용 미니 SceneState (Mac판 스키마 축소 이식)
    private var lastObjects: [(label: String, pos: String, dist: String)] = []
    // ARKit Scene Geometry (벽/문/창) — YOLO 보완, 룰베이스 경고만
    private let structureLock = NSLock()
    private var lastStructures: [StructureHit] = []
    private var structureStatus = ""
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

    func process(_ pixelBuffer: CVPixelBuffer, depth: CVPixelBuffer?) {
        guard !processing, let model else { return }
        processing = true
        queue.async { [self] in
            defer { processing = false }
            refreshJPEGIfNeeded(pixelBuffer)
            // YOLO는 원본 프레임에서 (탐지엔 전처리 불필요)
            let detect = VNCoreMLRequest(model: model)
            detect.imageCropAndScaleOption = .scaleFill
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                orientation: inputOrientation)
            try? handler.perform([detect])
            // YOLO26 end2end: MultiArray [1,300,6] — VNRecognizedObjectObservation 아님.
            let dets = Self.decodeYOLO26(detect.results)
            handleDetections(dets, depth: depth)

            // OCR은 별도 핸들러 — 빛번짐 완화 전처리된 이미지에서 (전광판/LED 대응)
            if Date().timeIntervalSince(lastOCRAt) > Config.ocrPeriod {
                lastOCRAt = Date()
                let ocr = Self.makeOCRRequest()
                let ciImage = Self.deglared(pixelBuffer)
                let ocrHandler = VNImageRequestHandler(ciImage: ciImage,
                                                       orientation: inputOrientation)
                try? ocrHandler.perform([ocr])
                handleTexts(ocr.results ?? [])
            }
        }
    }

    /// ARKit 메쉬 분류 결과 — 정면 벽/문/창만 룰베이스 경고 (LLM 금지).
    func processStructures(_ hits: [StructureHit]) {
        let hint = hits.prefix(3).map {
            String(format: "%@ %@ %.1fm", $0.label, $0.pos, $0.meters)
        }.joined(separator: " · ")
        structureLock.lock()
        lastStructures = hits
        structureStatus = hint
        structureLock.unlock()

        guard !SpeechOut.shared.suppressingWarnings else { return }
        // 정면 + 근접만. 벽은 더 빡센 거리(복도 스팸 방지), 문/창은 nearMeters.
        let candidates = hits.filter { hit in
            guard hit.pos == "center" else { return false }
            if hit.label == "wall" { return hit.meters <= Config.wallAlertMeters }
            return hit.meters <= Config.structureNearMeters
        }.sorted { $0.meters < $1.meters }

        for hit in candidates {
            let now = Date()
            if now.timeIntervalSince(lastAlertByLabel[hit.label] ?? .distantPast)
                < Config.alertCooldown { continue }
            if now.timeIntervalSince(lastGlobalAlert) < Config.alertGlobalGap { continue }
            lastAlertByLabel[hit.label] = now
            lastGlobalAlert = now
            let rounded = max(1, Int(hit.meters.rounded()))
            speak("\(hit.label) ahead, \(rounded) meter\(rounded > 1 ? "s" : "")",
                  priority: 0)
            logObjectPassed(hit.label, pos: hit.pos)
            break   // 한 틱에 최대 1건
        }
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
            guard conf > Config.confMin else { continue }
            let cls = Int(array[[0, i, 5] as [NSNumber]].floatValue)
            guard cls >= 0, cls < Config.cocoNames.count else { continue }
            let label = Config.cocoNames[cls]
            guard Config.trackLabels.contains(label) else { continue }
            let x1 = CGFloat(array[[0, i, 0] as [NSNumber]].floatValue) / imgsz
            let y1 = CGFloat(array[[0, i, 1] as [NSNumber]].floatValue) / imgsz
            let x2 = CGFloat(array[[0, i, 2] as [NSNumber]].floatValue) / imgsz
            let y2 = CGFloat(array[[0, i, 3] as [NSNumber]].floatValue) / imgsz
            // YOLO 좌상단 → Vision 좌하단
            let box = CGRect(x: x1, y: 1 - y2, width: max(0, x2 - x1), height: max(0, y2 - y1))
            out.append(Det(label: label, confidence: conf, box: box))
        }
        return out
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
            if dist != "far" { logObjectPassed(det.label, pos: pos) }
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
        lastObjects = objects
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
        guard now.timeIntervalSince(lastJPEGAt) >= 0.4 else { return }
        lastJPEGAt = now
        // Gemma4 vision: 크면 vision_280 + ~2400 patches → CPU 인코더만 10초+.
        // 448 전후면 vision_140 이하로 떨어져 체감이 훨씬 낫다.
        guard let data = Self.jpegData(from: pixelBuffer, maxSide: 448,
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
        let objs = lastObjects.map {
            "{\"label\": \"\($0.label)\", \"pos\": \"\($0.pos)\", \"dist\": \"\($0.dist)\"}"
        }
        let now = Date()
        let texts = signRecent.values
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
        var json = "{\"objects\": [\(objs.joined(separator: ", "))], " +
                   "\"texts\": [\(texts.joined(separator: ", "))], " +
                   "\"structures\": [\(structs.joined(separator: ", "))]"
        if includeHistory {   // 회상 질문에만 — 지나온 것들의 기록
            json += ", \"recent_history\": \(recentHistoryJSON())"
        }
        return json + "}"
    }

    // MARK: - 표지판 (Vision OCR — 알림 가치 필터는 Mac판 worth_announcing 이식)

    private func handleTexts(_ observations: [VNRecognizedTextObservation]) {
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
            if matchGoal(words: words, content: content, pos: pos) { continue }
            let isNav = !words.isDisjoint(with: Config.navWords)
            let isBig = obs.boundingBox.height >= Config.signMinHeight
            guard isNav || isBig else { continue }              // 의미있는 것만
            guard content.split(separator: " ").count <= 3, content.count <= 20,
                  content.contains(where: { $0.isLetter })
                    || content.allSatisfy({ $0.isNumber }) else { continue }
            let key = content.lowercased()
            let now = Date()
            let seenBefore = signRecent[key]?.seen
            signRecent[key] = (now, pos, content)
            if let seen = seenBefore,
               now.timeIntervalSince(seen) < Config.signRearmGap { continue }
            logEpisode("\(content) sign", pos: pos)   // 지나온 표지판 기억 (알림 여부와 무관)
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

    /// 회상 질문이면 true → 프롬프트에 recent_history 포함.
    static func isRecallQuestion(_ q: String) -> Bool {
        let lower = q.lowercased()
        return Config.recallPhrases.contains { lower.contains($0) }
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
