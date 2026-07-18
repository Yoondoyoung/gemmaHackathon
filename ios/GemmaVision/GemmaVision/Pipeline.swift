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
    static let trackLabels: Set<String> = [
        "person", "chair", "bicycle", "car", "dog", "backpack", "suitcase",
        "bench", "potted plant", "couch", "dining table", "bus", "truck",
        "motorcycle", "traffic light", "stop sign",
        "umbrella", "skateboard", "fire hydrant", "parking meter"]
    static let nearThresh: [String: CGFloat] = [
        "person": 0.60, "chair": 0.50, "bicycle": 0.55, "dining table": 0.80,
        "couch": 0.80, "bench": 0.70, "car": 0.70, "bus": 0.85, "truck": 0.85,
        "umbrella": 0.55, "skateboard": 0.35,
        "fire hydrant": 0.45, "parking meter": 0.45]
    static let defaultNear: CGFloat = 0.55
    static let nearBottomMaxY: CGFloat = 0.25    // Vision 좌표(원점 좌하단): minY < 0.25 = 바닥 접점
    static let nearMeters: Double = 2.5          // LiDAR 실측 기준 (Mac판 depth 튜닝값)
    static let mediumMeters: Double = 5.0
    static let confMin: Float = 0.4
    static let alertCooldown: TimeInterval = 5    // 라벨별 (ID churn 대응, Mac판 실측 반영)
    static let alertGlobalGap: TimeInterval = 2
    static let ocrPeriod: TimeInterval = 2.5
    static let navWords: Set<String> = [
        "exit", "restroom", "toilet", "toilets", "wc", "men", "women",
        "ladies", "gents", "gate", "elevator", "lift", "stairs", "escalator",
        "entrance", "emergency", "information", "info", "cafeteria", "cafe",
        "parking", "reception", "push", "pull", "caution", "danger", "wet", "floor"]
    static let signMinHeight: CGFloat = 0.05      // 프레임 대비 텍스트 높이
    static let signRearmGap: TimeInterval = 10    // 사라졌다 재등장 시 재알림
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
    // 입력 방향: 라이브 카메라(가로 센서, 세로 파지)=.right / 영상 파일(이미 정립)=.up
    var inputOrientation: CGImagePropertyOrientation = .right

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
            let detect = VNCoreMLRequest(model: model)
            detect.imageCropAndScaleOption = .scaleFill
            var requests: [VNRequest] = [detect]
            var ocr: VNRecognizeTextRequest?
            if Date().timeIntervalSince(lastOCRAt) > Config.ocrPeriod {
                lastOCRAt = Date()
                let r = VNRecognizeTextRequest()
                r.recognitionLevel = .accurate
                ocr = r
                requests.append(r)
            }
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                orientation: inputOrientation)
            try? handler.perform(requests)
            // YOLO26 end2end: MultiArray [1,300,6] — VNRecognizedObjectObservation 아님.
            let dets = Self.decodeYOLO26(detect.results)
            handleDetections(dets, depth: depth)
            if let ocr { handleTexts(ocr.results ?? []) }
        }
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
            guard near, pos == "center" else { continue }
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
        let line = summary.isEmpty ? "clear" : summary.joined(separator: " | ")
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
    func snapshotJSON() -> String {
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
        return "{\"objects\": [\(objs.joined(separator: ", "))], " +
               "\"texts\": [\(texts.joined(separator: ", "))]}"
    }

    // MARK: - 표지판 (Vision OCR — 알림 가치 필터는 Mac판 worth_announcing 이식)

    private func handleTexts(_ observations: [VNRecognizedTextObservation]) {
        for obs in observations {
            guard let candidate = obs.topCandidates(1).first,
                  candidate.confidence > 0.5 else { continue }
            let content = candidate.string.trimmingCharacters(in: .whitespaces)
            guard content.count >= 2 else { continue }
            let words = Set(content.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty })
            let isNav = !words.isDisjoint(with: Config.navWords)
            let isBig = obs.boundingBox.height >= Config.signMinHeight
            guard isNav || isBig else { continue }              // 의미있는 것만
            guard content.split(separator: " ").count <= 3, content.count <= 20,
                  content.contains(where: { $0.isLetter })
                    || content.allSatisfy({ $0.isNumber }) else { continue }
            let key = content.lowercased()
            let now = Date()
            let pos = position(ofMidX: obs.boundingBox.midX)
            let seenBefore = signRecent[key]?.seen
            signRecent[key] = (now, pos, content)
            if let seen = seenBefore,
               now.timeIntervalSince(seen) < Config.signRearmGap { continue }
            if now.timeIntervalSince(lastAnnounceAt) < Config.announceGap { continue }
            lastAnnounceAt = now
            speak("Sign detected: \(content), \(spoken(position(ofMidX: obs.boundingBox.midX)))",
                  priority: 1)
        }
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
