// 탐지 + 경고 + 표지판 파이프라인 — src/config.py, src/alerts.py, src/main.py의
// 검증된 임계값·룰을 그대로 이식. OCR은 Florence 대신 iOS Vision(네이티브, ~100ms).
import CoreML
import CoreVideo
import Foundation
import Vision

enum Config {
    static let trackLabels: Set<String> = [
        "person", "chair", "bicycle", "car", "dog", "backpack", "suitcase",
        "bench", "potted plant", "couch", "dining table", "bus", "truck",
        "motorcycle", "traffic light", "stop sign"]
    static let nearThresh: [String: CGFloat] = [
        "person": 0.60, "chair": 0.50, "bicycle": 0.55, "dining table": 0.80,
        "couch": 0.80, "bench": 0.70, "car": 0.70, "bus": 0.85, "truck": 0.85]
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
}

final class Pipeline: ObservableObject {
    @Published var statusLine = "camera starting…"
    @Published var lastSpoken = ""

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

    init() {
        // Xcode가 yolo11n.mlpackage에서 생성한 클래스 사용 (프로젝트에 모델 추가 필수)
        let cfg = MLModelConfiguration()
        // .all 금지: GPU(MPSGraph) 컴파일이 일부 기기/시뮬레이터에서
        // `MLIR pass manager failed` assertion으로 프로세스째 죽는다 (실기기 확인).
        #if targetEnvironment(simulator)
        cfg.computeUnits = .cpuOnly
        #else
        cfg.computeUnits = .cpuAndNeuralEngine   // 그래도 죽으면 .cpuOnly
        #endif
        if let ml = try? yolo11n(configuration: cfg).model {
            model = try? VNCoreMLModel(for: ml)
        }
    }

    func process(_ pixelBuffer: CVPixelBuffer, depth: CVPixelBuffer?) {
        guard !processing, let model else { return }
        processing = true
        queue.async { [self] in
            defer { processing = false }
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
                                                orientation: .right)  // 세로 파지
            try? handler.perform(requests)
            handleDetections(detect.results as? [VNRecognizedObjectObservation] ?? [],
                             depth: depth)
            if let ocr { handleTexts(ocr.results ?? []) }
        }
    }

    // MARK: - 장애물 경고 (룰베이스 — LLM 금지 원칙 유지)

    private func handleDetections(_ observations: [VNRecognizedObjectObservation],
                                  depth: CVPixelBuffer?) {
        var summary: [String] = []
        var objects: [(String, String, String)] = []
        for obs in observations {
            guard let top = obs.labels.first, top.confidence > Config.confMin,
                  Config.trackLabels.contains(top.identifier) else { continue }
            let box = obs.boundingBox           // 정규화, 원점 좌하단
            let pos = position(ofMidX: box.midX)
            let meters = depth.flatMap { medianDepth($0, box: box) }
            let near: Bool
            let dist: String
            if let m = meters {                  // LiDAR 실측 (Mac판 depth 판정 이식)
                near = m <= Config.nearMeters
                dist = near ? "near" : (m <= Config.mediumMeters ? "medium" : "far")
            } else {                             // 휴리스틱 폴백: 크기 + 바닥 접점
                let thresh = Config.nearThresh[top.identifier] ?? Config.defaultNear
                near = box.height >= thresh && box.minY < Config.nearBottomMaxY
                dist = near ? "near" : (box.height >= thresh * 0.5 ? "medium" : "far")
            }
            objects.append((top.identifier, pos, dist))
            let tag = meters.map { String(format: "%.1fm", $0) }
                ?? String(format: "h=%.2f", box.height)
            summary.append("\(top.identifier) \(pos) \(tag)")
            guard near, pos == "center" else { continue }
            let now = Date()
            if now.timeIntervalSince(lastAlertByLabel[top.identifier] ?? .distantPast)
                < Config.alertCooldown { continue }
            if now.timeIntervalSince(lastGlobalAlert) < Config.alertGlobalGap { continue }
            lastAlertByLabel[top.identifier] = now
            lastGlobalAlert = now
            if let m = meters {
                let rounded = max(1, Int(m.rounded()))
                speak("\(top.identifier) ahead, \(rounded) meter\(rounded > 1 ? "s" : "")",
                      priority: 0)
            } else {
                speak("\(top.identifier) ahead, close", priority: 0)
            }
        }
        lastObjects = objects
        let line = summary.isEmpty ? "clear" : summary.joined(separator: " | ")
        DispatchQueue.main.async { self.statusLine = line }
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
