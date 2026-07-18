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
    private var signLastSeen: [String: Date] = [:]
    private var lastAnnounceAt = Date.distantPast
    private var processing = false
    private let queue = DispatchQueue(label: "pipeline.queue")

    init() {
        // Xcode가 yolo11n.mlpackage에서 생성한 클래스 사용 (프로젝트에 모델 추가 필수)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all   // Neural Engine 우선
        if let ml = try? yolo11n(configuration: cfg).model {
            model = try? VNCoreMLModel(for: ml)
        }
    }

    func process(_ pixelBuffer: CVPixelBuffer) {
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
            handleDetections(detect.results as? [VNRecognizedObjectObservation] ?? [])
            if let ocr { handleTexts(ocr.results ?? []) }
        }
    }

    // MARK: - 장애물 경고 (룰베이스 — LLM 금지 원칙 유지)

    private func handleDetections(_ observations: [VNRecognizedObjectObservation]) {
        var summary: [String] = []
        for obs in observations {
            guard let top = obs.labels.first, top.confidence > Config.confMin,
                  Config.trackLabels.contains(top.identifier) else { continue }
            let box = obs.boundingBox           // 정규화, 원점 좌하단
            let pos = position(ofMidX: box.midX)
            let near = box.height >= (Config.nearThresh[top.identifier] ?? Config.defaultNear)
                && box.minY < Config.nearBottomMaxY   // 크기 + 바닥 접점 (Mac판 규칙)
            summary.append("\(top.identifier) \(pos) h=\(String(format: "%.2f", box.height))")
            guard near, pos == "center" else { continue }
            let now = Date()
            if now.timeIntervalSince(lastAlertByLabel[top.identifier] ?? .distantPast)
                < Config.alertCooldown { continue }
            if now.timeIntervalSince(lastGlobalAlert) < Config.alertGlobalGap { continue }
            lastAlertByLabel[top.identifier] = now
            lastGlobalAlert = now
            speak("\(top.identifier) ahead, close", priority: 0)
        }
        let line = summary.isEmpty ? "clear" : summary.joined(separator: " | ")
        DispatchQueue.main.async { self.statusLine = line }
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
            let seenBefore = signLastSeen[key]
            signLastSeen[key] = now
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
