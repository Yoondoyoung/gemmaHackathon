// 카메라 프리뷰 + 디버그 오버레이 (심사위원용 — 실사용자는 음성만 듣는다)
// Pro(LiDAR): ARKit Scene Geometry 경로 — 벽/문/창 메쉬 + sceneDepth.
// 그 외: 기존 AVCapture (+ DepthDataOutput 있으면 깊이만).
import AVFoundation
import SwiftUI

struct ContentView: View {
    private let useAR = ARCameraManager.isSupported
    @StateObject private var arCamera = ARCameraManager()
    @StateObject private var camera = CameraManager()
    @StateObject private var pipeline = Pipeline()
    @StateObject private var gemma = GemmaChat()
    @StateObject private var speechIn = SpeechIn()
    @State private var isPressingTalk = false

    var body: some View {
        ZStack(alignment: .top) {
            Group {
                if useAR {
                    ARCameraPreview(session: arCamera.session, boxes: pipeline.boxes)
                } else {
                    CameraPreview(session: camera.session, boxes: pipeline.boxes)
                }
            }
            .ignoresSafeArea()
            .allowsHitTesting(false)   // 프리뷰가 PTT 터치를 가로채지 않게

            VStack(alignment: .leading, spacing: 6) {
                Text(useAR ? "ARKit mesh + YOLO" : "YOLO (no scene mesh)")
                    .font(.caption2.monospaced())
                    .foregroundColor(.white.opacity(0.7))
                Text(pipeline.statusLine)
                    .font(.caption.monospaced())
                Text(pipeline.lastSpoken)
                    .font(.headline)
                    .foregroundColor(.yellow)
                Text(gemmaStatus)
                    .font(.caption)
                    .foregroundColor(.cyan)
                Text(speechStatus)
                    .font(.caption)
                    .foregroundColor(.orange)
                if !pipeline.activeGoal.isEmpty {
                    Text("🎯 goal: \(pipeline.activeGoal)")
                        .font(.caption)
                        .foregroundColor(.pink)
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.black.opacity(0.5))
            .foregroundColor(.green)
            .allowsHitTesting(false)

            VStack {
                Spacer()
                    .allowsHitTesting(false)
                pushToTalkButton
                    .padding(.bottom, 36)
            }
        }
        .onAppear {
            if useAR {
                arCamera.onFrame = { pipeline.process($0, depth: $1) }
                arCamera.onStructures = { pipeline.processStructures($0) }
                arCamera.start()
            } else {
                camera.onFrame = { pipeline.process($0, depth: $1) }
                camera.start()
            }
            gemma.load()
            speechIn.prepare()
            let boot = useAR
                ? "Vision assist started with scene mesh. Hold the button to ask."
                : "Vision assist started. Hold the button to ask."
            SpeechOut.shared.say(boot, priority: 1)
        }
    }

    private var gemmaStatus: String {
        switch gemma.state {
        case .idle: return "gemma: idle"
        case .loading: return "gemma: loading model…"
        case .ready: return "gemma: ready — " + gemma.lastAnswer
        case .busy: return "gemma: thinking…"
        case .failed(let why): return "gemma: FAILED — \(why)"
        }
    }

    private var speechStatus: String {
        switch speechIn.state {
        case .idle:
            return isPressingTalk ? "listening: …" : "mic: ready"
        case .requestingAuth:
            return "mic: requesting permission…"
        case .listening:
            return "listening: \(speechIn.partial.isEmpty ? "…" : speechIn.partial)"
        case .unavailable(let why):
            return "mic: \(why)"
        }
    }

    private var pushToTalkEnabled: Bool {
        gemma.state == .ready && speechIn.canListen && !gemmaBusy
    }

    private var gemmaBusy: Bool {
        if case .busy = gemma.state { return true }
        return false
    }

    private var pushToTalkButton: some View {
        let active = isPressingTalk || speechIn.state == .listening
        return Text(active ? "Release to ask" : "Hold to talk")
            .font(.headline.bold())
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 22)
            .padding(.horizontal, 24)
            .background(active ? Color.red : (pushToTalkEnabled ? Color.blue : Color.gray))
            .clipShape(Capsule())
            .contentShape(Capsule())          // 배경 전체 터치
            .padding(.horizontal, 28)
            .opacity(pushToTalkEnabled || active ? 1 : 0.5)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !isPressingTalk else { return }
                        guard pushToTalkEnabled, speechIn.state == .idle else { return }
                        isPressingTalk = true
                        speechIn.begin { question in
                            // 1) 회상 질문("아까 지나쳤어?")이면 에피소드 기억으로 답변
                            //    — 목표 판별보다 먼저 (목적지 단어가 들어있어도 회상 우선)
                            if Pipeline.isRecallQuestion(question) {
                                gemma.ask(question,
                                          scene: pipeline.snapshotJSON(includeHistory: true),
                                          imageJPEG: nil)   // 회상은 과거 → 현재 프레임 불필요
                            // 2) 목적지 발화면 목표 설정 (표지판 교차검증)
                            } else if let goal = Pipeline.extractGoal(from: question) {
                                pipeline.setGoal(spoken: goal.spoken,
                                                 keywords: goal.keywords)
                                GemmaChat.postPromptLogToMac(
                                    question: "GOAL: \(goal.spoken)",
                                    image: "none",
                                    scene: pipeline.snapshotJSON())
                                SpeechOut.shared.say(
                                    "Looking for the \(goal.spoken). "
                                    + "I'll tell you when I see a sign.", priority: 1)
                            // 3) 그 외 현재 장면 Q&A
                            } else {
                                let scene = pipeline.snapshotJSON()
                                let jpeg = pipeline.frameJPEG()
                                let kb = jpeg.map { ($0.count + 512) / 1024 } ?? 0
                                GemmaChat.postPromptLogToMac(
                                    question: question,
                                    image: jpeg == nil ? "none" : "yes \(kb)KB JPEG",
                                    scene: scene)
                                gemma.ask(question, scene: scene, imageJPEG: jpeg)
                            }
                        }
                    }
                    .onEnded { _ in
                        isPressingTalk = false
                        speechIn.end()
                    }
            )
            .accessibilityLabel("Push to talk")
            .accessibilityHint("Hold to speak your question, then release to send")
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var boxes: [DetBox] = []

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.isUserInteractionEnabled = false
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.render(boxes)
    }

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }

        private let overlay = CALayer()
        private var lastBoxes: [DetBox] = []

        override init(frame: CGRect) {
            super.init(frame: frame)
            overlay.zPosition = 1
            layer.addSublayer(overlay)
        }
        required init?(coder: NSCoder) { fatalError() }

        override func layoutSubviews() {
            super.layoutSubviews()
            overlay.frame = bounds
            render(lastBoxes)   // 회전/리사이즈 시 재배치
        }

        func render(_ boxes: [DetBox]) {
            lastBoxes = boxes
            overlay.sublayers?.forEach { $0.removeFromSuperlayer() }
            for b in boxes {
                // Vision(.right, 원점 좌하단) → 메타데이터(센서 네이티브, 원점 좌상단).
                // medianDepth의 검증된 매핑과 동일 축변환.
                let v = b.visionBox
                let metadataRect = CGRect(x: 1 - v.maxY, y: 1 - v.maxX,
                                          width: v.height, height: v.width)
                let rect = videoPreviewLayer
                    .layerRectConverted(fromMetadataOutputRect: metadataRect)
                guard rect.width > 1, rect.height > 1 else { continue }
                let color: CGColor = b.alert
                    ? UIColor.systemRed.cgColor : UIColor.systemGreen.cgColor

                let box = CAShapeLayer()
                box.path = UIBezierPath(roundedRect: rect, cornerRadius: 4).cgPath
                box.strokeColor = color
                box.lineWidth = 2.5
                box.fillColor = UIColor.clear.cgColor
                overlay.addSublayer(box)

                let label = CATextLayer()
                label.string = " \(b.text) "
                label.fontSize = 13
                label.foregroundColor = UIColor.black.cgColor
                label.backgroundColor = color
                label.alignmentMode = .left
                label.contentsScale = UIScreen.main.scale
                let lh: CGFloat = 17
                label.frame = CGRect(x: rect.minX,
                                     y: max(rect.minY - lh, 0),
                                     width: max(rect.width, 60), height: lh)
                overlay.addSublayer(label)
            }
        }
    }
}
