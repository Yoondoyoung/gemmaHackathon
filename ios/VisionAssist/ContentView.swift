// 카메라 프리뷰 + 디버그 오버레이 (심사위원용 — 실사용자는 음성만 듣는다)
import AVFoundation
import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var pipeline = Pipeline()
    @StateObject private var gemma = GemmaChat()

    var body: some View {
        ZStack(alignment: .top) {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()
            VStack(alignment: .leading, spacing: 6) {
                Text(pipeline.statusLine)
                    .font(.caption.monospaced())
                Text(pipeline.lastSpoken)
                    .font(.headline)
                    .foregroundColor(.yellow)
                Text(gemmaStatus)
                    .font(.caption)
                    .foregroundColor(.cyan)
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.black.opacity(0.5))
            .foregroundColor(.green)

            VStack {
                Spacer()
                HStack(spacing: 12) {
                    askButton("What's ahead of me?")
                    askButton("Do you see any signs?")
                }
                .padding(.bottom, 24)
            }
        }
        .onAppear {
            camera.onFrame = { pipeline.process($0) }
            camera.start()
            gemma.load()
            SpeechOut.shared.say("Vision assist started", priority: 1)
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

    private func askButton(_ question: String) -> some View {
        Button(question) {
            gemma.ask(question, scene: pipeline.snapshotJSON())
        }
        .font(.callout.bold())
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(gemma.state == .ready ? Color.blue : Color.gray)
        .foregroundColor(.white)
        .clipShape(Capsule())
        .disabled(gemma.state != .ready)
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}
