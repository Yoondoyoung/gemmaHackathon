// 카메라 프리뷰 + 디버그 오버레이 (심사위원용 — 실사용자는 음성만 듣는다)
import AVFoundation
import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var pipeline = Pipeline()

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
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.black.opacity(0.5))
            .foregroundColor(.green)
        }
        .onAppear {
            camera.onFrame = { pipeline.process($0) }
            camera.start()
            SpeechOut.shared.say("Vision assist started", priority: 1)
        }
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
