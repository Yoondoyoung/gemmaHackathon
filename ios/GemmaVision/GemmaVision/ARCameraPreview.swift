// ARKit 카메라 프리뷰 + YOLO 박스 오버레이.
// AVCaptureVideoPreviewLayer 대신 ARSCNView — 같은 ARSession을 공유한다.
import ARKit
import SwiftUI
import UIKit

struct ARCameraPreview: UIViewRepresentable {
    let session: ARSession
    var boxes: [DetBox] = []
    /// 센서 가로 버퍼 기준 (ARFrame.capturedImage와 동일 계열).
    var imageSize: CGSize = CGSize(width: 1920, height: 1440)

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView(frame: .zero)
        view.isUserInteractionEnabled = false
        view.session = session
        view.automaticallyUpdatesLighting = false
        view.rendersCameraGrain = false
        // 메쉬 와이어는 끔 — 심사 시 YOLO 박스가 더 읽기 쉬움.
        // 켜려면: view.debugOptions = [.showSceneUnderstanding]
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.imageSize = imageSize
        uiView.render(boxes)
    }

    final class PreviewView: ARSCNView {
        var imageSize: CGSize = CGSize(width: 1920, height: 1440)
        private let overlay = CALayer()
        private var lastBoxes: [DetBox] = []

        override init(frame: CGRect, options: [String: Any]? = nil) {
            super.init(frame: frame, options: options)
            overlay.zPosition = 1
            layer.addSublayer(overlay)
        }
        required init?(coder: NSCoder) { fatalError() }

        override func layoutSubviews() {
            super.layoutSubviews()
            overlay.frame = bounds
            render(lastBoxes)
        }

        func render(_ boxes: [DetBox]) {
            lastBoxes = boxes
            overlay.sublayers?.forEach { $0.removeFromSuperlayer() }
            let viewSize = bounds.size
            guard viewSize.width > 1, viewSize.height > 1 else { return }
            for b in boxes {
                // Vision(.right, 원점 좌하단) → 메타데이터(센서 네이티브, 원점 좌상단)
                let v = b.visionBox
                let metadataRect = CGRect(x: 1 - v.maxY, y: 1 - v.maxX,
                                          width: v.height, height: v.width)
                let rect = Self.aspectFillRect(metadataRect,
                                               imageSize: imageSize,
                                               viewSize: viewSize)
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

        /// AVCaptureVideoPreviewLayer `.resizeAspectFill` + metadataOutputRect 와 동일.
        private static func aspectFillRect(_ norm: CGRect, imageSize: CGSize,
                                           viewSize: CGSize) -> CGRect {
            let scale = max(viewSize.width / imageSize.width,
                            viewSize.height / imageSize.height)
            let scaledW = imageSize.width * scale
            let scaledH = imageSize.height * scale
            let ox = (viewSize.width - scaledW) / 2
            let oy = (viewSize.height - scaledH) / 2
            return CGRect(x: ox + norm.origin.x * scaledW,
                          y: oy + norm.origin.y * scaledH,
                          width: norm.width * scaledW,
                          height: norm.height * scaledH)
        }
    }
}
