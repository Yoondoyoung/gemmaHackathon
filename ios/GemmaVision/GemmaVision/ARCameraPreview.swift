// ARKit 카메라 프리뷰 + YOLO 박스 오버레이.
// 메쉬 디버그: 켤 때만 희소·스로틀 렌더 (이전 전면 컬러 복제는 중반 버벅임 주원인).
import ARKit
import SceneKit
import SwiftUI
import UIKit

struct ARCameraPreview: UIViewRepresentable {
    let session: ARSession
    var boxes: [DetBox] = []
    var showMesh: Bool = false
    var imageSize: CGSize = CGSize(width: 1920, height: 1440)

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView(frame: .zero)
        view.isUserInteractionEnabled = false
        view.session = session
        view.delegate = context.coordinator
        view.automaticallyUpdatesLighting = false
        view.rendersCameraGrain = false
        view.preferredFramesPerSecond = 30
        context.coordinator.showMesh = showMesh
        view.setMeshVisible(showMesh)
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.imageSize = imageSize
        context.coordinator.showMesh = showMesh
        uiView.setMeshVisible(showMesh)
        uiView.render(boxes)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, ARSCNViewDelegate {
        var showMesh = false
        private var lastRebuildAt: [UUID: TimeInterval] = [:]
        private static let rebuildMinGap: TimeInterval = 1.5
        private static let faceStride = 12   // 12면 중 1만 그림

        func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            guard showMesh, let mesh = anchor as? ARMeshAnchor else { return nil }
            let node = SCNNode()
            node.name = "mesh.\(mesh.identifier.uuidString)"
            node.renderingOrder = -1
            node.geometry = Self.sparseGeometry(from: mesh)
            lastRebuildAt[mesh.identifier] = Date().timeIntervalSince1970
            return node
        }

        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode,
                      for anchor: ARAnchor) {
            guard let mesh = anchor as? ARMeshAnchor else { return }
            if !showMesh {
                node.isHidden = true
                node.geometry = nil
                return
            }
            node.isHidden = false
            let now = Date().timeIntervalSince1970
            let last = lastRebuildAt[mesh.identifier] ?? 0
            guard now - last >= Self.rebuildMinGap else { return }
            lastRebuildAt[mesh.identifier] = now
            node.geometry = Self.sparseGeometry(from: mesh)
        }

        /// 면의 샘플 + 단색(분류별) — 전체 복제 대비 메모리 ~1/12.
        private static func sparseGeometry(from meshAnchor: ARMeshAnchor) -> SCNGeometry {
            let geo = meshAnchor.geometry
            let vertices = geo.vertices
            let faces = geo.faces
            let faceCount = faces.count
            let per = faces.indexCountPerPrimitive
            let stride = max(1, faceStride)

            var positions: [SCNVector3] = []
            var colors: [SIMD4<Float>] = []
            let est = max(1, faceCount / stride) * per
            positions.reserveCapacity(est)
            colors.reserveCapacity(est)

            var f = 0
            while f < faceCount {
                defer { f += stride }
                let c = color(for: classification(of: f, geometry: geo))
                for k in 0..<per {
                    let idx = vertexIndex(face: f, corner: k, faces: faces)
                    let v = vertex(at: idx, source: vertices)
                    positions.append(SCNVector3(v.x, v.y, v.z))
                    colors.append(c)
                }
            }
            let triCount = positions.count / 3
            guard triCount > 0 else {
                return SCNGeometry()
            }

            let posSrc = SCNGeometrySource(vertices: positions)
            let colorData = colors.withUnsafeBufferPointer { Data(buffer: $0) }
            let colorSrc = SCNGeometrySource(
                data: colorData,
                semantic: .color,
                vectorCount: colors.count,
                usesFloatComponents: true,
                componentsPerVector: 4,
                bytesPerComponent: MemoryLayout<Float>.size,
                dataOffset: 0,
                dataStride: MemoryLayout<SIMD4<Float>>.stride)
            var indices = [Int32](0..<Int32(positions.count))
            let idxData = indices.withUnsafeBufferPointer { Data(buffer: $0) }
            let element = SCNGeometryElement(
                data: idxData,
                primitiveType: .triangles,
                primitiveCount: triCount,
                bytesPerIndex: MemoryLayout<Int32>.size)
            let geometry = SCNGeometry(sources: [posSrc, colorSrc], elements: [element])
            let mat = SCNMaterial()
            mat.lightingModel = .constant
            mat.isDoubleSided = true
            mat.writesToDepthBuffer = false
            geometry.materials = [mat]
            return geometry
        }

        private static func vertex(at index: Int, source: ARGeometrySource) -> SIMD3<Float> {
            let ptr = source.buffer.contents()
                .advanced(by: source.offset + source.stride * index)
            return ptr.assumingMemoryBound(to: SIMD3<Float>.self).pointee
        }

        private static func vertexIndex(face: Int, corner: Int,
                                        faces: ARGeometryElement) -> Int {
            let offset = (face * faces.indexCountPerPrimitive + corner) * faces.bytesPerIndex
            let ptr = faces.buffer.contents().advanced(by: offset)
            if faces.bytesPerIndex == 2 {
                return Int(ptr.assumingMemoryBound(to: UInt16.self).pointee)
            }
            return Int(ptr.assumingMemoryBound(to: UInt32.self).pointee)
        }

        private static func classification(of faceIndex: Int,
                                           geometry: ARMeshGeometry) -> ARMeshClassification {
            guard let classifications = geometry.classification else { return .none }
            let ptr = classifications.buffer.contents()
                .advanced(by: classifications.offset + classifications.stride * faceIndex)
            let raw = Int(ptr.assumingMemoryBound(to: UInt8.self).pointee)
            return ARMeshClassification(rawValue: raw) ?? .none
        }

        private static func color(for cls: ARMeshClassification) -> SIMD4<Float> {
            switch cls {
            case .floor: return SIMD4(0.15, 0.90, 0.35, 0.40)
            case .wall: return SIMD4(1.00, 0.55, 0.10, 0.35)
            case .door: return SIMD4(0.20, 0.45, 1.00, 0.45)
            case .window: return SIMD4(0.20, 0.90, 0.95, 0.42)
            default: return SIMD4(0.55, 0.55, 0.55, 0.15)
            }
        }
    }

    final class PreviewView: ARSCNView {
        var imageSize: CGSize = CGSize(width: 1920, height: 1440)
        private let overlay = CALayer()
        private var lastBoxes: [DetBox] = []
        private var lastBoxSig = ""
        private var meshOn = false

        override init(frame: CGRect, options: [String: Any]? = nil) {
            super.init(frame: frame, options: options)
            overlay.zPosition = 1
            layer.addSublayer(overlay)
        }
        required init?(coder: NSCoder) { fatalError() }

        func setMeshVisible(_ on: Bool) {
            meshOn = on
            scene.rootNode.enumerateChildNodes { node, _ in
                if node.name?.hasPrefix("mesh.") == true {
                    node.isHidden = !on
                    if !on { node.geometry = nil }
                }
            }
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            overlay.frame = bounds
            lastBoxSig = ""
            render(lastBoxes)
        }

        func render(_ boxes: [DetBox]) {
            lastBoxes = boxes
            // geometry 포함 — 라벨만 같으면 박스가 멈추던 버그 방지
            let sig = boxes.map {
                let v = $0.visionBox
                return String(format: "%d:%d:%.3f,%.3f,%.3f,%.3f:%@",
                              $0.id, $0.alert ? 1 : 0,
                              v.minX, v.minY, v.width, v.height, $0.text)
            }.joined(separator: "|")
            guard sig != lastBoxSig else { return }
            lastBoxSig = sig

            overlay.sublayers?.forEach { $0.removeFromSuperlayer() }
            let viewSize = bounds.size
            guard viewSize.width > 1, viewSize.height > 1 else { return }

            let band = CAShapeLayer()
            let path = UIBezierPath()
            let x1 = viewSize.width / 3, x2 = viewSize.width * 2 / 3
            path.move(to: CGPoint(x: x1, y: 0)); path.addLine(to: CGPoint(x: x1, y: viewSize.height))
            path.move(to: CGPoint(x: x2, y: 0)); path.addLine(to: CGPoint(x: x2, y: viewSize.height))
            band.path = path.cgPath
            band.strokeColor = UIColor.white.withAlphaComponent(0.35).cgColor
            band.lineWidth = 1
            band.lineDashPattern = [6, 4]
            overlay.addSublayer(band)

            // ARSCNView 카메라와 동일 매핑: Vision(정립 BL) → metadata → displayTransform
            let iface = window?.windowScene?.interfaceOrientation ?? .portrait
            let displayT = session.currentFrame?
                .displayTransform(for: iface, viewportSize: viewSize)

            for b in boxes {
                let rect: CGRect
                if let displayT {
                    rect = Self.viewRect(visionBox: b.visionBox,
                                         displayTransform: displayT,
                                         viewSize: viewSize)
                } else {
                    // 프레임 없을 때: 정립 좌표 + 세로 oriented 버퍼로 aspect-fill
                    let v = b.visionBox
                    let norm = CGRect(x: v.minX, y: 1 - v.maxY,
                                      width: v.width, height: v.height)
                    let oriented = CGSize(width: imageSize.height,
                                          height: imageSize.width)
                    rect = Self.aspectFillRect(norm, imageSize: oriented,
                                               viewSize: viewSize)
                }
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
                label.contentsScale = UIScreen.main.scale
                let lh: CGFloat = 17
                label.frame = CGRect(x: rect.minX, y: max(rect.minY - lh, 0),
                                     width: max(rect.width, 60), height: lh)
                overlay.addSublayer(label)
            }
        }

        /// Vision(.right, 원점 좌하단) → 센서 metadata → AR displayTransform → 뷰 픽셀.
        private static func viewRect(visionBox v: CGRect,
                                     displayTransform: CGAffineTransform,
                                     viewSize: CGSize) -> CGRect {
            let metadata = CGRect(x: 1 - v.maxY, y: 1 - v.maxX,
                                  width: v.height, height: v.width)
            let corners = [
                CGPoint(x: metadata.minX, y: metadata.minY),
                CGPoint(x: metadata.maxX, y: metadata.minY),
                CGPoint(x: metadata.minX, y: metadata.maxY),
                CGPoint(x: metadata.maxX, y: metadata.maxY),
            ].map { $0.applying(displayTransform) }
            let xs = corners.map(\.x)
            let ys = corners.map(\.y)
            guard let x0 = xs.min(), let x1 = xs.max(),
                  let y0 = ys.min(), let y1 = ys.max() else { return .zero }
            return CGRect(x: x0 * viewSize.width,
                          y: y0 * viewSize.height,
                          width: (x1 - x0) * viewSize.width,
                          height: (y1 - y0) * viewSize.height)
        }

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
