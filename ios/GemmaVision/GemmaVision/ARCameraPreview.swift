// ARKit 카메라 프리뷰 + YOLO 박스 오버레이 + 분류별 컬러 메쉬(디버그).
// 메쉬 색: floor=녹 / wall=주황 / door=파랑 / window=청록 / 기타=회색.
import ARKit
import SceneKit
import SwiftUI
import UIKit

struct ARCameraPreview: UIViewRepresentable {
    let session: ARSession
    var boxes: [DetBox] = []
    var showMesh: Bool = true
    /// 센서 가로 버퍼 기준 (ARFrame.capturedImage와 동일 계열).
    var imageSize: CGSize = CGSize(width: 1920, height: 1440)

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView(frame: .zero)
        view.isUserInteractionEnabled = false
        view.session = session
        view.delegate = context.coordinator
        view.automaticallyUpdatesLighting = false
        view.rendersCameraGrain = false
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
        var showMesh = true

        func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            guard showMesh, let mesh = anchor as? ARMeshAnchor else { return nil }
            let node = SCNNode(geometry: Self.coloredGeometry(from: mesh))
            node.name = "mesh.\(mesh.identifier.uuidString)"
            node.renderingOrder = -1
            return node
        }

        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode,
                      for anchor: ARAnchor) {
            guard let mesh = anchor as? ARMeshAnchor else { return }
            if showMesh {
                node.isHidden = false
                node.geometry = Self.coloredGeometry(from: mesh)
            } else {
                node.isHidden = true
            }
        }

        private static func coloredGeometry(from meshAnchor: ARMeshAnchor) -> SCNGeometry {
            let geo = meshAnchor.geometry
            let vertices = geo.vertices
            let faces = geo.faces
            let faceCount = faces.count
            let per = faces.indexCountPerPrimitive

            var positions: [SCNVector3] = []
            var colors: [SIMD4<Float>] = []
            positions.reserveCapacity(faceCount * per)
            colors.reserveCapacity(faceCount * per)

            for f in 0..<faceCount {
                let c = color(for: classification(of: f, geometry: geo))
                for k in 0..<per {
                    let idx = vertexIndex(face: f, corner: k, faces: faces)
                    let v = vertex(at: idx, source: vertices)
                    positions.append(SCNVector3(v.x, v.y, v.z))
                    colors.append(c)
                }
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
                primitiveCount: faceCount,
                bytesPerIndex: MemoryLayout<Int32>.size)

            let geometry = SCNGeometry(sources: [posSrc, colorSrc], elements: [element])
            let mat = SCNMaterial()
            mat.lightingModel = .constant
            mat.isDoubleSided = true
            mat.fillMode = .fill
            mat.writesToDepthBuffer = false
            mat.readsFromDepthBuffer = true
            geometry.materials = [mat]
            return geometry
        }

        private static func vertex(at index: Int,
                                   source: ARGeometrySource) -> SIMD3<Float> {
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
            case .floor: return SIMD4(0.15, 0.90, 0.35, 0.42)   // green — walkable
            case .wall: return SIMD4(1.00, 0.55, 0.10, 0.38)    // orange
            case .door: return SIMD4(0.20, 0.45, 1.00, 0.48)    // blue
            case .window: return SIMD4(0.20, 0.90, 0.95, 0.45)  // cyan
            case .ceiling: return SIMD4(0.70, 0.70, 0.75, 0.22)
            case .table, .seat: return SIMD4(0.85, 0.30, 0.85, 0.35)
            default: return SIMD4(0.55, 0.55, 0.55, 0.20)
            }
        }
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

        func setMeshVisible(_ on: Bool) {
            scene.rootNode.enumerateChildNodes { node, _ in
                if node.name?.hasPrefix("mesh.") == true {
                    node.isHidden = !on
                }
            }
        }

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

            // 구조물 인식 밴드(가로 중앙 1/3) 가이드 — 좌우 벽 무시 구간 시각화
            let band = CAShapeLayer()
            let path = UIBezierPath()
            let x1 = viewSize.width / 3
            let x2 = viewSize.width * 2 / 3
            path.move(to: CGPoint(x: x1, y: 0)); path.addLine(to: CGPoint(x: x1, y: viewSize.height))
            path.move(to: CGPoint(x: x2, y: 0)); path.addLine(to: CGPoint(x: x2, y: viewSize.height))
            band.path = path.cgPath
            band.strokeColor = UIColor.white.withAlphaComponent(0.35).cgColor
            band.lineWidth = 1
            band.lineDashPattern = [6, 4]
            overlay.addSublayer(band)

            for b in boxes {
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
