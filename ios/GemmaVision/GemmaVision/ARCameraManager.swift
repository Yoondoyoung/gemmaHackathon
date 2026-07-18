// LiDAR + ARKit Scene Geometry — 벽/문/창을 메쉬 분류로 인식하고 미터 거리를 반환.
// YOLO가 못 잡는 구조물만 담당. 사람·가방 등 동적 물체는 기존 YOLO 경로 유지.
// SegFormer 같은 2D 세그 모델 없이 NPU/LiDAR 하드웨어 경로만 사용.
import ARKit
import Combine
import CoreVideo
import Foundation
import simd
import UIKit

struct StructureHit: Equatable {
    let label: String   // wall | door | window | floor | fork
    let meters: Double  // 장애물=최근접 / floor·fork=바닥이 이어지는 거리
    let pos: String     // left | center | right | both(fork)
}

final class ARCameraManager: NSObject, ObservableObject, @unchecked Sendable {
    /// iPhone 12 Pro+ 등 scene reconstruction + classification 지원 여부.
    static var isSupported: Bool {
        ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification)
    }

    let session = ARSession()
    @Published var hasSceneMesh = false
    var onFrame: ((CVPixelBuffer, CVPixelBuffer?) -> Void)?
    var onStructures: (([StructureHit]) -> Void)?

    private let queue = DispatchQueue(label: "ar.camera")
    private var lastStructureAt = Date.distantPast
    private static let structurePeriod: TimeInterval = 0.25
    /// 화면 가로 3등분 — 벽/문/창은 중앙만. 바닥은 좌·중·우(갈림길).
    private static let screenCenterMinX: CGFloat = 1.0 / 3.0
    private static let screenCenterMaxX: CGFloat = 2.0 / 3.0
    private static let screenCenterMinY: CGFloat = 0.15
    private static let screenCenterMaxY: CGFloat = 0.85
    /// 측면 통로: 카메라 좌표 |x|가 이 이상이어야 "옆으로 열린 바닥"
    /// (직진 복도 바닥이 화면 좌우에 보이는 것과 구분).
    private static let sideBranchLateralM: Double = 0.65
    private static let sideBranchMinDepthM: Double = 1.2
    private static let sideBranchMaxDepthM: Double = 4.0
    /// 장애물 클래스 (seat/table은 YOLO와 중복 → 제외).
    private static let obstacleClasses: Set<ARMeshClassification> = [.wall, .door, .window]
    private static let walkClasses: Set<ARMeshClassification> = [.floor]

    func start() {
        guard Self.isSupported else { return }
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = .meshWithClassification
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        session.delegate = self
        session.delegateQueue = queue
        queue.async {
            self.session.run(config, options: [.resetTracking, .removeExistingAnchors])
            DispatchQueue.main.async { self.hasSceneMesh = true }
        }
    }

    func pause() {
        session.pause()
    }
}

extension ARCameraManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let pixel = frame.capturedImage
        let depth = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap
        onFrame?(pixel, depth)

        let now = Date()
        guard now.timeIntervalSince(lastStructureAt) >= Self.structurePeriod else { return }
        lastStructureAt = now
        let hits = Self.scanStructures(frame: frame)
        if !hits.isEmpty { onStructures?(hits) }
    }
}

// MARK: - Mesh → 정면 구조물

private extension ARCameraManager {
    static func scanStructures(frame: ARFrame) -> [StructureHit] {
        let meshes = frame.anchors.compactMap { $0 as? ARMeshAnchor }
        guard !meshes.isEmpty else { return [] }

        let worldToCamera = simd_inverse(frame.camera.transform)
        let viewport = CGSize(width: 390, height: 844)
        let orientation = UIInterfaceOrientation.portrait

        var bestObstacle: [String: StructureHit] = [:]
        // 바닥: 밴드별 최원 거리 (center=직진 / left·right=옆으로 열린 통로)
        var floorCenterM: Double?
        var floorLeftM: Double?
        var floorRightM: Double?

        for anchor in meshes {
            let geometry = anchor.geometry
            let faceCount = geometry.faces.count
            let step = max(1, faceCount / 800)
            var faceIndex = 0
            while faceIndex < faceCount {
                defer { faceIndex += step }
                let cls = classificationOf(faceIndex: faceIndex, geometry: geometry)
                let isObstacle = obstacleClasses.contains(cls)
                let isFloor = walkClasses.contains(cls)
                guard isObstacle || isFloor else { continue }
                guard let centerWorld = faceCenter(faceIndex: faceIndex,
                                                   geometry: geometry,
                                                   transform: anchor.transform)
                else { continue }

                let p4 = worldToCamera * SIMD4<Float>(centerWorld, 1)
                guard p4.z < 0 else { continue }
                let depth = Double(-p4.z)
                guard depth > 0.25, depth < Config.mediumMeters + 1.5 else { continue }

                let projected = frame.camera.projectPoint(
                    centerWorld, orientation: orientation, viewportSize: viewport)
                let nx = projected.x / viewport.width
                let ny = projected.y / viewport.height
                guard ny >= screenCenterMinY, ny <= screenCenterMaxY else { continue }

                let band: String
                if nx < screenCenterMinX { band = "left" }
                else if nx > screenCenterMaxX { band = "right" }
                else { band = "center" }

                if isFloor {
                    if let nWorld = faceNormalWorld(faceIndex: faceIndex, geometry: geometry,
                                                    anchor: anchor.transform),
                       abs(nWorld.y) < 0.65 {
                        continue
                    }
                    if band == "center" {
                        if floorCenterM == nil || depth > floorCenterM! {
                            floorCenterM = depth
                        }
                    } else {
                        // 측면 통로: 화면 좌/우 + 실제 옆으로 벌어진 바닥만
                        // (직진 복도 바닥이 프레임 좌우에 비치는 것 제외)
                        let lateral = abs(Double(p4.x))
                        guard lateral >= sideBranchLateralM,
                              depth >= sideBranchMinDepthM,
                              depth <= sideBranchMaxDepthM else { continue }
                        if band == "left" {
                            if floorLeftM == nil || depth > floorLeftM! { floorLeftM = depth }
                        } else {
                            if floorRightM == nil || depth > floorRightM! { floorRightM = depth }
                        }
                    }
                    continue
                }

                // 벽/문/창: 중앙 밴드만
                guard band == "center" else { continue }
                if let nCam = faceNormalCamera(faceIndex: faceIndex, geometry: geometry,
                                               anchor: anchor.transform,
                                               worldToCamera: worldToCamera),
                   abs(nCam.z) < abs(nCam.x) {
                    continue
                }
                let label = labelName(cls)
                let hit = StructureHit(label: label, meters: depth, pos: "center")
                if let prev = bestObstacle[label], prev.meters <= hit.meters { continue }
                bestObstacle[label] = hit
            }
        }

        var out = Array(bestObstacle.values)
        if let m = floorCenterM {
            out.append(StructureHit(label: "floor", meters: m, pos: "center"))
        }
        if let m = floorLeftM {
            out.append(StructureHit(label: "floor", meters: m, pos: "left"))
        }
        if let m = floorRightM {
            out.append(StructureHit(label: "floor", meters: m, pos: "right"))
        }
        // 갈림길: 좌·우 측면 바닥이 동시에, 또는 한쪽에만 열린 통로
        let left = floorLeftM != nil
        let right = floorRightM != nil
        if left || right {
            let pos: String
            let meters: Double
            if left && right {
                pos = "both"
                meters = min(floorLeftM!, floorRightM!)
            } else if left {
                pos = "left"
                meters = floorLeftM!
            } else {
                pos = "right"
                meters = floorRightM!
            }
            out.append(StructureHit(label: "fork", meters: meters, pos: pos))
        }
        return out.sorted { $0.meters < $1.meters }
    }

    static func labelName(_ c: ARMeshClassification) -> String {
        switch c {
        case .door: return "door"
        case .window: return "window"
        case .floor: return "floor"
        default: return "wall"
        }
    }

    static func faceNormalWorld(faceIndex: Int, geometry: ARMeshGeometry,
                                anchor: simd_float4x4) -> SIMD3<Float>? {
        let indices = vertexIndices(faceIndex: faceIndex, geometry: geometry)
        guard indices.count == 3 else { return nil }
        let v0 = worldVertex(indices[0], geometry: geometry, transform: anchor)
        let v1 = worldVertex(indices[1], geometry: geometry, transform: anchor)
        let v2 = worldVertex(indices[2], geometry: geometry, transform: anchor)
        return simd_normalize(simd_cross(v1 - v0, v2 - v0))
    }

    static func classificationOf(faceIndex: Int,
                                 geometry: ARMeshGeometry) -> ARMeshClassification {
        guard let classifications = geometry.classification else { return .none }
        let ptr = classifications.buffer.contents()
            .advanced(by: classifications.offset + classifications.stride * faceIndex)
        let raw = Int(ptr.assumingMemoryBound(to: UInt8.self).pointee)
        return ARMeshClassification(rawValue: raw) ?? .none
    }

    static func faceCenter(faceIndex: Int, geometry: ARMeshGeometry,
                           transform: simd_float4x4) -> SIMD3<Float>? {
        let indices = vertexIndices(faceIndex: faceIndex, geometry: geometry)
        guard indices.count == 3 else { return nil }
        let v0 = worldVertex(indices[0], geometry: geometry, transform: transform)
        let v1 = worldVertex(indices[1], geometry: geometry, transform: transform)
        let v2 = worldVertex(indices[2], geometry: geometry, transform: transform)
        return (v0 + v1 + v2) / 3
    }

    static func faceNormalCamera(faceIndex: Int, geometry: ARMeshGeometry,
                                 anchor: simd_float4x4,
                                 worldToCamera: simd_float4x4) -> SIMD3<Float>? {
        let indices = vertexIndices(faceIndex: faceIndex, geometry: geometry)
        guard indices.count == 3 else { return nil }
        let v0 = worldVertex(indices[0], geometry: geometry, transform: anchor)
        let v1 = worldVertex(indices[1], geometry: geometry, transform: anchor)
        let v2 = worldVertex(indices[2], geometry: geometry, transform: anchor)
        let n = simd_normalize(simd_cross(v1 - v0, v2 - v0))
        let n4 = worldToCamera * SIMD4<Float>(n.x, n.y, n.z, 0)
        return SIMD3<Float>(n4.x, n4.y, n4.z)
    }

    static func worldVertex(_ index: UInt32, geometry: ARMeshGeometry,
                            transform: simd_float4x4) -> SIMD3<Float> {
        let vertices = geometry.vertices
        let ptr = vertices.buffer.contents()
            .advanced(by: vertices.offset + vertices.stride * Int(index))
        let local = ptr.assumingMemoryBound(to: SIMD3<Float>.self).pointee
        let w = transform * SIMD4<Float>(local, 1)
        return SIMD3<Float>(w.x, w.y, w.z)
    }

    static func vertexIndices(faceIndex: Int,
                              geometry: ARMeshGeometry) -> [UInt32] {
        let faces = geometry.faces
        let per = faces.indexCountPerPrimitive
        var out: [UInt32] = []
        out.reserveCapacity(per)
        let base = faces.buffer.contents()
        for i in 0..<per {
            let offset = (faceIndex * per + i) * faces.bytesPerIndex
            let ptr = base.advanced(by: offset)
            if faces.bytesPerIndex == 2 {
                out.append(UInt32(ptr.assumingMemoryBound(to: UInt16.self).pointee))
            } else {
                out.append(ptr.assumingMemoryBound(to: UInt32.self).pointee)
            }
        }
        return out
    }
}
