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
    let label: String   // wall | door | window | floor
    let meters: Double  // 장애물=최근접 / floor=전방 바닥이 이어지는 거리
    let pos: String     // left | center | right
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
    /// 화면 가로 3등분 — 중앙 밴드만 구조물(벽/문/창/바닥) 인식. 좌·우는 무시.
    private static let screenCenterMinX: CGFloat = 1.0 / 3.0
    private static let screenCenterMaxX: CGFloat = 2.0 / 3.0
    /// 세로도 가장자리(천장/발밑) 제외
    private static let screenCenterMinY: CGFloat = 0.15
    private static let screenCenterMaxY: CGFloat = 0.85
    /// 장애물 클래스 (seat/table은 YOLO와 중복 → 제외).
    private static let obstacleClasses: Set<ARMeshClassification> = [.wall, .door, .window]
    /// 이동 가능 바닥.
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
        // 프리뷰와 같은 세로 화면 기준으로 투영 (가로 3등분 중앙만 사용)
        let viewport = CGSize(width: 390, height: 844)
        let orientation = UIInterfaceOrientation.portrait

        // 장애물: 라벨별 최근접 / 바닥: 중앙 밴드 최원
        var bestObstacle: [String: StructureHit] = [:]
        var bestFloor: StructureHit?

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
                guard p4.z < 0 else { continue }   // ARKit: -Z = 전방
                let depth = Double(-p4.z)
                guard depth > 0.25, depth < Config.mediumMeters + 1.5 else { continue }

                // ★ 화면 3등분: 좌·우에 투영되면 벽/문/창/바닥 전부 무시
                let projected = frame.camera.projectPoint(
                    centerWorld, orientation: orientation, viewportSize: viewport)
                let nx = projected.x / viewport.width
                let ny = projected.y / viewport.height
                guard nx >= screenCenterMinX, nx <= screenCenterMaxX,
                      ny >= screenCenterMinY, ny <= screenCenterMaxY
                else { continue }

                if isFloor {
                    if let nWorld = faceNormalWorld(faceIndex: faceIndex, geometry: geometry,
                                                    anchor: anchor.transform),
                       abs(nWorld.y) < 0.65 {
                        continue
                    }
                    if bestFloor == nil || depth > bestFloor!.meters {
                        bestFloor = StructureHit(label: "floor", meters: depth, pos: "center")
                    }
                    continue
                }

                // 옆을 향한 면(복도 옆벽이 프레임 중앙에 살짝 걸치는 경우) 추가 차단
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
        if let floor = bestFloor { out.append(floor) }
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
