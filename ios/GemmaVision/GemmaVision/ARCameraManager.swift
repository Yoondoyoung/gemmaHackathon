// LiDAR + ARKit Scene Geometry — 벽/문/창/바닥/갈림길.
// 경로·갈림길 기억은 GPS가 아니라 ARKit 월드 좌표(온디바이스)를 쓴다.
// ARWorldMap 로컬 저장으로 같은 공간을 다시 열면 재로컬라이즈 가능.
import ARKit
import Combine
import CoreVideo
import Foundation
import simd
import UIKit

struct StructureHit: Equatable {
    let label: String   // wall | door | window | floor | fork
    let meters: Double
    let pos: String     // left | center | right | both
}

/// 한 틱의 구조물 스캔 + 카메라 포즈 (경로/갈림길 웨이포인트용).
struct StructureUpdate {
    let hits: [StructureHit]
    let cameraPosition: SIMD3<Float>   // 월드 좌표
    let cameraForward: SIMD3<Float>    // 수평 전방 단위벡터
}

final class ARCameraManager: NSObject, ObservableObject, @unchecked Sendable {
    static var isSupported: Bool {
        ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification)
    }

    let session = ARSession()
    @Published var hasSceneMesh = false
    @Published var mapStatus = ""          // WorldMap 저장/로드 상태
    /// (영상, 깊이, 카메라 월드 위치, 수평 전방)
    var onFrame: ((CVPixelBuffer, CVPixelBuffer?, SIMD3<Float>, SIMD3<Float>) -> Void)?
    var onStructures: ((StructureUpdate) -> Void)?

    private let queue = DispatchQueue(label: "ar.camera")
    private var lastStructureAt = Date.distantPast
    private static let structurePeriod: TimeInterval = 0.4   // 메쉬 스캔 부하↓

    private static let screenCenterMinX: CGFloat = 1.0 / 3.0
    private static let screenCenterMaxX: CGFloat = 2.0 / 3.0
    private static let screenCenterMinY: CGFloat = 0.20
    private static let screenCenterMaxY: CGFloat = 0.80

    /// 측면 통로: 옆으로 충분히 벌어진 바닥 + 샘플 수 + 연속 프레임.
    private static let sideBranchLateralM: Double = 0.90
    private static let sideBranchMinDepthM: Double = 1.4
    private static let sideBranchMaxDepthM: Double = 3.5
    private static let sideMinSamples = 5
    private static let forkConfirmStreak = 3          // ~0.75초 유지돼야 확정
    /// 넓은 홀 오탐 방지: 좌우가 둘 다 열려 있고 전방 바닥이 너무 멀면 갈림길 아님.
    private static let openRoomCenterMinM: Double = 3.8

    private static let obstacleClasses: Set<ARMeshClassification> = [.wall, .door, .window]
    private static let walkClasses: Set<ARMeshClassification> = [.floor]

    private var leftStreak = 0
    private var rightStreak = 0

    private static var mapURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("visionassist_world.map")
    }

    func start(withSavedMap: Bool = false) {
        guard Self.isSupported else { return }
        queue.async {
            let config = self.makeConfig()
            if withSavedMap, let map = self.loadMapFromDisk() {
                config.initialWorldMap = map
                DispatchQueue.main.async { self.mapStatus = "map: relocalizing…" }
            }
            self.session.delegate = self
            self.session.delegateQueue = self.queue
            self.session.run(config, options: [.resetTracking, .removeExistingAnchors])
            DispatchQueue.main.async { self.hasSceneMesh = true }
        }
    }

    func pause() { session.pause() }

    /// 현재 공간 특징점+앵커를 Documents에 저장 (완전 온디바이스, 인터넷 불필요).
    func saveWorldMap() {
        session.getCurrentWorldMap { [weak self] map, error in
            guard let self else { return }
            if let error {
                DispatchQueue.main.async { self.mapStatus = "map save failed: \(error.localizedDescription)" }
                return
            }
            guard let map else {
                DispatchQueue.main.async { self.mapStatus = "map save failed: empty" }
                return
            }
            do {
                let data = try NSKeyedArchiver.archivedData(withRootObject: map,
                                                            requiringSecureCoding: true)
                try data.write(to: Self.mapURL, options: .atomic)
                DispatchQueue.main.async {
                    self.mapStatus = "map saved (\(data.count / 1024)KB)"
                }
            } catch {
                DispatchQueue.main.async {
                    self.mapStatus = "map save failed: \(error.localizedDescription)"
                }
            }
        }
    }

    /// 저장된 맵으로 세션 재시작 → 같은 공간이면 재로컬라이즈.
    func loadWorldMap() {
        queue.async {
            guard let map = self.loadMapFromDisk() else {
                DispatchQueue.main.async { self.mapStatus = "no saved map" }
                return
            }
            let config = self.makeConfig()
            config.initialWorldMap = map
            self.session.run(config, options: [.resetTracking, .removeExistingAnchors])
            DispatchQueue.main.async { self.mapStatus = "map loaded — look around to relocalize" }
        }
    }

    private func makeConfig() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = .meshWithClassification
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        return config
    }

    private func loadMapFromDisk() -> ARWorldMap? {
        guard let data = try? Data(contentsOf: Self.mapURL) else { return nil }
        return try? NSKeyedUnarchiver.unarchivedObject(ofClass: ARWorldMap.self, from: data)
    }
}

extension ARCameraManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let pixel = frame.capturedImage
        let depth = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap
        let (position, forward) = Self.pose(from: frame.camera.transform)
        onFrame?(pixel, depth, position, forward)

        let now = Date()
        guard now.timeIntervalSince(lastStructureAt) >= Self.structurePeriod else { return }
        lastStructureAt = now

        let hits = scanStructures(frame: frame)
        onStructures?(StructureUpdate(hits: hits, cameraPosition: position,
                                      cameraForward: forward))
    }

    static func pose(from cam: simd_float4x4) -> (SIMD3<Float>, SIMD3<Float>) {
        let position = SIMD3<Float>(cam.columns.3.x, cam.columns.3.y, cam.columns.3.z)
        var forward = -SIMD3<Float>(cam.columns.2.x, 0, cam.columns.2.z)
        let flen = simd_length(forward)
        if flen > 1e-4 { forward /= flen } else { forward = SIMD3(0, 0, -1) }
        return (position, forward)
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        DispatchQueue.main.async {
            switch camera.trackingState {
            case .normal:
                if self.mapStatus.contains("relocaliz") || self.mapStatus.contains("loaded") {
                    self.mapStatus = "map: tracking OK"
                }
            case .limited(let reason):
                self.mapStatus = "tracking limited: \(reason)"
            case .notAvailable:
                self.mapStatus = "tracking unavailable"
            @unknown default: break
            }
        }
    }
}

// MARK: - Mesh 스캔

private extension ARCameraManager {
    func scanStructures(frame: ARFrame) -> [StructureHit] {
        let meshes = frame.anchors.compactMap { $0 as? ARMeshAnchor }
        guard !meshes.isEmpty else { return [] }

        let worldToCamera = simd_inverse(frame.camera.transform)
        let viewport = CGSize(width: 390, height: 844)
        let orientation = UIInterfaceOrientation.portrait

        var bestObstacle: [String: StructureHit] = [:]
        var floorCenterM: Double?
        var leftDepthSum = 0.0, leftCount = 0
        var rightDepthSum = 0.0, rightCount = 0

        for anchor in meshes {
            let geometry = anchor.geometry
            let faceCount = geometry.faces.count
            let step = max(1, faceCount / 800)
            var faceIndex = 0
            while faceIndex < faceCount {
                defer { faceIndex += step }
                let cls = Self.classificationOf(faceIndex: faceIndex, geometry: geometry)
                let isObstacle = Self.obstacleClasses.contains(cls)
                let isFloor = Self.walkClasses.contains(cls)
                guard isObstacle || isFloor else { continue }
                guard let centerWorld = Self.faceCenter(faceIndex: faceIndex,
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
                guard ny >= Self.screenCenterMinY, ny <= Self.screenCenterMaxY else { continue }

                let band: String
                if nx < Self.screenCenterMinX { band = "left" }
                else if nx > Self.screenCenterMaxX { band = "right" }
                else { band = "center" }

                if isFloor {
                    if let nWorld = Self.faceNormalWorld(faceIndex: faceIndex, geometry: geometry,
                                                         anchor: anchor.transform),
                       abs(nWorld.y) < 0.65 {
                        continue
                    }
                    if band == "center" {
                        if floorCenterM == nil || depth > floorCenterM! { floorCenterM = depth }
                    } else {
                        let lateral = abs(Double(p4.x))
                        guard lateral >= Self.sideBranchLateralM,
                              depth >= Self.sideBranchMinDepthM,
                              depth <= Self.sideBranchMaxDepthM else { continue }
                        if band == "left" {
                            leftDepthSum += depth; leftCount += 1
                        } else {
                            rightDepthSum += depth; rightCount += 1
                        }
                    }
                    continue
                }

                guard band == "center" else { continue }
                if let nCam = Self.faceNormalCamera(faceIndex: faceIndex, geometry: geometry,
                                                    anchor: anchor.transform,
                                                    worldToCamera: worldToCamera),
                   abs(nCam.z) < abs(nCam.x) {
                    continue
                }
                let label = Self.labelName(cls)
                let hit = StructureHit(label: label, meters: depth, pos: "center")
                if let prev = bestObstacle[label], prev.meters <= hit.meters { continue }
                bestObstacle[label] = hit
            }
        }

        var leftM = leftCount > 0 ? leftDepthSum / Double(leftCount) : nil
        var rightM = rightCount > 0 ? rightDepthSum / Double(rightCount) : nil
        // 전방 복도 깊이와 비슷한 측면만 남김 (스캔 순서와 무관하게 최종 필터)
        if let c = floorCenterM {
            let ref = min(c, 3.0)
            if let m = leftM, abs(m - ref) > 1.6 { leftM = nil; leftCount = 0 }
            if let m = rightM, abs(m - ref) > 1.6 { rightM = nil; rightCount = 0 }
        }

        let rawLeft = leftCount >= Self.sideMinSamples && leftM != nil
        let rawRight = rightCount >= Self.sideMinSamples && rightM != nil
        leftStreak = rawLeft ? leftStreak + 1 : 0
        rightStreak = rawRight ? rightStreak + 1 : 0
        let leftOK = leftStreak >= Self.forkConfirmStreak
        let rightOK = rightStreak >= Self.forkConfirmStreak

        var out = Array(bestObstacle.values)
        if let m = floorCenterM {
            out.append(StructureHit(label: "floor", meters: m, pos: "center"))
        }
        if leftOK, let m = leftM {
            out.append(StructureHit(label: "floor", meters: m, pos: "left"))
        }
        if rightOK, let m = rightM {
            out.append(StructureHit(label: "floor", meters: m, pos: "right"))
        }

        // 갈림길: 전방 바닥(복도 맥락) 있을 때만. 넓은 홀(좌우+먼 전방) 제외.
        let hasCorridor = (floorCenterM ?? 0) >= 1.2
        let openRoom = leftOK && rightOK && (floorCenterM ?? 0) >= Self.openRoomCenterMinM
        if hasCorridor && !openRoom && (leftOK || rightOK) {
            let pos: String
            let meters: Double
            if leftOK && rightOK {
                pos = "both"
                meters = min(leftM!, rightM!)
            } else if leftOK {
                pos = "left"; meters = leftM!
            } else {
                pos = "right"; meters = rightM!
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
