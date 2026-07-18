// 후면 카메라 캡처 — 최신 프레임만 파이프라인에 전달 (버퍼링 없음, Mac판과 동일 원칙).
// Pro 기기(12 Pro+)는 LiDAR로 미터 단위 깊이 맵을 동기 전달, 아니면 nil (휴리스틱 폴백).
import AVFoundation
import Combine
import CoreVideo

final class CameraManager: NSObject, ObservableObject,
                           AVCaptureVideoDataOutputSampleBufferDelegate,
                           AVCaptureDataOutputSynchronizerDelegate {
    let session = AVCaptureSession()
    @Published var hasLiDAR = false
    private let queue = DispatchQueue(label: "camera.queue")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    var onFrame: ((CVPixelBuffer, CVPixelBuffer?) -> Void)?   // (영상, 깊이[m]|nil)

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return }
            self.queue.async {
                self.configure()
                self.session.startRunning()
            }
        }
    }

    private func configure() {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        videoOutput.alwaysDiscardsLateVideoFrames = true   // 오래된 프레임은 버림

        // 1순위: LiDAR (실측 미터 깊이 — Mac판 Depth Anything의 하드웨어 대체)
        if let lidar = AVCaptureDevice.default(.builtInLiDARDepthCamera,
                                               for: .video, position: .back),
           let input = try? AVCaptureDeviceInput(device: lidar),
           session.canAddInput(input) {
            session.sessionPreset = .high
            session.addInput(input)
            guard session.canAddOutput(videoOutput),
                  session.canAddOutput(depthOutput) else { return }
            session.addOutput(videoOutput)
            session.addOutput(depthOutput)
            depthOutput.isFilteringEnabled = true          // 구멍 메움 (안정화)
            let sync = AVCaptureDataOutputSynchronizer(
                dataOutputs: [videoOutput, depthOutput])
            sync.setDelegate(self, queue: queue)
            synchronizer = sync
            DispatchQueue.main.async { self.hasLiDAR = true }
            return
        }

        // 폴백: 일반 광각 (깊이 없음 → bbox 휴리스틱)
        session.sessionPreset = .vga640x480
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return }
        session.addInput(input)
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
    }

    // LiDAR 경로: 영상+깊이 동기 프레임
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let video = collection.synchronizedData(for: videoOutput)
                as? AVCaptureSynchronizedSampleBufferData,
              !video.sampleBufferWasDropped,
              let pixel = CMSampleBufferGetImageBuffer(video.sampleBuffer) else { return }
        var depthMap: CVPixelBuffer?
        if let depth = collection.synchronizedData(for: depthOutput)
                as? AVCaptureSynchronizedDepthData,
           !depth.depthDataWasDropped {
            depthMap = depth.depthData
                .converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
                .depthDataMap
        }
        onFrame?(pixel, depthMap)
    }

    // 폴백 경로: 영상만
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(buffer, nil)
    }
}
