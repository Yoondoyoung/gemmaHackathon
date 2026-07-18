// 영상 파일 테스트 하네스 — Mac판 `--video`의 iOS 대응.
// 번들에 넣은 .mp4를 디코드해 파이프라인에 먹이고, 프레임+박스+경고를 화면에 표시.
// 사용법: 테스트 클립(nyc2.mp4 등)을 앱 타깃에 드래그 → VisionAssistApp의 루트를
//   VideoTestView()로 잠깐 교체 → 실행. (LiDAR 없음 → 거리는 bbox 휴리스틱 폴백)
import AVFoundation
import CoreImage
import SwiftUI
import UIKit

final class VideoTester: ObservableObject {
    @Published var frame: UIImage?
    @Published var info = "loading video…"
    private var task: Task<Void, Never>?
    private let ctx = CIContext(options: [.useSoftwareRenderer: false])

    func start(pipeline: Pipeline, clip: String?) {
        pipeline.inputOrientation = .up   // 영상은 이미 정립 (센서 회전 불필요)
        task = Task.detached(priority: .userInitiated) { [weak self] in
            await self?.loop(pipeline: pipeline, clip: clip)
        }
    }

    func stop() { task?.cancel() }

    private func bundledClip(_ clip: String?) -> URL? {
        if let clip, let u = Bundle.main.url(forResource: clip, withExtension: nil) {
            return u
        }
        for ext in ["mp4", "mov", "m4v"] {
            if let u = Bundle.main.urls(forResourcesWithExtension: ext,
                                        subdirectory: nil)?.first {
                return u
            }
        }
        return nil
    }

    private func loop(pipeline: Pipeline, clip: String?) async {
        guard let url = bundledClip(clip) else {
            await MainActor.run { self.info = "번들에 영상 없음 — .mp4를 타깃에 추가" }
            return
        }
        await MainActor.run { self.info = "playing \(url.lastPathComponent)" }
        while !Task.isCancelled {                 // 반복 재생
            await playOnce(url: url, pipeline: pipeline)
        }
    }

    private func playOnce(url: URL, pipeline: Pipeline) async {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              let reader = try? AVAssetReader(asset: asset) else { return }
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [String(kCVPixelBufferPixelFormatTypeKey):
                                kCVPixelFormatType_32BGRA])
        guard reader.canAdd(output) else { return }
        reader.add(output)
        reader.startReading()

        let start = CACurrentMediaTime()
        while !Task.isCancelled,
              let sample = output.copyNextSampleBuffer(),
              let pixel = CMSampleBufferGetImageBuffer(sample) {
            let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            // 원본 재생 속도에 맞춰 페이싱
            let wait = pts - (CACurrentMediaTime() - start)
            if wait > 0 { try? await Task.sleep(nanoseconds: UInt64(wait * 1e9)) }

            pipeline.process(pixel, depth: nil)   // 깊이 없음 → 휴리스틱 폴백
            if let img = uiImage(from: pixel) {
                await MainActor.run { self.frame = img }
            }
        }
        reader.cancelReading()
    }

    private func uiImage(from pixelBuffer: CVPixelBuffer) -> UIImage? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        return UIImage(cgImage: cg)
    }
}

struct VideoTestView: View {
    var clip: String? = nil            // 특정 파일명 지정, nil이면 번들 첫 영상
    @StateObject private var pipeline = Pipeline()
    @StateObject private var tester = VideoTester()

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .top) {
                Color.black.ignoresSafeArea()
                if let frame = tester.frame {
                    let fit = fittedRect(imageSize: frame.size, in: geo.size)
                    Image(uiImage: frame)
                        .resizable()
                        .frame(width: fit.width, height: fit.height)
                        .position(x: fit.midX, y: fit.midY)
                    ForEach(pipeline.boxes) { b in
                        boxView(b, in: fit)
                    }
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text(tester.info).font(.caption2).foregroundColor(.white)
                    Text(pipeline.statusLine).font(.caption.monospaced())
                        .foregroundColor(.green)
                    Text(pipeline.lastSpoken).font(.headline).foregroundColor(.yellow)
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.black.opacity(0.5))
            }
        }
        .onAppear { tester.start(pipeline: pipeline, clip: clip) }
        .onDisappear { tester.stop() }
    }

    // 영상 정규화 박스(원점 좌하단) → 화면 좌표 (aspectFit이라 매핑이 정확)
    private func boxView(_ b: DetBox, in fit: CGRect) -> some View {
        let v = b.visionBox
        let x = fit.minX + v.minX * fit.width
        let y = fit.minY + (1 - v.maxY) * fit.height     // Y 뒤집기
        let w = v.width * fit.width
        let h = v.height * fit.height
        let color: Color = b.alert ? .red : .green
        return ZStack(alignment: .topLeading) {
            Rectangle().stroke(color, lineWidth: 2.5)
                .frame(width: w, height: h)
            Text(b.text).font(.caption2).foregroundColor(.black)
                .padding(.horizontal, 3)
                .background(color)
                .offset(y: -15)
        }
        .frame(width: w, height: h, alignment: .topLeading)
        .position(x: x + w / 2, y: y + h / 2)
    }

    private func fittedRect(imageSize: CGSize, in container: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let scale = min(container.width / imageSize.width,
                        container.height / imageSize.height)
        let w = imageSize.width * scale
        let h = imageSize.height * scale
        return CGRect(x: (container.width - w) / 2, y: (container.height - h) / 2,
                      width: w, height: h)
    }
}
