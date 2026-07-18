// 시각 보조 어시스턴트 iOS PoC — Mac 파이프라인의 모바일 이식 검증용.
// 스코프: ARKit 메쉬(벽/문/창) + YOLO(CoreML/ANE) + 룰베이스 경고 + Vision OCR
// + AVSpeech + Gemma 4 E2B 온디바이스 Q&A.
import SwiftUI

@main
struct VisionAssistApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
            // 영상 파일로 테스트할 때만 위를 주석 처리하고 아래를 사용:
            // VideoTestView(clip: "nyc2.mp4")   // nil이면 번들 첫 영상
        }
    }
}
