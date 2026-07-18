// 발화 출력 — Mac판 Speaker의 우선순위 정책 이식:
// priority 0(경고)은 진행 중인 안내(priority 1)를 즉시 끊는다.
import AVFoundation

final class SpeechOut: NSObject {
    static let shared = SpeechOut()
    private let synth = AVSpeechSynthesizer()
    private var currentPriority = 1

    func say(_ text: String, priority: Int = 1) {
        if priority == 0, synth.isSpeaking, currentPriority > 0 {
            synth.stopSpeaking(at: .immediate)   // 안전 > 안내 (대기열까지 비워짐)
        }
        currentPriority = priority
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.55   // 시각장애 사용자는 빠른 음성에 익숙
        synth.speak(utterance)   // AVSpeech가 자체적으로 순차 큐잉 (스트리밍 문장 대응)
    }
}
