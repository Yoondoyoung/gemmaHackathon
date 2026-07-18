// 발화 출력 — Mac판 Speaker의 우선순위 정책 이식:
// priority 0(경고)은 진행 중인 안내(priority 1)를 즉시 끊는다.
import AVFoundation

final class SpeechOut: NSObject {
    static let shared = SpeechOut()
    private let synth = AVSpeechSynthesizer()
    private var currentPriority = 1

    func say(_ text: String, priority: Int = 1) {
        if priority == 0 {
            if synth.isSpeaking && currentPriority > 0 {
                synth.stopSpeaking(at: .immediate)   // 안전 > 안내
            }
        } else if synth.isSpeaking {
            return   // 발화 중 저순위 추가는 버림 (PoC 단순화)
        }
        currentPriority = priority
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.55   // 시각장애 사용자는 빠른 음성에 익숙
        synth.speak(utterance)
    }
}
