// 발화 출력 — Mac판 Speaker의 우선순위 정책 이식:
// priority 0(경고)은 진행 중인 안내(priority 1)를 즉시 끊는다.
import AVFoundation

final class SpeechOut: NSObject {
    static let shared = SpeechOut()
    private let synth = AVSpeechSynthesizer()
    private var currentPriority = 1

    func say(_ text: String, priority: Int = 1) {
        activatePlaybackSession()
        if priority == 0, synth.isSpeaking, currentPriority > 0 {
            synth.stopSpeaking(at: .immediate)   // 안전 > 안내 (대기열까지 비워짐)
        }
        currentPriority = priority
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.55   // 시각장애 사용자는 빠른 음성에 익숙
        utterance.volume = 1.0
        synth.speak(utterance)   // AVSpeech가 자체적으로 순차 큐잉 (스트리밍 문장 대응)
    }

    /// Push-to-talk 시작 시 TTS를 끊어 마이크 인식을 방해하지 않게 한다.
    func stop() {
        if synth.isSpeaking {
            synth.stopSpeaking(at: .immediate)
        }
    }

    /// PTT의 `.playAndRecord`/`.measurement` 세션이 TTS 볼륨을 깎은 뒤 복구.
    func activatePlaybackSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            // 세션 복구 실패해도 발화는 시도
        }
    }
}
