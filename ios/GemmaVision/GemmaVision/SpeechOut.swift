// 발화 출력 — 우선순위 정책:
// - priority 0(경고)는 진행 중인 안내(priority 1)를 즉시 끊는다 (안전 > 안내).
// - Q&A 답변 재생 중 / PTT로 듣는 중에는 경고를 억제한다.
// - PTT 시작 시 stop()으로 TTS·답변 보호를 전부 끊는다 (마이크 인식 방해 방지).
import AVFoundation

final class SpeechOut: NSObject, AVSpeechSynthesizerDelegate {
    static let shared = SpeechOut()
    private let synth = AVSpeechSynthesizer()
    private let lock = NSLock()
    private var currentPriority = 1
    private var isAnswering = false     // 답변 생성~발화 완료 구간
    private var streamDone = true       // 답변 문장이 더 안 들어옴
    private var answerIDs = Set<ObjectIdentifier>()
    private var micOpen = false         // PTT 듣는 중
    private var muted = false           // interrupt 후 잔여 say() 드롭

    override init() {
        super.init()
        synth.delegate = self
    }

    /// Pipeline에서 경고를 낼지 판단 — 답변·마이크 중이면 true.
    var suppressingWarnings: Bool {
        lock.lock(); defer { lock.unlock() }
        return isAnswering || micOpen
    }

    func say(_ text: String, priority: Int = 1) {
        lock.lock()
        if muted {
            lock.unlock()
            return
        }
        if isAnswering, priority == 0 {     // 답변 보호: 경고 무시
            lock.unlock()
            return
        }
        if micOpen {                       // 듣는 중엔 어떤 발화도 넣지 않음
            lock.unlock()
            return
        }
        let interrupt = priority == 0 && synth.isSpeaking && currentPriority > 0
        currentPriority = priority
        if isAnswering, priority == 1 {
            // utterance 추적은 speak 전에 넣을 수 없어 아래에서 처리
        }
        let trackingAnswer = isAnswering && priority == 1
        lock.unlock()

        activatePlaybackSession()
        let u = AVSpeechUtterance(string: text)
        u.voice = AVSpeechSynthesisVoice(language: "en-US")
        u.rate = 0.55
        u.volume = 1.0

        lock.lock()
        if muted || micOpen {
            lock.unlock()
            return
        }
        if trackingAnswer { answerIDs.insert(ObjectIdentifier(u)) }
        lock.unlock()

        if interrupt { synth.stopSpeaking(at: .immediate) }
        synth.speak(u)
    }

    /// Q&A 답변 시작 — 이전 발화를 정리하고 경고 억제 구간 진입.
    func beginAnswer() {
        activatePlaybackSession()
        lock.lock()
        muted = false
        answerIDs.removeAll()
        isAnswering = true
        streamDone = false
        currentPriority = 1
        lock.unlock()
        synth.stopSpeaking(at: .immediate)
    }

    func endAnswerStream() {
        lock.lock()
        streamDone = true
        if answerIDs.isEmpty { isAnswering = false }
        lock.unlock()
    }

    /// Push-to-talk 시작 — 재생 중인 TTS를 즉시 끊고, 이후 잔여 답변 문장도 무시.
    func stop() {
        lock.lock()
        muted = true
        isAnswering = false
        streamDone = true
        answerIDs.removeAll()
        lock.unlock()
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
    }

    /// PTT 마이크 열림/닫힘 — 듣는 동안 경고·안내 발화 억제.
    func setMicOpen(_ open: Bool) {
        lock.lock()
        micOpen = open
        lock.unlock()
    }

    private func utteranceDone(_ u: AVSpeechUtterance) {
        lock.lock()
        answerIDs.remove(ObjectIdentifier(u))
        if streamDone && answerIDs.isEmpty { isAnswering = false }
        lock.unlock()
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer,
                           didFinish u: AVSpeechUtterance) { utteranceDone(u) }
    func speechSynthesizer(_ s: AVSpeechSynthesizer,
                           didCancel u: AVSpeechUtterance) { utteranceDone(u) }

    func activatePlaybackSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {}
    }
}
