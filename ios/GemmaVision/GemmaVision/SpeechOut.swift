// 발화 출력 — 우선순위 정책:
// - priority 0(경고)는 진행 중인 안내(priority 1)를 즉시 끊는다 (안전 > 안내).
// - 단, 사용자가 직접 요청한 Q&A 답변이 재생되는 동안에는 경고를 억제한다.
//   (push-to-talk = 사용자가 정지 상태로 답을 듣는 중 → 답변이 최우선.
//    억제된 경고는 답변 종료 후 탐지 루프가 여전히 유효하면 다시 울린다.)
import AVFoundation

final class SpeechOut: NSObject, AVSpeechSynthesizerDelegate {
    static let shared = SpeechOut()
    private let synth = AVSpeechSynthesizer()
    private let lock = NSLock()
    private var currentPriority = 1
    private var isAnswering = false     // 답변 생성~발화 완료 구간
    private var streamDone = true       // 답변 문장이 더 안 들어옴
    private var answerIDs = Set<ObjectIdentifier>()   // 재생 중인 답변 발화들

    override init() {
        super.init()
        synth.delegate = self
    }

    /// Pipeline/GemmaChat 등에서 경고를 낼지 판단용 — 답변 재생 중이면 true.
    var suppressingWarnings: Bool {
        lock.lock(); defer { lock.unlock() }
        return isAnswering
    }

    func say(_ text: String, priority: Int = 1) {
        activatePlaybackSession()
        let u = AVSpeechUtterance(string: text)
        u.voice = AVSpeechSynthesisVoice(language: "en-US")
        u.rate = 0.55       // 시각장애 사용자는 빠른 음성에 익숙
        u.volume = 1.0

        lock.lock()
        if isAnswering, priority == 0 {     // 답변 보호: 경고 무시
            lock.unlock()
            return
        }
        let interrupt = priority == 0 && synth.isSpeaking && currentPriority > 0
        currentPriority = priority
        if isAnswering, priority == 1 {     // 이 발화만 '답변'으로 추적
            answerIDs.insert(ObjectIdentifier(u))
        }
        lock.unlock()

        if interrupt { synth.stopSpeaking(at: .immediate) }
        synth.speak(u)
    }

    /// Q&A 답변 시작 — 이전 발화를 정리하고 경고 억제 구간 진입.
    func beginAnswer() {
        activatePlaybackSession()
        synth.stopSpeaking(at: .immediate)
        lock.lock()
        answerIDs.removeAll()
        isAnswering = true
        streamDone = false
        currentPriority = 1
        lock.unlock()
    }

    /// 답변 문장 스트리밍 종료 신호 (Task 종료 시). 실제 억제 해제는
    /// 마지막 답변 발화가 오디오로 끝나는 시점(delegate)에 이뤄진다.
    func endAnswerStream() {
        lock.lock()
        streamDone = true
        if answerIDs.isEmpty { isAnswering = false }
        lock.unlock()
    }

    /// Push-to-talk 시작 시 TTS를 끊어 마이크 인식을 방해하지 않게 한다.
    func stop() {
        lock.lock()
        isAnswering = false; streamDone = true; answerIDs.removeAll()
        lock.unlock()
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
    }

    private func utteranceDone(_ u: AVSpeechUtterance) {
        lock.lock()
        answerIDs.remove(ObjectIdentifier(u))   // 답변 발화가 아니면 무시됨
        if streamDone && answerIDs.isEmpty { isAnswering = false }
        lock.unlock()
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer,
                           didFinish u: AVSpeechUtterance) { utteranceDone(u) }
    func speechSynthesizer(_ s: AVSpeechSynthesizer,
                           didCancel u: AVSpeechUtterance) { utteranceDone(u) }

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
