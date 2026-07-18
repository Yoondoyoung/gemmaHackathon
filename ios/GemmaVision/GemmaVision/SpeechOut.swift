// 발화 출력 — 우선순위 정책:
// - 평소: priority 0(경고)가 안내(priority 1)를 끊을 수 있음 (안전).
// - Gemma 답변 중(생성+TTS 재생): 경고 완전 억제 — 답을 끊지 않음.
// - PTT 듣는 중에도 경고 억제. interrupt/stop으로 즉시 해제.
import AVFoundation

final class SpeechOut: NSObject, AVSpeechSynthesizerDelegate {
    static let shared = SpeechOut()
    private let synth = AVSpeechSynthesizer()
    private let lock = NSLock()
    private var currentPriority = 1
    private var isAnswering = false
    private var streamDone = true
    private var answerIDs = Set<ObjectIdentifier>()
    private var micOpen = false
    private var muted = false
    private var answerGeneration = 0
    private var answerStartedAt = Date.distantPast
    /// delegate 누락 대비 — 이 시간이 지나면 경고 억제 강제 해제
    private static let answerMaxSec: TimeInterval = 45

    override init() {
        super.init()
        synth.delegate = self
    }

    /// Pipeline 경고 억제 — Gemma 답변(생성·발화) 또는 PTT 중.
    var suppressingWarnings: Bool {
        lock.lock()
        refreshAnswerTimeoutLocked()
        let on = isAnswering || micOpen
        lock.unlock()
        return on
    }

    func say(_ text: String, priority: Int = 1) {
        lock.lock()
        refreshAnswerTimeoutLocked()
        if muted || micOpen {
            lock.unlock()
            return
        }
        // 답변 중엔 p0 경고 문장 자체를 큐에 넣지 않음
        if isAnswering, priority == 0 {
            lock.unlock()
            return
        }
        // 답변 TTS 재생 중에는 p0가 끊지 못함
        let interrupt = priority == 0 && synth.isSpeaking && currentPriority > 0 && !isAnswering
        currentPriority = priority
        let trackAsAnswer = isAnswering && priority == 1
        lock.unlock()

        activatePlaybackSession()
        let u = AVSpeechUtterance(string: text)
        u.voice = AVSpeechSynthesisVoice(language: "en-US")
        u.rate = 0.55
        u.volume = 1.0

        lock.lock()
        if muted || micOpen || (isAnswering && priority == 0) {
            lock.unlock()
            return
        }
        if trackAsAnswer {
            answerIDs.insert(ObjectIdentifier(u))
        }
        lock.unlock()

        if interrupt { synth.stopSpeaking(at: .immediate) }
        synth.speak(u)
    }

    /// Q&A 답변 시작 — 경고 억제 구간 진입 (TTS 끝날 때까지 유지).
    func beginAnswer() {
        activatePlaybackSession()
        lock.lock()
        muted = false
        isAnswering = true
        streamDone = false
        answerIDs.removeAll()
        answerGeneration += 1
        answerStartedAt = Date()
        currentPriority = 1
        lock.unlock()
        synth.stopSpeaking(at: .immediate)
    }

    /// 토큰 스트림 종료. 아직 재생 중인 답변 문장이 있으면 억제 유지.
    func endAnswerStream() {
        lock.lock()
        streamDone = true
        if answerIDs.isEmpty {
            isAnswering = false
        }
        lock.unlock()
    }

    /// PTT / interrupt — TTS·억제 즉시 해제.
    func stop() {
        lock.lock()
        muted = true
        isAnswering = false
        streamDone = true
        answerIDs.removeAll()
        answerGeneration += 1
        lock.unlock()
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
    }

    func setMicOpen(_ open: Bool) {
        lock.lock()
        micOpen = open
        if !open { muted = false }
        lock.unlock()
    }

    func unmuteForSpeech() {
        lock.lock()
        muted = false
        micOpen = false
        lock.unlock()
    }

    private func utteranceDone(_ u: AVSpeechUtterance) {
        lock.lock()
        answerIDs.remove(ObjectIdentifier(u))
        if streamDone && answerIDs.isEmpty {
            isAnswering = false
        }
        lock.unlock()
    }

    private func refreshAnswerTimeoutLocked() {
        guard isAnswering else { return }
        if Date().timeIntervalSince(answerStartedAt) > Self.answerMaxSec {
            isAnswering = false
            streamDone = true
            answerIDs.removeAll()
        }
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
