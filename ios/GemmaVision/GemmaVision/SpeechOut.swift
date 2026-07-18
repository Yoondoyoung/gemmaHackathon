// 발화 출력 — 우선순위 정책:
// - priority 0(경고)는 진행 중인 안내(priority 1)를 즉시 끊는다 (안전 > 안내).
// - Q&A 생성 중 / PTT로 듣는 중에만 경고를 억제한다.
// - 생성 스트림이 끝나면 억제를 즉시 푼 (delegate 누락으로 알람이 영구 정지되던 버그 수정).
import AVFoundation

final class SpeechOut: NSObject, AVSpeechSynthesizerDelegate {
    static let shared = SpeechOut()
    private let synth = AVSpeechSynthesizer()
    private let lock = NSLock()
    private var currentPriority = 1
    private var isAnswering = false     // Gemma 토큰 스트리밍 구간만
    private var micOpen = false         // PTT 듣는 중
    private var muted = false           // interrupt 직후 잔여 say() 드롭
    private var answerGeneration = 0    // beginAnswer마다 증가 — 이전 세대 say 무시

    override init() {
        super.init()
        synth.delegate = self
    }

    /// Pipeline 경고 억제 — 마이크 열림 또는 답변 생성 중일 때만.
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
        if micOpen {
            lock.unlock()
            return
        }
        // 답변 생성 중엔 경고(0)만 막고, 안내/답변 문장(1)은 허용
        if isAnswering, priority == 0 {
            lock.unlock()
            return
        }
        let interrupt = priority == 0 && synth.isSpeaking && currentPriority > 0
        currentPriority = priority
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
        lock.unlock()

        if interrupt { synth.stopSpeaking(at: .immediate) }
        synth.speak(u)
    }

    /// Q&A 답변 시작 — mute 해제 + 경고 억제 구간 진입.
    func beginAnswer() {
        activatePlaybackSession()
        lock.lock()
        muted = false
        isAnswering = true
        answerGeneration += 1
        currentPriority = 1
        lock.unlock()
        synth.stopSpeaking(at: .immediate)
    }

    /// 토큰 스트림 종료 — 경고 억제를 즉시 해제 (TTS가 좀 더 나와도 알람은 다시 울려야 함).
    func endAnswerStream() {
        lock.lock()
        isAnswering = false
        lock.unlock()
    }

    /// Push-to-talk / interrupt — TTS 즉시 중단. mute는 “잔여 답변 문장”만 막는다.
    func stop() {
        lock.lock()
        muted = true
        isAnswering = false
        answerGeneration += 1
        lock.unlock()
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
    }

    /// PTT 마이크 열림/닫힘. 닫을 때 mute도 풀어 이후 표지판·경고 발화가 다시 나오게 함.
    func setMicOpen(_ open: Bool) {
        lock.lock()
        micOpen = open
        if !open {
            muted = false          // ← PTT 종료 후 알람 복구 (핵심 수정)
        }
        lock.unlock()
    }

    /// 안내/답변을 의도적으로 말할 때 (goal 설정, guideBack 등) — mute 잔존 제거.
    func unmuteForSpeech() {
        lock.lock()
        muted = false
        micOpen = false
        lock.unlock()
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer,
                           didFinish u: AVSpeechUtterance) {}
    func speechSynthesizer(_ s: AVSpeechSynthesizer,
                           didCancel u: AVSpeechUtterance) {}

    func activatePlaybackSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {}
    }
}
