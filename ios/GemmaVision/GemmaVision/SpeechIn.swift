// Push-to-talk 음성 입력 — 누르는 동안만 듣고, 떼면 최종 문장을 반환.
import AVFoundation
import Combine
import Speech

@MainActor
final class SpeechIn: ObservableObject {
    enum State: Equatable {
        case idle, requestingAuth, listening, unavailable(String)
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var partial = ""

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var onFinal: ((String) -> Void)?

    var canListen: Bool {
        switch state {
        case .idle, .listening, .unavailable: return true
        case .requestingAuth: return false
        }
    }

    func prepare() {
        guard state == .idle else { return }
        state = .requestingAuth
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            Task { @MainActor in
                guard let self else { return }
                switch status {
                case .authorized:
                    AVAudioApplication.requestRecordPermission { granted in
                        Task { @MainActor in
                            self.state = granted
                                ? .idle
                                : .unavailable("Microphone permission denied")
                        }
                    }
                case .denied, .restricted:
                    self.state = .unavailable("Speech recognition permission denied")
                case .notDetermined:
                    self.state = .unavailable("Speech recognition not determined")
                @unknown default:
                    self.state = .unavailable("Speech recognition unavailable")
                }
            }
        }
    }

    /// 버튼을 누르는 순간 호출. `onFinal`은 손을 뗄 때 확정된 문장.
    func begin(onFinal: @escaping (String) -> Void) {
        if case .unavailable = state { state = .idle }  // 이전 실패 후 재시도
        guard case .idle = state else { return }
        guard let recognizer, recognizer.isAvailable else {
            state = .unavailable("Speech recognizer unavailable")
            return
        }
        self.onFinal = onFinal
        partial = ""

        // TTS가 마이크/세션을 잡고 있으면 인식이 깨지므로 끊는다.
        SpeechOut.shared.stop()

        let session = AVAudioSession.sharedInstance()
        do {
            // `.measurement`는 입력 계측용이라 이후 TTS 출력이 작아진다.
            try session.setCategory(.playAndRecord, mode: .default,
                                    options: [.defaultToSpeaker, .allowBluetooth])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            try session.overrideOutputAudioPort(.speaker)
        } catch {
            state = .unavailable("Audio session failed: \(error.localizedDescription)")
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.taskHint = .dictation
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        self.request = request

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            request.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            input.removeTap(onBus: 0)
            state = .unavailable("Mic start failed: \(error.localizedDescription)")
            return
        }

        state = .listening
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result {
                    self.partial = result.bestTranscription.formattedString
                }
                if error != nil, self.state == .listening {
                    // end()가 이미 stop을 호출한 뒤 오는 종료 에러는 무시
                }
            }
        }
    }

    /// 버튼을 떼는 순간 호출 — 최종 문장을 넘기고 마이크를 닫는다.
    func end() {
        guard state == .listening else { return }
        let text = partial.trimmingCharacters(in: .whitespacesAndNewlines)
        let finish = onFinal
        teardownAudio()
        state = .idle
        onFinal = nil
        partial = ""
        if !text.isEmpty {
            finish?(text)
        }
    }

    func cancel() {
        guard state == .listening else { return }
        teardownAudio()
        state = .idle
        onFinal = nil
        partial = ""
    }

    private func teardownAudio() {
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        // 마이크 세션을 닫고 TTS용 playback으로 돌려 볼륨을 복구한다.
        let session = AVAudioSession.sharedInstance()
        try? session.setActive(false, options: .notifyOthersOnDeactivation)
        SpeechOut.shared.activatePlaybackSession()
    }
}
