// Gemma 4 E2B 온디바이스 Q&A — LiteRT-LM Swift API.
// 경고 경로는 룰베이스 유지. Q&A에는 SceneState JSON + 현재 프레임(비전)을 함께 투입.
// SPM: Vendor/LiteRT-LM (0.13.1)
import Combine
import Foundation
import LiteRTLM

@MainActor
final class GemmaChat: ObservableObject {
    enum State: Equatable {
        case idle, loading, ready, busy
        case failed(String)
    }

    @Published var state: State = .idle
    @Published var lastAnswer = ""
    /// "GPU" | "CPU" | "" (로딩 전) — 오버레이 표시용
    @Published var backendName = ""
    /// CPU 폴백 시 이유 (오버레이)
    @Published var backendNote = ""
    private var engine: Engine?
    private var askTask: Task<Void, Never>?

    /// 맥 `python tools/prompt_log_server.py` 가 출력한 IP로 맞출 것 (같은 Wi‑Fi).
    /// 예: "10.220.8.129"  — 비우면 맥 전송 생략, Xcode 콘솔 print만.
    private static let macLogHost = "10.220.8.129"
    private static let macLogPort = 8765

    static let systemPrompt = """
    You are a calm walking companion for a blind pedestrian. Speak like a helpful \
    friend beside them — plain spoken English, not a report or a computer log. \
    Look at the CAMERA IMAGE first — that is the primary source of truth for what \
    is ahead.     Optional detector_hints JSON may list object labels/distances and \
    ARKit structures (wall/door/window/floor/fork with depth_m). path_clear=true \
    means open floor ahead. fork means a side opening or split (pos left/right/both). \
    Hints may be incomplete; never answer from hints alone when an image is present. \
    Rules: \
    1. Describe what you SEE in the image to answer the question. \
    2. Use detector_hints only to refine distance/side if they match the image. \
    3. If hints are empty or conflict with the image, trust the image. \
    4. SAFETY FIRST — mention close obstacles ahead. Never invent hazards. \
    5. ANSWER ONLY THE QUESTION in at most 2 short spoken sentences. \
    6. Sound human: no "I recall", "in the image", "center of the frame", \
    "detector", "JSON", or exact second counts. Say left / ahead / right. \
    7. For PAST questions ("did I pass...", "earlier"), use recent_history \
    (what, pos, age_sec). Round time loosely: under ~15s → "just now" or \
    "a moment ago"; under ~60s → "about half a minute ago"; else "a minute ago" \
    / "a couple minutes ago". Good: "Yeah, your backpack was just ahead a moment \
    ago." / "Yes — we passed it on your left just now." Bad: "Yes, I recall \
    seeing a backpack about 7 seconds ago in the center of the image." If \
    recent_history is empty or lacks it, say you don't remember seeing it.
    """

    func load() {
        guard state == .idle else { return }
        state = .loading
        Task {
            guard let path = Self.modelPath() else {
                state = .failed("모델(.litertlm) 없음 — SETUP.md 참고")
                return
            }
            // Gemma4는 이미지를 max_num_patches에 맞춰 업스케일한다.
            // budget 미설정 → vision_280(~2400 patches, CPU 인코더 ~10초).
            // 70/140/280 중 택1. 11 Pro에선 140이 속도와 품질 타협점.
            ExperimentalFlags.optIntoExperimentalAPIs()
            // 70이 가장 가볍고, 140은 품질↑·메모리/지연↑. 16 Pro 체감 여유를 위해 70.
            ExperimentalFlags.visualTokenBudget = 70

            // 실기기: LLM은 Metal GPU. 비전 어댑터는 모델이 cpu constraint.
            // 시뮬레이터는 GPU 불가. ARKit보다 먼저 init해야 Metal 경합으로 GPU 실패가 줄어듦.
            // iPhone 11 Pro(4GB): 1536 (Gemma4 최소 = prefill1024+window512).
            UserDefaults.standard.set(false, forKey: Self.forceCPUKey)  // 과거 sticky CPU 해제
            backendNote = ""
            #if targetEnvironment(simulator)
            let tryGPU = false
            var gpuFail = "simulator"
            #else
            let tryGPU = true
            var gpuFail = ""
            #endif

            if tryGPU {
                do {
                    let config = try EngineConfig(
                        modelPath: path,
                        backend: .gpu,
                        visionBackend: .cpu(),
                        maxNumTokens: 1536,
                        cacheDir: Self.cacheDirectory())
                    let engine = Engine(engineConfig: config)
                    try await engine.initialize()
                    self.engine = engine
                    backendName = "GPU"
                    backendNote = ""
                    state = .ready
                    Self.postPromptLogToMac(
                        question: "(app ready — log link ok)",
                        image: "none",
                        scene: "{}")
                    SpeechOut.shared.say("Assistant ready on GPU.", priority: 1)
                    return
                } catch {
                    gpuFail = String(describing: error)
                    print("[gemma] GPU init failed, falling back to CPU: \(error)")
                }
            }

            do {
                let cpu = try EngineConfig(
                    modelPath: path,
                    backend: .cpu(),
                    visionBackend: .cpu(),
                    maxNumTokens: 1536,
                    cacheDir: Self.cacheDirectory())
                let engine = Engine(engineConfig: cpu)
                try await engine.initialize()
                self.engine = engine
                backendName = "CPU"
                backendNote = gpuFail.isEmpty ? "" : gpuFail
                state = .ready
                Self.postPromptLogToMac(
                    question: "(app ready — log link ok)",
                    image: "none",
                    scene: "{}")
                SpeechOut.shared.say("Assistant ready on CPU.", priority: 1)
            } catch {
                state = .failed("\(error)")
            }
        }
    }

    private static let forceCPUKey = "gemma.forceCPU"

    private static func cacheDirectory() -> String {
        let url = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("litertlm", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url.path
    }

    /// 번들 → Documents 순으로 .litertlm 탐색
    static func modelPath() -> String? {
        if let bundled = Bundle.main.paths(forResourcesOfType: "litertlm",
                                           inDirectory: nil).first {
            return bundled
        }
        let docs = FileManager.default.urls(for: .documentDirectory,
                                            in: .userDomainMask)[0]
        let files = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: nil)) ?? []
        return files.first { $0.pathExtension == "litertlm" }?.path
    }

    func ask(_ question: String, scene: String, imageJPEG: Data? = nil) {
        let hasImage = imageJPEG.map { !$0.isEmpty } ?? false
        let imageKB = hasImage ? (imageJPEG!.count + 512) / 1024 : 0
        let imageDesc = hasImage ? "yes \(imageKB)KB JPEG" : "none"

        guard state == .ready || state == .busy, let engine else {
            print("[gemma] ask skipped — state=\(state)")
            Self.postPromptLogToMac(
                question: "SKIPPED (\(String(describing: state))): \(question)",
                image: imageDesc,
                scene: scene)
            return
        }
        interrupt()   // 이전 답변·TTS가 있으면 끊고 새 질문
        state = .busy
        SpeechOut.shared.beginAnswer()
        print("""
        [gemma →] Q: \(question)
        image: \(imageDesc)
        detector_hints: \(scene)
        """)
        askTask = Task { [weak self] in
            guard let self else { return }
            defer {
                // 취소돼도 경고 억제는 반드시 해제 (영구 무음 버그 방지)
                SpeechOut.shared.endAnswerStream()
                if !Task.isCancelled {
                    self.state = .ready
                    print("[gemma] ready again")
                }
            }
            do {
                let conversation = try await engine.createConversation()
                var parts: [Content] = []
                if let imageJPEG, !imageJPEG.isEmpty {
                    parts.append(.imageData(imageJPEG))
                }
                parts.append(.text("""
                \(Self.systemPrompt)

                Question: \(question)

                detector_hints (secondary, may be empty): \(scene)
                """))
                let message = Message(contents: parts)
                var buffer = "", full = ""
                for try await chunk in conversation.sendMessageStream(message) {
                    try Task.checkCancellation()
                    let piece = chunk.toString
                    buffer += piece
                    full += piece
                    while let idx = buffer.firstIndex(where: { ".!?".contains($0) }) {
                        try Task.checkCancellation()
                        let sentence = String(buffer[...idx])
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        buffer = String(buffer[buffer.index(after: idx)...])
                        if !sentence.isEmpty {
                            SpeechOut.shared.say(sentence, priority: 1)
                        }
                    }
                }
                try Task.checkCancellation()
                let tail = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
                if !tail.isEmpty { SpeechOut.shared.say(tail, priority: 1) }
                lastAnswer = full.trimmingCharacters(in: .whitespacesAndNewlines)
            } catch is CancellationError {
                print("[gemma] interrupted")
            } catch {
                guard !Task.isCancelled else { return }
                lastAnswer = "error: \(error)"
                SpeechOut.shared.say("Sorry, I couldn't process that.", priority: 1)
            }
        }
    }

    /// PTT 등으로 현재 생성·발화를 즉시 중단하고 ready로 돌린다.
    func interrupt() {
        askTask?.cancel()
        askTask = nil
        SpeechOut.shared.stop()
        if state == .busy { state = .ready }
    }

    /// 맥 터미널 로그 서버로 fire-and-forget POST (실패해도 Q&A는 계속).
    static func postPromptLogToMac(question: String, image: String, scene: String) {
        let hosts = [
            macLogHost,
            "jeonghowon-ui-MacBookAir.local",
        ].map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
         .filter { !$0.isEmpty }
        let body: [String: String] = [
            "question": question,
            "image": image,
            "detector_hints": scene,
        ]
        guard let httpBody = try? JSONSerialization.data(withJSONObject: body) else { return }

        for host in hosts {
            guard let url = URL(string: "http://\(host):\(macLogPort)/log") else { continue }
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.setValue("close", forHTTPHeaderField: "Connection")
            req.timeoutInterval = 3
            req.httpBody = httpBody
            URLSession.shared.dataTask(with: req) { _, response, error in
                if let error {
                    print("[gemma-log] \(host) failed: \(error.localizedDescription)")
                } else if let http = response as? HTTPURLResponse {
                    print("[gemma-log] \(host) → \(http.statusCode)")
                }
            }.resume()
        }
    }
}
