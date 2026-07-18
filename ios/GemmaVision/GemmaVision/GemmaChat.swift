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
    private var engine: Engine?

    static let systemPrompt = """
    You are a voice assistant for a blind pedestrian. Look at the CAMERA IMAGE \
    first — that is the primary source of truth for what is ahead. \
    Optional detector_hints JSON may list labels/distances, but it is incomplete \
    and often empty; never answer from hints alone when an image is present. \
    Rules: \
    1. Describe what you SEE in the image to answer the question. \
    2. Use detector_hints only to refine distance/side if they match the image. \
    3. If hints are empty or conflict with the image, trust the image. \
    4. SAFETY FIRST — mention close obstacles ahead. Never invent hazards. \
    5. ANSWER ONLY THE QUESTION in at most 2 short spoken sentences.
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
            ExperimentalFlags.visualTokenBudget = 140

            // 실기기: LLM은 Metal GPU. 비전 어댑터는 모델이 cpu constraint.
            // 시뮬레이터 GPU는 MPSGraph assertion으로 프로세스 사망 → CPU만.
            // iPhone 11 Pro(4GB): 1536 (Gemma4 최소 = prefill1024+window512).
            let backend = Self.llmBackend
            do {
                if backend == .gpu {
                    // assertion 크래시 대비: 성공 전에 플래그를 켜 두고, 성공 시 끈다.
                    UserDefaults.standard.set(true, forKey: Self.forceCPUKey)
                }
                let config = try EngineConfig(
                    modelPath: path,
                    backend: backend,
                    visionBackend: .cpu(),
                    maxNumTokens: 1536,
                    cacheDir: NSTemporaryDirectory())
                let engine = Engine(engineConfig: config)
                try await engine.initialize()
                if backend == .gpu {
                    UserDefaults.standard.set(false, forKey: Self.forceCPUKey)
                }
                self.engine = engine
                state = .ready
                SpeechOut.shared.say(
                    "Assistant ready on \(backend == .gpu ? "GPU" : "CPU").",
                    priority: 1)
            } catch {
                // GPU init이 throw로 실패하면 CPU로 재시도.
                guard backend == .gpu else {
                    state = .failed("\(error)")
                    return
                }
                do {
                    UserDefaults.standard.set(true, forKey: Self.forceCPUKey)
                    let cpu = try EngineConfig(
                        modelPath: path,
                        backend: .cpu(),
                        visionBackend: .cpu(),
                        maxNumTokens: 1536,
                        cacheDir: NSTemporaryDirectory())
                    let engine = Engine(engineConfig: cpu)
                    try await engine.initialize()
                    self.engine = engine
                    state = .ready
                    SpeechOut.shared.say("Assistant ready on CPU.", priority: 1)
                } catch {
                    state = .failed("\(error)")
                }
            }
        }
    }

    private static let forceCPUKey = "gemma.forceCPU"

    /// 실기기에서 GPU를 쓰고 싶은지 (시뮬레이터는 항상 nil).
    private static var preferredGPUBackend: Backend? {
        #if targetEnvironment(simulator)
        return nil
        #else
        return .gpu
        #endif
    }

    /// 이전 GPU 크래시/실패 기록이 있으면 CPU, 아니면 선호 백엔드.
    private static var llmBackend: Backend {
        if preferredGPUBackend == nil { return .cpu() }
        if UserDefaults.standard.bool(forKey: forceCPUKey) { return .cpu() }
        return .gpu
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
        guard state == .ready, let engine else { return }
        state = .busy
        SpeechOut.shared.beginAnswer()   // 답변 구간 진입 — 경고가 답을 끊지 못하게
        Task {
            defer {
                state = .ready
                SpeechOut.shared.endAnswerStream()   // 스트림 끝(발화 완료는 별도)
            }
            do {
                let conversation = try await engine.createConversation()
                // 이미지 → 질문 → hints 순. hints를 "Current scene"으로 부르면
                // 모델이 JSON만 보고 답하는 편향이 생긴다.
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
                    let piece = chunk.toString
                    buffer += piece
                    full += piece
                    while let idx = buffer.firstIndex(where: { ".!?".contains($0) }) {
                        let sentence = String(buffer[...idx])
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        buffer = String(buffer[buffer.index(after: idx)...])
                        if !sentence.isEmpty {
                            SpeechOut.shared.say(sentence, priority: 1)
                        }
                    }
                }
                let tail = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
                if !tail.isEmpty { SpeechOut.shared.say(tail, priority: 1) }
                lastAnswer = full.trimmingCharacters(in: .whitespacesAndNewlines)
            } catch {
                lastAnswer = "error: \(error)"
                SpeechOut.shared.say("Sorry, I couldn't process that.", priority: 1)
            }
        }
    }
}
