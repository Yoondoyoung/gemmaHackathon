// Gemma 4 E2B 온디바이스 Q&A — LiteRT-LM Swift API.
// Mac판과 동일 원칙: LLM에는 검증된 SceneState 스냅샷만 (프레임 직접 투입 금지),
// 문장 단위 스트리밍 발화, 경고 경로에는 관여하지 않음.
// SPM: https://github.com/google-ai-edge/LiteRT-LM (0.12.0+)
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

    // Mac판 system_prompt.txt 축소 이식 (우선순위 구조 유지)
    static let systemPrompt = """
    You are a voice assistant for a blind pedestrian. You receive a JSON snapshot \
    of what the camera sees: objects (position left/center/right, distance \
    near/medium/far) and sign text (with position and age in seconds). \
    Answer using ONLY the snapshot, in this priority order: \
    1. SAFETY FIRST — if any object is center and near, mention it and never say \
    the path is clear. \
    2. ANSWER ONLY THE QUESTION — no other objects or signs. \
    3. BE BRIEF — maximum 2 short spoken sentences. \
    Positions are exact: left = on your left, center = directly ahead, right = on \
    your right. If the answer is not in the snapshot, say you don't see it — \
    never substitute something else.
    """

    func load() {
        guard state == .idle else { return }
        state = .loading
        Task {
            do {
                guard let path = Self.modelPath() else {
                    state = .failed("모델(.litertlm) 없음 — SETUP.md 참고")
                    return
                }
                let config = try EngineConfig(
                    modelPath: path,
                    backend: .gpu,          // 구형 기기에서 메모리 부족 시 .cpu()
                    maxNumTokens: 1024,
                    cacheDir: NSTemporaryDirectory())
                let engine = Engine(engineConfig: config)
                try await engine.initialize()
                self.engine = engine
                state = .ready
                SpeechOut.shared.say("Assistant ready.", priority: 1)
            } catch {
                state = .failed("\(error)")
            }
        }
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

    func ask(_ question: String, scene: String) {
        guard state == .ready, let engine else { return }
        state = .busy
        Task {
            defer { state = .ready }
            do {
                // 매 질문 새 대화 (짧은 Q&A라 히스토리보다 프리필 절약 우선)
                let conversation = try await engine.createConversation()
                let prompt = "\(Self.systemPrompt)\n\nCurrent scene:\n\(scene)\n\n" +
                             "User question: \(question)"
                var buffer = "", full = ""
                for try await chunk in conversation.sendMessageStream(Message(prompt)) {
                    let piece = chunk.toString
                    buffer += piece
                    full += piece
                    // 문장 경계마다 즉시 발화 (Mac판 ask_streaming 이식)
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
                SpeechOut.shared.say("Sorry, I couldn't process that.", priority: 1)
            }
        }
    }
}
