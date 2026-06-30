// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: swift
// SOURCE: MacPaw/OpenAI swift package
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

import Foundation
import OpenAI

/// Suggests a concise commit message from a unified diff using MacPaw/OpenAI.
struct CommitMessageSuggester {
    private let client: OpenAI

    init(apiKey: String) {
        self.client = OpenAI(apiToken: apiKey)
    }

    func suggest(diff: String) async throws -> String {
        let query = ChatQuery(
            messages: [
                .system(.init(content: "Write a one-line imperative git commit message.")),
                .user(.init(content: .string(diff))),
            ],
            model: .gpt4_o_mini,
            maxTokens: 60,
            temperature: 0.3
        )

        let result = try await client.chats(query: query)
        let content = result.choices.first?.message.content ?? ""
        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

@main
struct Demo {
    static func main() async {
        let apiKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"] ?? ""
        let suggester = CommitMessageSuggester(apiKey: apiKey)
        do {
            let message = try await suggester.suggest(
                diff: "diff --git a/app.swift ... +    print(\"hello\")")
            print(message)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        }
    }
}
