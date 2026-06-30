// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: kotlin
// SOURCE: openai-kotlin (com.aallam.openai)
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

import com.aallam.openai.api.chat.ChatCompletionRequest
import com.aallam.openai.api.chat.ChatMessage
import com.aallam.openai.api.chat.ChatRole
import com.aallam.openai.api.model.ModelId
import com.aallam.openai.client.OpenAI
import kotlinx.coroutines.runBlocking

/**
 * Rewrites a rough draft into a polished, professional tone using openai-kotlin.
 */
class ToneRewriter(apiKey: String) {
    private val openAI = OpenAI(token = apiKey)

    suspend fun polish(draft: String): String {
        val request = ChatCompletionRequest(
            model = ModelId("gpt-4o-mini"),
            temperature = 0.5,
            maxTokens = 400,
            messages = listOf(
                ChatMessage(
                    role = ChatRole.System,
                    content = "Rewrite the user's text in a professional, friendly tone."
                ),
                ChatMessage(role = ChatRole.User, content = draft),
            ),
        )

        val completion = openAI.chatCompletion(request)
        return completion.choices.firstOrNull()?.message?.content?.trim().orEmpty()
    }
}

fun main() = runBlocking {
    val rewriter = ToneRewriter(System.getenv("OPENAI_API_KEY") ?: "")
    val polished = rewriter.polish("hey just wanted 2 say the build is broken again, pls fix asap")
    println(polished)
}
