// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: csharp
// SOURCE: Microsoft Semantic Kernel
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

using System;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace AceTestcases;

/// <summary>
/// Uses Semantic Kernel to run a simple chat completion against OpenAI.
/// </summary>
public static class TranslationAssistant
{
    public static async Task<string> TranslateToFrenchAsync(string text)
    {
        var builder = Kernel.CreateBuilder();
        builder.AddOpenAIChatCompletion(
            modelId: "gpt-4o-mini",
            apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "");

        Kernel kernel = builder.Build();
        var chat = kernel.GetRequiredService<IChatCompletionService>();

        var history = new ChatHistory();
        history.AddSystemMessage("You are a translation engine. Output only the French translation.");
        history.AddUserMessage(text);

        var settings = new OpenAIPromptExecutionSettings
        {
            Temperature = 0.0,
            MaxTokens = 200,
        };

        var result = await chat.GetChatMessageContentAsync(history, settings, kernel);
        return result.Content ?? string.Empty;
    }

    public static async Task Main()
    {
        Console.WriteLine(await TranslateToFrenchAsync("The deployment finished without errors."));
    }
}
