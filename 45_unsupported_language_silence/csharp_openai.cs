// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: csharp
// SOURCE: OpenAI (official .NET nuget)
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

using System;
using System.Threading.Tasks;
using OpenAI.Chat;

namespace AceTestcases;

/// <summary>
/// Minimal wrapper around the official OpenAI .NET chat client.
/// </summary>
public sealed class ProductDescriber
{
    private readonly ChatClient _client;

    public ProductDescriber(string apiKey)
    {
        _client = new ChatClient(model: "gpt-4o-mini", apiKey: apiKey);
    }

    /// <summary>Generates a one-line marketing blurb for a product name.</summary>
    public async Task<string> DescribeAsync(string productName)
    {
        var messages = new ChatMessage[]
        {
            new SystemChatMessage("Write a single punchy marketing sentence."),
            new UserChatMessage($"Product: {productName}"),
        };

        var options = new ChatCompletionOptions
        {
            Temperature = 0.7f,
            MaxOutputTokenCount = 64,
        };

        ChatCompletion completion = await _client.CompleteChatAsync(messages, options);
        return completion.Content[0].Text.Trim();
    }

    public static async Task Main()
    {
        var describer = new ProductDescriber(
            Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "");
        Console.WriteLine(await describer.DescribeAsync("Noise-cancelling travel mug"));
    }
}
