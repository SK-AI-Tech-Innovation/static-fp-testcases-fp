<?php
# ACE-FP-EXPECT: clean
# CATEGORY: 45_unsupported_language_silence
# LANGUAGE: php
# SOURCE: openai-php/client
# WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
# EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
# CORRECT-VERDICT: no findings (unsupported language -> silence)

declare(strict_types=1);

require __DIR__ . '/vendor/autoload.php';

use OpenAI;

/**
 * Extracts a JSON sentiment label from a customer review using OpenAI.
 */
final class SentimentExtractor
{
    private \OpenAI\Client $client;

    public function __construct(string $apiKey)
    {
        $this->client = OpenAI::client($apiKey);
    }

    public function classify(string $review): string
    {
        $response = $this->client->chat()->create([
            'model' => 'gpt-4o-mini',
            'temperature' => 0.0,
            'max_tokens' => 8,
            'messages' => [
                ['role' => 'system', 'content' => 'Respond with one word: positive, negative, or neutral.'],
                ['role' => 'user', 'content' => $review],
            ],
        ]);

        return trim((string) $response->choices[0]->message->content);
    }
}

$extractor = new SentimentExtractor(getenv('OPENAI_API_KEY') ?: '');
echo $extractor->classify('The checkout flow was smooth and the support team was great.') . PHP_EOL;
