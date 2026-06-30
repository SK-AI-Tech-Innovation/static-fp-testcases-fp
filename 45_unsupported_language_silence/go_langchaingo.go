// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: go
// SOURCE: github.com/tmc/langchaingo
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

// summarizeIncident sends an incident report to an LLM and returns a short summary.
func summarizeIncident(ctx context.Context, report string) (string, error) {
	llm, err := openai.New(
		openai.WithModel("gpt-4o-mini"),
		openai.WithToken("OPENAI_API_KEY_FROM_ENV"),
	)
	if err != nil {
		return "", fmt.Errorf("init llm: %w", err)
	}

	messages := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem,
			"You are an SRE assistant. Summarize incidents in two sentences."),
		llms.TextParts(llms.ChatMessageTypeHuman, report),
	}

	resp, err := llm.GenerateContent(ctx, messages,
		llms.WithTemperature(0.2),
		llms.WithMaxTokens(256),
	)
	if err != nil {
		return "", fmt.Errorf("generate: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices returned")
	}
	return resp.Choices[0].Content, nil
}

func main() {
	ctx := context.Background()
	summary, err := summarizeIncident(ctx,
		"At 03:14 UTC the payment service returned 503s for 12 minutes due to a DB pool exhaustion.")
	if err != nil {
		log.Fatalf("summarize failed: %v", err)
	}
	fmt.Println(summary)
}
