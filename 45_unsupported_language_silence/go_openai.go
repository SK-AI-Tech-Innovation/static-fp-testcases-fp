// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: go
// SOURCE: github.com/sashabaranov/go-openai
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

// streamChat streams a completion token-by-token to stdout.
func streamChat(ctx context.Context, client *openai.Client, prompt string) error {
	req := openai.ChatCompletionRequest{
		Model:     openai.GPT4oMini,
		MaxTokens: 512,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: "You are a concise assistant."},
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
		Stream: true,
	}

	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return fmt.Errorf("create stream: %w", err)
	}
	defer stream.Close()

	for {
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("stream recv: %w", err)
		}
		if len(resp.Choices) > 0 {
			fmt.Print(resp.Choices[0].Delta.Content)
		}
	}
}

func main() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	if err := streamChat(context.Background(), client, "Explain goroutines in one paragraph."); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Println()
}
