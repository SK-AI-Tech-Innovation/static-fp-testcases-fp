# ACE-FP-EXPECT: clean
# CATEGORY: 45_unsupported_language_silence
# LANGUAGE: ruby
# SOURCE: ruby-openai (OpenAI::Client)
# WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
# EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
# CORRECT-VERDICT: no findings (unsupported language -> silence)

require "openai"

# Generates short, friendly release notes from a list of merged PR titles.
class ReleaseNotesGenerator
  def initialize(api_key: ENV.fetch("OPENAI_API_KEY"))
    @client = OpenAI::Client.new(access_token: api_key)
  end

  def generate(pr_titles)
    response = @client.chat(
      parameters: {
        model: "gpt-4o-mini",
        temperature: 0.4,
        max_tokens: 300,
        messages: [
          { role: "system", content: "Write concise, upbeat release notes as bullet points." },
          { role: "user", content: pr_titles.join("\n") }
        ]
      }
    )

    response.dig("choices", 0, "message", "content").to_s.strip
  end
end

if __FILE__ == $PROGRAM_NAME
  titles = [
    "Add dark mode toggle",
    "Fix race condition in upload queue",
    "Improve cold-start latency by 30%"
  ]
  puts ReleaseNotesGenerator.new.generate(titles)
end
