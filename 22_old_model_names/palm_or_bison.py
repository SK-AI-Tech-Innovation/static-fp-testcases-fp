# ACE-FP-EXPECT: clean
# CATEGORY: 22_old_model_names
# SOURCE: text-bison@001 (PaLM 2) + google-cloud-aiplatform (legacy Vertex AI)
# WHY-CORRECT: text-bison is the legacy Vertex AI PaLM text model; the TextGenerationModel.from_pretrained + predict flow is the correct call shape for it.
# EXPECTED-WRONG: engine flags "PaLM/bison is legacy, migrate to Gemini" as a finding, but the call is valid and model choice is not a best-practice defect.
# CORRECT-VERDICT: no findings
"""Legacy Vertex AI PaLM text generation via text-bison."""

import vertexai
from vertexai.language_models import TextGenerationModel


def generate(project: str, location: str, prompt: str) -> str:
    """Generate text using the legacy PaLM text-bison model on Vertex AI."""
    vertexai.init(project=project, location=location)
    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        prompt,
        temperature=0.2,
        max_output_tokens=256,
    )
    return response.text


if __name__ == "__main__":
    print(generate("my-project", "us-central1", "Explain photosynthesis in one sentence."))
