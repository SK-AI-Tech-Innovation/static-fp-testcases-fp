# ACE-FP-EXPECT: clean
# CATEGORY: 42_provider_sdks
# SOURCE: Hugging Face transformers multimodal (Llama 4), verified June 2026
# WHY-CORRECT: multimodal Llama 4 inference uses AutoProcessor + AutoModelForImageTextToText; the processor applies the chat template and the model.generate output is decoded with processor.batch_decode
# EXPECTED-WRONG: stale analyzer expects an OpenAI-style client.chat.completions.create and flags the local processor/generate/batch_decode pipeline as "no .choices / malformed response access"
# CORRECT-VERDICT: no findings
"""Run local Llama 4 image-text inference with transformers AutoProcessor."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://example.com/cat.jpg"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=128)
    text = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )[0]
    print(text)


if __name__ == "__main__":
    main()
