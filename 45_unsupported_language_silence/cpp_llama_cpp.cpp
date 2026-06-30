// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: cpp
// SOURCE: llama.cpp (llama.h) local inference
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

#include "llama.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// RAII helper around a llama.cpp model + context for local inference.
class LlamaSession {
public:
    LlamaSession(const std::string &model_path, int n_ctx) {
        llama_backend_init();

        auto model_params = llama_model_default_params();
        model_ = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model_) {
            throw std::runtime_error("failed to load model: " + model_path);
        }

        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_ = llama_init_from_model(model_, ctx_params);
        if (!ctx_) {
            llama_model_free(model_);
            throw std::runtime_error("failed to create context");
        }

        vocab_ = llama_model_get_vocab(model_);
        sampler_ = llama_sampler_init_greedy();
    }

    ~LlamaSession() {
        llama_sampler_free(sampler_);
        llama_free(ctx_);
        llama_model_free(model_);
        llama_backend_free();
    }

    std::string generate(const std::string &prompt, int n_predict) {
        const int n_prompt = -llama_tokenize(
            vocab_, prompt.c_str(), static_cast<int>(prompt.size()),
            nullptr, 0, true, true);

        std::vector<llama_token> tokens(n_prompt);
        if (llama_tokenize(vocab_, prompt.c_str(),
                           static_cast<int>(prompt.size()),
                           tokens.data(), n_prompt, true, true) < 0) {
            throw std::runtime_error("tokenization failed");
        }

        llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
        std::string out;

        for (int i = 0; i < n_predict; ++i) {
            if (llama_decode(ctx_, batch) != 0) {
                throw std::runtime_error("decode failed");
            }

            llama_token next = llama_sampler_sample(sampler_, ctx_, -1);
            if (llama_vocab_is_eog(vocab_, next)) {
                break;
            }

            char buf[256];
            int n = llama_token_to_piece(vocab_, next, buf, sizeof(buf), 0, true);
            if (n > 0) {
                out.append(buf, static_cast<size_t>(n));
            }

            batch = llama_batch_get_one(&next, 1);
        }

        return out;
    }

private:
    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    const llama_vocab *vocab_ = nullptr;
    llama_sampler *sampler_ = nullptr;
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> <prompt>\n";
        return 1;
    }

    try {
        LlamaSession session(argv[1], 2048);
        std::cout << session.generate(argv[2], 64) << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
