// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: LangChain4j RAG with EmbeddingStoreContentRetriever (dev.langchain4j), verified June 2026
// WHY-CORRECT: complete RAG wiring — embed documents into an EmbeddingStore, build an
//              EmbeddingStoreContentRetriever, and attach it to an AiServices proxy via
//              .contentRetriever(...). Retrieval augmentation is handled by the framework at call time.
// EXPECTED-WRONG: engine may flag "RAG context never injected into the prompt" because the retrieval step is
//                 declarative and not a visible string-concatenation into the user message.
// CORRECT-VERDICT: no findings
/** End-to-end LangChain4j RAG: ingest documents, then query via a retriever-backed AiService. */
package com.example.ai.langchain4j;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

public class langchain4j_rag {

    interface DocsAssistant {
        String answer(String question);
    }

    public static DocsAssistant build(String apiKey, List<String> documents) {
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        for (String doc : documents) {
            TextSegment segment = TextSegment.from(doc);
            embeddingStore.add(embeddingModel.embed(segment).content(), segment);
        }

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(4)
                .minScore(0.6)
                .build();

        ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .build();

        return AiServices.builder(DocsAssistant.class)
                .chatModel(chatModel)
                .contentRetriever(retriever)
                .build();
    }

    public static void main(String[] args) {
        DocsAssistant assistant = build(
                System.getenv("OPENAI_API_KEY"),
                List.of("Refunds are processed within 5 business days.", "Support hours are 9am-6pm UTC."));
        System.out.println(assistant.answer("How long do refunds take?"));
    }
}
