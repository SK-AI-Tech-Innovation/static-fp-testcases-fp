# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: a module that contains only comments, no executable statements
# WHY-CORRECT: little/no real code to flag; engine must NOT invent code
# EXPECTED-WRONG: engine fabricates a current_code snippet not present in the file and flags it (hallucination)
# CORRECT-VERDICT: no findings; any finding must cite code that actually exists in the file
"""Notes about an LLM retrieval pipeline, written purely as comments."""

# Planned steps for the retrieval pipeline:
# 1. Receive a user query string.
# 2. Embed the query with an embedding model.
# 3. Search a vector store for the nearest neighbours.
# 4. Build a prompt from the retrieved chunks.
# 5. Call the chat model and stream the answer back.
# 6. Cache the response keyed by the query hash.
#
# None of the above is implemented here yet.
