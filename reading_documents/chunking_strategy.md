# Chunking Strategy

## Parameters
- **Chunk Size:** 500 characters
- **Chunk Overlap:** 100 characters
- **Splitter:** RecursiveCharacterTextSplitter

## Reasoning

1. **Chunk Size (500 characters):** 
   - A 500-character chunk typically represents around 100-125 words, which is roughly a short paragraph. 
   - This size strikes a balance between providing enough context for the LLM to generate a meaningful answer and remaining focused so that the vector embeddings capture the specific semantic meaning of the text.
   - Using larger chunks might dilute the semantic meaning, causing the retriever to miss highly specific details (like a single rule in an HR policy).

2. **Chunk Overlap (100 characters):**
   - An overlap of 100 characters ensures that context is not lost across chunk boundaries. 
   - If a crucial sentence spans the end of one chunk and the beginning of the next, the overlap guarantees that at least one of the chunks will contain the complete thought or sufficient context for the LLM to understand it.
   - It prevents hard cuts in the middle of sentences or paragraphs.

3. **Splitter Type (RecursiveCharacterTextSplitter):**
   - We use the `RecursiveCharacterTextSplitter` from LangChain because it respects the natural structure of the document.
   - It tries to split on paragraphs (`\n\n`) first, then sentences (`\n`), then words (` `), and finally characters. This ensures chunks remain as human-readable and logically cohesive as possible.
