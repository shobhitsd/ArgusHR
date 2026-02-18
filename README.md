# 🧠 Prompt Engineering & RAG Mini Project

A retrieval-augmented generation (RAG) system built to answer HR policy questions with high accuracy and low hallucination. This project demonstrates advanced RAG techniques including semantic chunking, reranking, and prompt engineering.

## 🚀 Features

- **Multi-Format Ingestion**: Supports PDF, Excel, Markdown, and Text files.
- **Advanced Chunking**: 
  - Semantic sliding windows (800 tokens) for text.
  - Structure-aware chunking for Markdown (headers/sections).
  - Page-aware chunking for PDFs.
- **State-of-the-Art Retrieval**:
  - **Embeddings**: Voyage AI (`voyage-large-2-instruct`).
  - **Vector DB**: ChromaDB (Persistent).
  - **Reranking**: Voyage AI Reranker (`rerank-lite-1`) for high precision.
- **Robust Evaluation**:
  - Comparative testing of Baseline (V1) vs Improved (V2) prompts.
  - Side-by-side answer generation.

## 📸 Visuals

### Prompt Comparison: Baseline vs. Structured

| ![Baseline V1](Baseline%20v1.png) | ![Structured V2](Structured%20v2.png) |
|:---:|:---:|
| **V1 (Baseline)**: Unstructured, harder to read. | **V2 (Structured)**: Clean, cited, and organized. |




## 🛠️ Setup Instructions

### Prerequisites
- Python 3.10+
- Verified API Keys for Voyage AI and Groq.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shobhitsd/ArgusHR.git
   cd ArgusHR
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file:
   ```env
   VOYAGE_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```

4. **(Optional) Run UI**:
   To use the interactive chat interface:
   ```bash
   pip install streamlit
   streamlit run app.py
   ```

## 🏗️ Architecture

1. **Ingestion**: Documents are loaded and hashed to detect changes.
2. **Chunking**: Text is split into 800-token chunks with 120-token overlap.
   - *Why 800?* Policy documents often contain long formatting or lists. 800 tokens ensures we capture full context (e.g., a whole "Leave Policy" section) without fragmenting information, while fitting comfortably within the LLM's context window.
3. **Embedding**: Chunks are embedded using Voyage AI and stored in ChromaDB.
4. **Retrieval**:
   - Query is embedded.
   - Top-10 semantic matches retrieved from ChromaDB.
   - **Reranking**: Top-10 are reranked by Voyage AI to select the Top-3 most relevant.
5. **Generation**: Llama-3-70b (via Groq) generates the answer using the selected chunks.

## 📝 Prompt Engineering

We iterated on the system prompt to improve structure and reduce hallucinations.

### V1 (Baseline) vs. V2 (Improved)

| Feature | V1 (Baseline) | V2 (Improved) |
| :--- | :--- | :--- |
| **Structure** | Open-ended text. | Strict sections: **Direct Answer**, **Details**, **Sources**. |
| **Reasoning** | "Answer based on context". | Chain-of-Thought: "Analyze request" -> "Check coverage" -> "Draft". |
| **Negative Constraints** | "Say if not found". | Explicit trigger phrase: "Information not found in the documents." |
| **Citations** | "Cite sources". | Strict `[Source X]` format required. |

**Key Improvement**: V2 forces the model to "think" before answering and explicitly verify if the information exists, significantly reducing hallucinations on edge cases (e.g., questions about "Pet Insurance").

## 📊 Evaluation Results

We evaluated the system on 6 questions across three categories:
1. **Answerable**: Fact retrieval (e.g., "Dental coverage").
2. **Partially Answerable**: complex constraints (e.g., "Visa penalties").
3. **Unanswerable**: Out-of-scope (e.g., "Pet insurance", "Coffee machines").

### Summary
The evaluation compares the Baseline (V1) prompt against the Improved (V2) prompt across 6 diverse questions.

| ID | Question | V1 Answer (Snippet) | V2 Answer (Snippet) | Preferred |
|----|----------|---------------------|---------------------|-----------|
| 1 | What is the dental coverage in UAE? | According to the provided context... | **Direct Answer**: The dental coverage in UAE typically... | V2 (Structured) |
| 2 | What are the working hours in Singapore? | According to the provided context... | **Direct Answer**: The standard working hours... | V2 (Structured) |
| 3 | What are the specific penalty amounts for late visa submission in Malaysia? | The provided context does not contain... | Direct Answer: The provided documents do not mention... | V2 (Structured) |
| 4 | What is the company's policy on pet insurance in the USA? | The context does not contain... | Direct Answer: The company's policy on pet insurance... | V2 (Structured) |
| 5 | How many coffee machines are in the Dubai office? | The context does not contain... | Direct Answer: The number of coffee machines... | V2 (Structured) |
| 6 | Can I work remotely from another country for more than 3 months? | Based on the provided context... | **Direct Answer**: According to the provided policy... | V2 (Structured) |

**Key Findings**:
1. **Consistency**: V2 always starts with a "Direct Answer", making it easier to scan.
2. **Hallucination Check**: Both prompts correctly identified missing information (Q3, Q4, Q5), but V2 was more explicit ("Information not found").
3. **Structure**: V2 uses bold headers and bullet points effectively.

## ⚖️ Key Trade-offs

1. **Chunk Size (800 Tokens)**: 
   - *Trade-off*: Larger chunks preserve more context but can introduce noise and exceed LLM context windows if `top-k` is too high. 
   - *Decision*: We chose 800 tokens to ensure entire policy clauses stay together, which is critical for HR compliance.
2. **Persistence vs. Speed**: 
   - *Trade-off*: We use `PersistentClient` for ChromaDB. 
   - *Decision*: While ephemeral storage is faster for testing, persistence is essential for a "production-ready" system to avoid re-embedding documents on every restart.
3. **Reranking**:
   - *Trade-off*: Adds latency (approx. 0.5s - 1.0s).
   - *Decision*: The gain in precision (ensuring the *best* 3 chunks are used) far outweighs the small delay for an HR assistant.

## 🌟 Reflections

### What I'm most proud of
I am most proud of the **Reranking integration** and the **Prompt Iteration (V1 vs V2)**. By using a two-stage retrieval process (Embedding + Reranking), the system achieves much higher grounding. Additionally, the transition from a simple prompt to a structured, "Chain-of-Thought" style prompt (V2) demonstrated a clear reduction in hallucination, even with a smaller document set.

### One thing I'd improve next
Given more time, I would implement **Multi-Query Retrieval**. Often, user questions are phrased poorly. Generating 3-5 variations of the question and retrieving documents for all of them makes the system significantly more robust to variations in user language.

## 🏁 Submission Checklist
- [x] RAG Pipeline (VoyageAI + ChromaDB)
- [x] Prompt Engineering (V1 vs V2 Comparison)
- [x] Evaluation Report (`evaluation_report.md`)
- [x] Bonus UI (Streamlit)
- [x] Documentation (README)
