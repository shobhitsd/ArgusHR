# ArgusHR - Intelligent HR Policy Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent RAG (Retrieval-Augmented Generation) system that answers HR policy questions with high accuracy and minimal hallucination. Built for multi-regional HR compliance (Singapore, Malaysia, UAE) with advanced semantic search and prompt engineering.

## Features

- **Multi-Format Document Support**: PDF, Excel, Markdown, and Text files
- **Advanced Semantic Chunking**:
  - Sliding window chunking (800 tokens) with 120-token overlap
  - Structure-aware chunking for Markdown (headers/sections)
  - Page-aware chunking for PDFs
- **State-of-the-Art Retrieval Stack**:
  - **Embeddings**: Voyage AI (`voyage-large-2-instruct`)
  - **Vector Database**: ChromaDB (Persistent storage)
  - **Reranking**: Voyage AI Reranker (`rerank-lite-1`) for precision
  - **LLM**: Llama 3.3 70B via Groq
- **Smart Prompt Engineering**: Structured outputs with chain-of-thought reasoning
- **Interactive UI**: Streamlit-based chat interface

## Demo

### Prompt Comparison: Baseline vs. Structured

| Baseline V1 | Structured V2 |
|:---:|:---:|
| Unstructured, harder to read | Clean, cited, and organized |
| ![Baseline V1](Baseline%20v1.png) | ![Structured V2](Structured%20v2.png) |

## Quick Start

### Prerequisites
- Python 3.10+
- API Keys: [Voyage AI](https://www.voyageai.com) and [Groq](https://groq.com)

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

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**:
   ```bash
   # Interactive UI (Recommended)
   streamlit run app.py
   
   # Or use the batch file (Windows)
   run.bat
   ```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INGESTION PIPELINE                      │
├─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐ │
│   Documents     │───▶│  Document Load   │───▶│   Smart Chunking       │ │
│  (PDF/MD/TXT)   │    │  & Hash Check    │    │   (800 tokens)         │ │
└─────────────────┘    └──────────────────┘    └───────────┬────────────┘ │
                                                           │              │
                              ┌────────────────────────────┘              │
                              ▼                                           │
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│   Voyage AI     │◀───│   Embedding      │───▶│      ChromaDB          │  │
│  (Embeddings)   │    │   Generation     │    │   (Vector Store)       │  │
└─────────────────┘    └──────────────────┘    └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Query Time
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY PROCESSING PIPELINE                        │
├─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐ │
│  User Question  │───▶│  Query Embedding │───▶│  Semantic Search       │ │
│                 │    │   (Voyage AI)    │    │  (Top-10 Results)      │ │
└─────────────────┘    └──────────────────┘    └───────────┬────────────┘ │
                                                           │              │
                              ┌────────────────────────────┘              │
                              ▼                                           │
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│   Voyage AI     │◀───│    Reranking     │───▶│    Top-3 Chunks        │  │
│  (Reranker)     │    │  (rerank-lite-1) │    │    Selected            │  │
└─────────────────┘    └──────────────────┘    └───────────┬────────────┘  │
                                                           │               │
                              ┌────────────────────────────┘               │
                              ▼                                            │
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐   │
│ Structured      │◀───│   LLM Generate   │◀───│   Context + Prompt     │   │
│   Answer        │    │  (Llama 3 Groq)  │    │   (V2 Engineering)     │   │
└─────────────────┘    └──────────────────┘    └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## How It Works

1. **Document Ingestion**: Files are loaded and hashed to detect changes
2. **Intelligent Chunking**: Documents split into 800-token chunks with context preservation
3. **Embedding**: Chunks embedded using Voyage AI and stored in ChromaDB
4. **Two-Stage Retrieval**:
   - Semantic search retrieves top-10 matches
   - Reranker selects top-3 most relevant chunks
5. **Generation**: LLM generates structured answers with citations

## Prompt Engineering

### Evolution: V1 (Baseline) → V2 (Improved)

| Feature | V1 (Baseline) | V2 (Improved) |
|:---|:---|:---|
| **Structure** | Free-form text | Strict sections: Direct Answer, Details, Sources |
| **Reasoning** | Simple instruction | Chain-of-Thought: Analyze → Check → Draft |
| **Hallucination Guard** | "Say if not found" | Explicit trigger: "Information not found in documents" |
| **Citations** | Optional | Mandatory `[Source X]` format |

**Impact**: V2 reduces hallucinations by forcing the model to verify information existence before answering.

## Evaluation Results

Tested on 6 questions across three difficulty levels:

| Category | Example | Result |
|:---|:---|:---|
| **Answerable** | "What is dental coverage in UAE?" | Accurate with citations |
| **Partial** | "Visa submission penalties?" | Correctly identifies gaps |
| **Unanswerable** | "Pet insurance policy?" | Explicit "not found" response |

**Key Metrics**:
- ✅ Zero hallucinations on out-of-scope questions
- ✅ Consistent structured output format
- ✅ Proper source attribution

## Project Structure

```
ArgusHR/
├── app.py                 # Streamlit UI
├── rag_chromadb.py       # Core RAG pipeline
├── evaluation.py         # Evaluation framework
├── requirements.txt      # Dependencies
├── .env.example          # Environment template
├── documents/            # HR policy documents
│   └── Policies/
├── chroma_db/            # Vector database (auto-generated)
└── README.md
```

## Environment Variables

Create a `.env` file:

```env
VOYAGE_API_KEY=your_voyage_api_key
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_url        # Optional
SUPABASE_KEY=your_supabase_key        # Optional
```

## Key Design Decisions

### 1. Chunk Size: 800 Tokens
- **Trade-off**: Larger chunks preserve context but add noise
- **Decision**: 800 tokens captures full policy clauses while fitting LLM context

### 2. Two-Stage Retrieval (Embedding + Reranking)
- **Trade-off**: Adds ~0.5-1s latency
- **Decision**: Precision gain outweighs delay for compliance-critical HR queries

### 3. Persistent Vector Store
- **Trade-off**: Slower than ephemeral storage
- **Decision**: Essential for production to avoid re-embedding on restart

## Future Improvements

- [ ] Multi-Query Retrieval for handling ambiguous questions
- [ ] Conversation memory for multi-turn dialogues
- [ ] A/B testing framework for prompt variants
- [ ] Real-time document sync from cloud storage

## Tech Stack

- **Embeddings**: Voyage AI
- **Vector DB**: ChromaDB
- **LLM**: Llama 3.3 70B (Groq)
- **UI**: Streamlit
- **Language**: Python 3.10+

## License

MIT License - feel free to use this project for your own HR automation needs.

## Acknowledgments

Built as part of an AI Engineering assignment demonstrating RAG best practices and prompt engineering techniques.

---

**Note**: This repository excludes sensitive data (API keys, logs, vector DB). See `.env.example` for required configuration.
