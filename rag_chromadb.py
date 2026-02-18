"""
ArgusHR RAG Engine with ChromaDB
Enhanced RAG implementation with persistent vector storage.
"""
import os
import re
import time
import hashlib
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
from datetime import datetime

# Suppress all warnings (including Pydantic and OpenTelemetry)
warnings.filterwarnings("ignore")

import numpy as np
import tiktoken
import voyageai
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import pandas as pd
import pdfplumber

# Configure logging
logging.basicConfig(level=logging.ERROR) # Only show errors
logger = logging.getLogger(__name__)

# Suppress noisy libraries aggressively
for lib in ["httpx", "watchdog", "chromadb", "voyageai", "opentelemetry", "pydantic"]:
    logging.getLogger(lib).setLevel(logging.CRITICAL)

# Load environment variables
load_dotenv(override=True)

# ---------------- CONFIGURATION ----------------
class Config:
    # Embedding settings
    EMBED_MODEL = "voyage-4-large"
    RERANK_MODEL = "rerank-2.5"
    
    # Chunking settings
    MAX_TOKENS = 800
    OVERLAP_TOKENS = 120
    
    # Retrieval settings
    TOP_K = 10
    RERANK_TOP_K = 3
    
    # API limits
    MAX_BATCH_SIZE = 100
    MAX_BATCH_TOKENS = 90000
    WAIT_TIME = 0.3
    
    # LLM settings
    MAX_CONTEXT_TOKENS = 12000  # Increased from 6000
    LLM_MODEL = "llama-3.3-70b-versatile"
    
    # Directories (can be overridden)
    DOCUMENTS_DIR = "documents"
    CHROMA_DIR = "chroma_db"
    
    @classmethod
    def set_documents_dir(cls, path: str):
        """Set documents directory to a specific path or subfolder."""
        cls.DOCUMENTS_DIR = path

config = Config()

# ---------------- TOKENIZER ----------------
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(enc.encode(text))

# ---------------- CUSTOM EXCEPTIONS ----------------
class RAGError(Exception):
    """Base RAG exception."""
    pass

class DocumentProcessingError(RAGError):
    """Error processing a document."""
    pass

class EmbeddingError(RAGError):
    """Error generating embeddings."""
    pass

class QueryError(RAGError):
    """Error during query."""
    pass

# ---------------- FILE UTILITIES ----------------
def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash for change detection."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        logger.error(f"Could not hash file {file_path}: {e}")
        raise DocumentProcessingError(f"Cannot read file: {file_path}")

def discover_files(directory: str) -> List[Dict]:
    """Discover all supported files in directory."""
    supported_extensions = {'.xlsx', '.xls', '.pdf', '.md', '.txt'}
    files = []
    
    doc_path = Path(directory)
    if not doc_path.exists():
        logger.info(f"Creating {directory}/ folder...")
        doc_path.mkdir(parents=True, exist_ok=True)
        return []
    
    for file_path in doc_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                relative_path = file_path.relative_to(doc_path)
            except ValueError:
                relative_path = file_path.name

            files.append({
                'path': str(file_path),
                'relative_path': str(relative_path),
                'name': file_path.name,
                'type': file_path.suffix.lower(),
                'hash': get_file_hash(str(file_path)),
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    return sorted(files, key=lambda x: x['name'])

# ---------------- DOCUMENT PROCESSORS ----------------
def process_excel(file_path: str) -> List[Dict]:
    """Process Excel file with enhanced metadata."""
    chunks = []
    
    try:
        excel_file = pd.ExcelFile(file_path)
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if df.empty:
                continue
            
            sheet_text = f"## Sheet: {sheet_name}\n\n"
            sheet_text += df.to_markdown(index=False) if hasattr(df, 'to_markdown') else df.to_string(index=False)
            
            if count_tokens(sheet_text) <= config.MAX_TOKENS:
                chunks.append({
                    'text': sheet_text,
                    'metadata': {
                        'source': file_path,
                        'source_name': Path(file_path).name,
                        'sheet': sheet_name,
                        'type': 'excel',
                        'rows': len(df),
                        'columns': list(df.columns),
                        'indexed_at': datetime.now().isoformat()
                    }
                })
            else:
                chunk_size = 50
                for i in range(0, len(df), chunk_size):
                    chunk_df = df.iloc[i:i+chunk_size]
                    chunk_text = f"## Sheet: {sheet_name} (Rows {i+1}-{min(i+chunk_size, len(df))})\n\n"
                    chunk_text += chunk_df.to_markdown(index=False) if hasattr(chunk_df, 'to_markdown') else chunk_df.to_string(index=False)
                    
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'source': file_path,
                            'source_name': Path(file_path).name,
                            'sheet': sheet_name,
                            'type': 'excel',
                            'row_range': f"{i+1}-{min(i+chunk_size, len(df))}",
                            'columns': list(df.columns),
                            'indexed_at': datetime.now().isoformat()
                        }
                    })
        
        logger.info(f"✓ Processed {len(excel_file.sheet_names)} sheets from {Path(file_path).name}, created {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error processing Excel {file_path}: {e}")
        raise DocumentProcessingError(f"Excel processing failed: {e}")
    
    return chunks

def process_pdf(file_path: str) -> List[Dict]:
    """Process PDF file with enhanced metadata."""
    chunks = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                
                if not text or len(text.strip()) < 50:
                    continue
                
                page_text = f"## Page {page_num} of {total_pages}\n\n{text}"
                
                if count_tokens(page_text) <= config.MAX_TOKENS:
                    chunks.append({
                        'text': page_text,
                        'metadata': {
                            'source': file_path,
                            'source_name': Path(file_path).name,
                            'page': page_num,
                            'total_pages': total_pages,
                            'type': 'pdf',
                            'indexed_at': datetime.now().isoformat()
                        }
                    })
                else:
                    paragraphs = text.split('\n\n')
                    current_chunk = f"## Page {page_num} of {total_pages}\n\n"
                    
                    for para in paragraphs:
                        if count_tokens(current_chunk + para) > config.MAX_TOKENS:
                            if current_chunk.strip():
                                chunks.append({
                                    'text': current_chunk.strip(),
                                    'metadata': {
                                        'source': file_path,
                                        'source_name': Path(file_path).name,
                                        'page': page_num,
                                        'total_pages': total_pages,
                                        'type': 'pdf',
                                        'indexed_at': datetime.now().isoformat()
                                    }
                                })
                            current_chunk = f"## Page {page_num} of {total_pages}\n\n{para}\n\n"
                        else:
                            current_chunk += para + "\n\n"
                    
                    if current_chunk.strip():
                        chunks.append({
                            'text': current_chunk.strip(),
                            'metadata': {
                                'source': file_path,
                                'source_name': Path(file_path).name,
                                'page': page_num,
                                'total_pages': total_pages,
                                'type': 'pdf',
                                'indexed_at': datetime.now().isoformat()
                            }
                        })
            
            logger.info(f"✓ Processed {total_pages} pages from {Path(file_path).name}, created {len(chunks)} chunks")
            
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        raise DocumentProcessingError(f"PDF processing failed: {e}")
    
    return chunks

def split_markdown_blocks(md: str) -> List[str]:
    """Split markdown into semantic blocks."""
    blocks = []
    buf = []
    in_code = False

    lines = md.splitlines()
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            continue

        if in_code:
            buf.append(line)
            continue

        if re.match(r"^#{1,6}\s+", line):
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            buf.append(line)
            continue

        if "|" in line and re.match(r"^\s*\|.*\|\s*$", line):
            buf.append(line)
            continue

        if re.match(r"^\s*[-*+]\s+", line):
            buf.append(line)
            continue

        if line.strip() == "":
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            continue

        buf.append(line)

    if buf:
        blocks.append("\n".join(buf).strip())

    return [b for b in blocks if b.strip()]

def chunk_markdown_blocks(blocks: List[str], max_tokens: int, overlap_tokens: int) -> List[str]:
    """Chunk markdown blocks with overlap."""
    chunks = []
    current = []
    current_tokens = 0

    def flush_with_overlap():
        nonlocal current, current_tokens
        if not current:
            return
        chunk_text = "\n\n".join(current).strip()
        chunks.append(chunk_text)

        if overlap_tokens > 0:
            tokens = enc.encode(chunk_text)
            overlap = tokens[-overlap_tokens:] if len(tokens) > overlap_tokens else tokens
            overlap_text = enc.decode(overlap)
            current = [overlap_text]
            current_tokens = count_tokens(overlap_text)
        else:
            current = []
            current_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block)

        if block_tokens > max_tokens:
            if current:
                flush_with_overlap()

            tokens = enc.encode(block)
            start = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk = enc.decode(tokens[start:end])
                chunks.append(chunk)
                start = end - overlap_tokens if overlap_tokens > 0 and end - overlap_tokens > start else end
            current = []
            current_tokens = 0
            continue

        if current_tokens + block_tokens <= max_tokens:
            current.append(block)
            current_tokens += block_tokens
        else:
            flush_with_overlap()
            current.append(block)
            current_tokens += block_tokens

    if current:
        flush_with_overlap()

    return chunks

def process_markdown(file_path: str) -> List[Dict]:
    """Process Markdown file with header extraction."""
    chunks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract document title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else Path(file_path).stem
        
        blocks = split_markdown_blocks(content)
        chunk_texts = chunk_markdown_blocks(blocks, config.MAX_TOKENS, config.OVERLAP_TOKENS)
        
        for i, text in enumerate(chunk_texts):
            # Extract section header if present
            header_match = re.search(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
            section = header_match.group(1) if header_match else None
            
            chunks.append({
                'text': text,
                'metadata': {
                    'source': file_path,
                    'source_name': Path(file_path).name,
                    'chunk_id': i,
                    'type': 'markdown',
                    'title': doc_title,
                    'section': section,
                    'indexed_at': datetime.now().isoformat()
                }
            })
        
        logger.info(f"✓ Processed {Path(file_path).name}, created {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error processing Markdown {file_path}: {e}")
        raise DocumentProcessingError(f"Markdown processing failed: {e}")
    
    return chunks

def process_text(file_path: str) -> List[Dict]:
    """Process plain text file."""
    chunks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        paragraphs = content.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            if count_tokens(current_chunk + para) > config.MAX_TOKENS:
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'source': file_path,
                            'source_name': Path(file_path).name,
                            'chunk_id': chunk_id,
                            'type': 'text',
                            'indexed_at': datetime.now().isoformat()
                        }
                    })
                    chunk_id += 1
                current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'source': file_path,
                    'source_name': Path(file_path).name,
                    'chunk_id': chunk_id,
                    'type': 'text',
                    'indexed_at': datetime.now().isoformat()
                }
            })
        
        logger.info(f"✓ Processed {Path(file_path).name}, created {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error processing text {file_path}: {e}")
        raise DocumentProcessingError(f"Text processing failed: {e}")
    
    return chunks

def process_file(file_info: Dict) -> List[Dict]:
    """Process file based on type."""
    file_path = file_info['path']
    file_type = file_info['type']
    
    logger.info(f"📄 Processing {file_info['name']}...")
    
    processors = {
        '.xlsx': process_excel,
        '.xls': process_excel,
        '.pdf': process_pdf,
        '.md': process_markdown,
        '.txt': process_text
    }
    
    processor = processors.get(file_type)
    if processor:
        return processor(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_type}")
        return []

# ---------------- RAG ENGINE CLASS ----------------
class RAGEngine:
    """ChromaDB-based RAG Engine."""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or config.CHROMA_DIR
        self.voyage_client = self._get_voyage_client()
        self.chroma_client = None
        self.collection = None
        self._init_chroma()
    
    def _get_voyage_client(self) -> voyageai.Client:
        """Get Voyage AI client."""
        key = os.getenv("VOYAGE_API_KEY")
        if not key:
            raise ValueError("VOYAGE_API_KEY not set in environment")
        return voyageai.Client(api_key=key)
    
    def _init_chroma(self):
        """Initialize ChromaDB."""
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="argushr_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"✅ ChromaDB initialized at {self.persist_directory}")
        logger.info(f"   Collection has {self.collection.count()} documents")
    
    def _embed_with_retry(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """Embed texts with exponential backoff retry."""
        all_embeddings = []
        
        # Create batches
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            text_tokens = count_tokens(text)
            
            if (current_tokens + text_tokens > config.MAX_BATCH_TOKENS and current_batch) or \
               len(current_batch) >= config.MAX_BATCH_SIZE:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        # Process batches
        for batch_num, batch in enumerate(batches, 1):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    res = self.voyage_client.embed(batch, model=config.EMBED_MODEL)
                    all_embeddings.extend(res.embeddings)
                    break
                except Exception as e:
                    retry_count += 1
                    wait_time = min(60, 2 ** retry_count)  # Exponential backoff
                    logger.warning(f"Embedding error (attempt {retry_count}): {e}")
                    if retry_count < max_retries:
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise EmbeddingError(f"Failed after {max_retries} attempts: {e}")
            
            if batch_num < len(batches):
                time.sleep(config.WAIT_TIME)
        
        return all_embeddings
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """Build or update the index."""
        logger.info("=" * 60)
        logger.info("📂 Discovering documents...")
        
        files = discover_files(config.DOCUMENTS_DIR)
        
        if not files:
            logger.warning("No documents found. Please add files to documents/ folder")
            return False
        
        logger.info(f"Found {len(files)} file(s)")
        
        # Check if rebuild needed
        if not force_rebuild and self.collection.count() > 0:
            logger.info("✅ Using existing ChromaDB index")
            return True
        
        # Clear existing data if rebuilding
        if force_rebuild and self.collection.count() > 0:
            logger.info("🗑️ Clearing existing index...")
            self.chroma_client.delete_collection("argushr_documents")
            self.collection = self.chroma_client.create_collection(
                name="argushr_documents",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Process all files
        logger.info("=" * 60)
        logger.info("🔨 Processing documents...")
        
        all_chunks = []
        for file_info in files:
            try:
                chunks = process_file(file_info)
                all_chunks.extend(chunks)
            except DocumentProcessingError as e:
                logger.error(f"Skipping {file_info['name']}: {e}")
        
        if not all_chunks:
            logger.warning("No chunks created from documents")
            return False
        
        logger.info(f"✅ Total chunks: {len(all_chunks)}")
        
        # Generate embeddings
        logger.info("=" * 60)
        logger.info("🧮 Generating embeddings...")
        
        chunk_texts = [c['text'] for c in all_chunks]
        embeddings = self._embed_with_retry(chunk_texts)
        
        # Add to ChromaDB
        logger.info("💾 Storing in ChromaDB...")
        
        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        documents = chunk_texts
        metadatas = [c['metadata'] for c in all_chunks]
        
        # ChromaDB has batch limits, add in chunks
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        
        logger.info(f"✅ Index built: {self.collection.count()} chunks stored")
        return True
    
    def query(self, query_text: str, top_k: int = None, use_llm: bool = True, prompt_version: str = "v1") -> Dict:
        """
        Query the RAG system.
        
        Args:
            query_text: The user query
            top_k: Number of documents to retrieve
            use_llm: Whether to generate an answer with LLM
            prompt_version: "v1" (Baseline) or "v2" (Improved)
        """
        top_k = top_k or config.RERANK_TOP_K
        
        if self.collection.count() == 0:
            return {
                'answer': "No documents indexed. Please add documents to the documents/ folder.",
                'sources': []
            }
        
        # Embed query
        query_embedding = self._embed_with_retry([query_text])[0]
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=config.TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return {
                'answer': "No relevant documents found for your query.",
                'sources': []
            }
        
        # Prepare for reranking
        candidates = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Rerank with Voyage
        rerank_results = self.voyage_client.rerank(
            query=query_text,
            documents=candidates,
            model=config.RERANK_MODEL,
            top_k=top_k
        )
        
        # Build sources
        sources = []
        context_chunks = []
        for r in rerank_results.results:
            idx = r.index
            sources.append({
                'text': candidates[idx],
                'metadata': metadatas[idx],
                'score': r.relevance_score
            })
            context_chunks.append({
                'text': candidates[idx],
                'metadata': metadatas[idx]
            })
        
        # Generate answer
        answer = None
        if use_llm:
            answer = self._generate_answer(query_text, context_chunks, prompt_version)
        
        return {
            'answer': answer,
            'sources': sources
        }
    
    def _generate_answer(self, query: str, context_chunks: List[Dict], prompt_version: str) -> str:
        """Generate answer using Groq."""
        try:
            from groq import Groq
        except ImportError:
            return "⚠️ groq not installed. Run: pip install groq"
        
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            return "⚠️ GROQ_API_KEY not set. Add it to your .env file."
        
        client = Groq(api_key=groq_key)
        
        # Build context
        current_tokens = 0
        context_parts = []
        
        for i, chunk in enumerate(context_chunks, 1):
            source_name = chunk['metadata'].get('source_name', 'Unknown')
            meta_info = ""
            
            if chunk['metadata']['type'] == 'excel':
                meta_info = f" (Sheet: {chunk['metadata'].get('sheet', 'N/A')})"
            elif chunk['metadata']['type'] == 'pdf':
                meta_info = f" (Page: {chunk['metadata'].get('page', 'N/A')})"
            elif chunk['metadata']['type'] == 'markdown':
                section = chunk['metadata'].get('section')
                if section:
                    meta_info = f" (Section: {section})"
            
            part_text = f"[Source {i}: {source_name}{meta_info}]\n{chunk['text']}"
            part_tokens = count_tokens(part_text)
            
            if current_tokens + part_tokens > config.MAX_CONTEXT_TOKENS:
                break
            
            context_parts.append(part_text)
            current_tokens += part_tokens
        
        context = "\n\n---\n\n".join(context_parts)
        
        # ---------------- PROMPT ENGINEERING ----------------
        
        if prompt_version == "v2":
            # IMPROVED PROMPT: Structured, Role-Playing, Chain-of-Thought
            prompt = f"""You are ArgusHR, an expert HR policy assistant. Your goal is to provide accurate, comprehensive, and well-cited answers based STRICTLY on the provided company policy documents.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. **Analyze the Request**: specific details, regions (e.g., UAE vs Singapore), or policy types.
2. **Check Coverage**: If the answer is NOT in the context, state "Information not found in the documents." Do not hallucinate or use outside knowledge.
3. **Structure Your Answer**:
   - **Direct Answer**: A concise 1-2 sentence summary.
   - **Details**: Use bullet points for specific figures, conditions, or steps.
   - **Citations**: End with a "Sources" section strictly referencing the [Source X] tags from the context.
4. **Tone**: Professional, helpful, and direct.

ANSWER:"""
        
        else:
            # BASELINE PROMPT (V1)
            prompt = f"""You are ArgusHR, an AI assistant specialized in HR policies, employment regulations, and workplace compliance.

You help employees and HR professionals understand:
- Employment contracts and terms
- Regional labor laws (Singapore, UAE, Malaysia)
- Benefits (CPF, EPF, SOCSO, gratuity)
- Visa and work permit requirements
- Company policies and procedures

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. If the context doesn't contain the answer, say so clearly
3. Be specific and cite sources when possible
4. Use bullet points for clarity
5. For regional questions, apply the correct jurisdiction's rules

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"⚠️ Error generating response: {e}"
    
    def get_chunk_count(self) -> int:
        """Get number of chunks in collection."""
        return self.collection.count() if self.collection else 0
    
    def get_doc_count(self) -> int:
        """Get number of unique source documents."""
        if not self.collection or self.collection.count() == 0:
            return 0
        
        # Query all metadata to count unique sources
        results = self.collection.get(include=["metadatas"])
        sources = set(m.get('source', '') for m in results['metadatas'])
        return len(sources)

# ---------------- CLI INTERFACE ----------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ArgusHR RAG Engine (ChromaDB)")
    parser.add_argument("--folder", "-f", type=str, default="documents",
                        help="Folder to index (e.g., 'documents/policies' or just 'policies')")
    parser.add_argument("--rebuild", "-r", action="store_true",
                        help="Force rebuild the index")
    parser.add_argument("--query", "-q", type=str,
                        help="Run a single query and exit")
    
    args = parser.parse_args()
    
    # Set documents directory
    folder = args.folder
    if not folder.startswith("documents") and not os.path.isabs(folder):
        folder = f"documents/{folder}"
    Config.set_documents_dir(folder)
    
    print("=" * 60)
    print("🚀 ArgusHR RAG Engine (ChromaDB)")
    print("=" * 60)
    print(f"📂 Document folder: {config.DOCUMENTS_DIR}")
    
    engine = RAGEngine()
    engine.build_index(force_rebuild=args.rebuild)
    
    # Single query mode
    if args.query:
        print(f"\n🔍 Query: {args.query}\n")
        result = engine.query(args.query)
        print("=" * 60)
        print("🤖 Answer:")
        print("=" * 60)
        print(result['answer'])
        print("\n📚 Sources:")
        for i, src in enumerate(result['sources'], 1):
            print(f"  {i}. {src['metadata'].get('source_name', 'Unknown')} (score: {src['score']:.3f})")
        exit(0)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("RAG Query Interface")
    print("=" * 60)
    print("Commands: 'exit', 'rebuild', 'stats'")
    print()
    
    while True:
        user_input = input("❓ Your question: ").strip()
        
        if user_input.lower() in ["exit", "quit", ""]:
            print("Goodbye!")
            break
        
        if user_input.lower() == "rebuild":
            engine.build_index(force_rebuild=True)
            continue
        
        if user_input.lower() == "stats":
            print(f"\n📊 Index Statistics")
            print(f"   Chunks: {engine.get_chunk_count()}")
            print(f"   Documents: {engine.get_doc_count()}\n")
            continue
        
        print("\n🔍 Searching...\n")
        result = engine.query(user_input)
        
        print("=" * 60)
        print("🤖 Answer:")
        print("=" * 60)
        print(result['answer'])
        
        print("\n📚 Sources:")
        for i, src in enumerate(result['sources'], 1):
            print(f"  {i}. {src['metadata'].get('source_name', 'Unknown')} (score: {src['score']:.3f})")
        print()

