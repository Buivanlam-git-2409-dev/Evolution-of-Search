# CLAUDE.md - Semantic Search Comparison Project

## Project

This project demonstrates the evolution of search systems from traditional keyword-based retrieval (TF-IDF) to semantic search using sentence embeddings.

The application provides a side-by-side comparison between:
- TF-IDF keyword search
- Embedding-based semantic search

The project uses the MS MARCO dataset for experimentation and evaluation.

---

# Stack

## Core Technologies
- Python 3.10+
- Streamlit
- scikit-learn
- sentence-transformers
- FAISS
- NumPy
- pandas

## Machine Learning / NLP
- TF-IDF Vectorization
- Sentence Embeddings
- Cosine Similarity
- Semantic Retrieval

## Dependencies
- sentence-transformers >= 2.2.0
- scikit-learn >= 1.3.0
- streamlit >= 1.28.0
- faiss-cpu >= 1.7.4
- numpy >= 1.24.0
- pandas >= 2.0.0

---

# Project Structure

## .claude/
Contains Claude Code workflows, rules, and agents.

### skills/
Custom reusable workflows.
- `build-index/` - Build semantic and TF-IDF indices
- `evaluate-search/` - Evaluate retrieval quality
- `run-webapp/` - Launch Streamlit UI
- `process-dataset/` - Dataset preprocessing and cleaning

### agents/
Specialized project agents.
- `search-expert.md` - Semantic retrieval optimization and FAISS tuning
- `nlp-expert.md` - NLP preprocessing and text normalization

---

## data/
Dataset storage and processed corpus files.

- `raw/`
  - Original MS MARCO dataset
- `processed/`
  - Cleaned and normalized passages
- `cache/`
  - Cached embeddings and intermediate artifacts

---

## models/
Embedding model configuration and cached models.

- `embedding_model/`
- `faiss_index/`

---

## app/
Streamlit web application.

- `app.py`
  - Main Streamlit UI
- `components.py`
  - Reusable UI components
- `utils.py`
  - UI helpers and formatting

---

## core/
Core search engine logic and infrastructure.

- `engine.py`
  - SearchEngine class: orchestrates indexing and queries
  - Loads MS MARCO dataset and manages document pipeline
- `processor.py`
  - TextProcessor class: text normalization and cleaning
  - URL removal, punctuation handling, stopword filtering
- `retrievers.py`
  - BaseRetriever abstract class
  - TFIDFRetriever: traditional keyword-based retrieval
  - SemanticRetriever: embedding-based retrieval with FAISS support
- `schemas.py`
  - SearchResult dataclass: rank, score, text

---

## scripts/
Utility and automation scripts.

- `download_data.py`
  - Download MS MARCO dataset from Hugging Face
  - Usage: `python scripts/download_data.py --sample-size 1500`
- `build_index.py`
  - Build and cache TF-IDF and Semantic indices
  - Saves pickled retrievers to models/ directory
  - Usage: `python scripts/build_index.py`
- `evaluate.py`
  - Evaluate and compare retrieval quality on test queries
  - Measures latency for both methods
  - Usage: `python scripts/evaluate.py --queries "query1" "query2"`
- `benchmark.py`
  - Benchmark performance across different sample sizes
  - Outputs JSON results to logs/
  - Usage: `python scripts/benchmark.py --sample-sizes 1000 1500 2000`

---

## models/
Cached models and indices (generated at runtime).

- `tfidf_retriever.pkl` - Pickled TFIDFRetriever
- `semantic_retriever.pkl` - Pickled SemanticRetriever with embeddings

---

## logs/
Application logs and benchmark results.

- `benchmark_results.json` - Performance metrics from benchmark.py

---

# Architecture

## Core Components

### TextProcessor
Responsible for (in core/processor.py):
- Lowercasing text
- URL removal via regex
- Punctuation cleaning
- Stopword filtering (English)
- Multi-space normalization

### BaseRetriever (Abstract)
Abstract base class defining retriever interface:
- `build_index(documents)`: Build index from documents
- `search(query, top_k)`: Search for top-k results

### TFIDFRetriever
Baseline lexical search (in core/retrievers.py):
- Builds TfidfVectorizer with max_features=20000, ngram_range=(1,2)
- Uses scikit-learn for vectorization
- Cosine similarity ranking
- Fast, deterministic, transparent

### SemanticRetriever
Advanced embedding-based search (in core/retrievers.py):
- Loads sentence-transformers model (all-MiniLM-L6-v2)
- Generates normalized embeddings
- FAISS IndexFlatIP for fast similarity search
- Falls back to NumPy if FAISS unavailable

### SearchEngine
Main orchestration (in core/engine.py):
- Loads MS MARCO dataset from Hugging Face
- Creates cleaned documents via TextProcessor
- Builds both TFIDFRetriever and SemanticRetriever
- Provides search_tfidf(), search_semantic(), compare() methods
- Returns results with timing metrics

### SearchResult
Response dataclass (in core/schemas.py):
- rank: Position in results (1-indexed)
- score: Similarity score (0-1)
- text: Document text

---

# Search Approaches

## 1. TF-IDF Search (Baseline)

Traditional lexical search - implemented in TFIDFRetriever:
- `TfidfVectorizer` from scikit-learn
- Unigrams + bigrams (1-2 word sequences)
- Cosine similarity ranking

### TF-IDF Configuration
- `max_features=20000` (vocabulary size)
- `ngram_range=(1,2)` (unigrams + bigrams)
- Preprocessed via TextProcessor.clean()

### Characteristics
✅ Deterministic, reproducible
✅ Interpretable (keyword matching)
✅ Fast for small/medium corpora
❌ Limited semantic understanding
❌ No word order beyond bigrams

---

## 2. Semantic Search (Advanced)

Embedding-based semantic retrieval - implemented in SemanticRetriever:
- `sentence-transformers` (huggingface)
- `all-MiniLM-L6-v2` model
- FAISS IndexFlatIP for similarity search

### Embedding Configuration
- Model: `all-MiniLM-L6-v2` (384-dim embeddings)
- Normalization: Enabled (cosine similarity)
- FAISS index: IndexFlatIP for IP similarity
- Fallback: NumPy if FAISS unavailable

### Characteristics
✅ Semantic understanding
✅ Handles synonyms and paraphrases
✅ Query reformulations
❌ Slower than TF-IDF (embedding computation)
❌ Model dependent
❌ Less interpretable

---

# Search Configuration Rules

## Default Parameters
- `top_k=5`
- `num_passages=1500`

## Configuration Consistency
- Both TF-IDF and semantic search MUST use the same cleaned corpus.
- Keep preprocessing deterministic across all pipelines.
- Store search configuration in centralized constants or config files.
- Avoid duplicated configuration values across files.

---

# Data Flow

1. Download MS MARCO dataset
2. Clean and normalize passages
3. Build TF-IDF vectors
4. Generate sentence embeddings
5. Normalize embeddings
6. Build FAISS index
7. Execute queries against both systems
8. Compare ranking quality and latency

---

# Commands

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py --sample-size 1500
```

## Development
```bash
# Run Streamlit web application
streamlit run app/app.py

# Build and cache indices
python scripts/build_index.py --sample-size 1500 --output-dir models

# Evaluate search quality
python scripts/evaluate.py --top-k 5

# Benchmark performance
python scripts/benchmark.py --sample-sizes 1000 1500 2000 --num-queries 10
```

## Testing
```python
# Test TF-IDF
python -c "from core.engine import SearchEngine; e=SearchEngine(); e.prepare(); print(e.search_tfidf('machine learning', 3)['results'][0])"

# Test Semantic
python -c "from core.engine import SearchEngine; e=SearchEngine(); e.prepare(); print(e.search_semantic('AI techniques', 3)['results'][0])"

# Compare
python -c "from core.engine import SearchEngine; e=SearchEngine(); e.prepare(); print(e.compare('deep learning', 5))"
```

---

# Conventions

## Python Style
- Follow PEP 8
- Use type hints for all functions
- Add docstrings to all classes and functions
- Use `snake_case` naming convention
- Use dataclasses where appropriate

## Imports
Import order:
1. Standard library
2. Third-party libraries
3. Local imports

Separate import groups with blank lines.

## Project Structure
- Keep business logic outside Streamlit UI
- Separate preprocessing from retrieval logic
- Avoid duplicated search logic
- Keep configuration centralized

---

# YOU MUST

## Search & Retrieval
- Always normalize embeddings before similarity search
- Use cosine similarity consistently
- Keep TF-IDF and semantic corpus identical
- Use batch embedding generation for performance
- Cache embedding model loading
- Rebuild embeddings only when corpus changes

## Performance
- Use FAISS whenever available
- Avoid loading unnecessary data into memory
- Cache heavy resources with `@st.cache_resource`
- Benchmark search latency after major changes

## Code Safety
- Never hardcode absolute paths
- Use `pathlib` instead of `os.path`
- Add proper error handling for model loading
- Validate dataset existence before indexing
- Log important indexing and retrieval events

## Data Management
- Do not commit large embedding cache files
- Do not commit generated FAISS indices larger than 100MB
- Keep raw and processed datasets separated
- Version preprocessing logic carefully

## Streamlit
- Use `@st.cache_resource` for model loading
- Avoid rebuilding indices during UI reruns
- Separate UI rendering from retrieval logic

---

# Environment

Create a `.env` file in the project root:

```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=5
NUM_PASSAGES=1500
USE_FAISS=true
MAX_FEATURES=5000
NGRAM_RANGE=1,2
```

---

# Skills (Auto-invoked)

| When | Skill | Reference |
|------|-------|-----------|
| Prepare fresh dataset | `process-dataset` | `.claude/skills/process-dataset/SKILL.md` |
| Evaluate retrieval quality | `evaluate-search` | `.claude/skills/evaluate-search/SKILL.md` |
| Run Streamlit app | `run-webapp` | `.claude/skills/run-webapp/SKILL.md` |
| Benchmark performance | `benchmark` | `.claude/skills/benchmark/SKILL.md` |

---

# External References

## Main Source Files
- `@core/engine.py` - SearchEngine orchestration
- `@core/processor.py` - TextProcessor implementation
- `@core/retrievers.py` - TFIDFRetriever & SemanticRetriever
- `@app/app.py` - Streamlit main interface

## Utility Scripts
- `@scripts/download_data.py` - Dataset loading
- `@scripts/build_index.py` - Index building
- `@scripts/evaluate.py` - Quality evaluation
- `@scripts/benchmark.py` - Performance benchmarking

## Rules & Guidelines
- `@.claude/rules/code-style.md` - Python style guide
- `@.claude/rules/ml-guidelines.md` - ML best practices
- `@.claude/rules/performance.md` - Performance optimization

## Agents
- `@.claude/agents/search-expert.md` - Semantic retrieval expert
- `@.claude/agents/nlp-expert.md` - NLP preprocessing expert

## Skills
- `@.claude/skills/process-dataset/SKILL.md` - Dataset preparation
- `@.claude/skills/evaluate-search/SKILL.md` - Search evaluation

---

# Current State
- **Last session:** May 7, 2026 - Project structure refactored & completed
- **Completed:** 
  - ✅ Core architecture with BaseRetriever
  - ✅ TF-IDF retriever implementation
  - ✅ Semantic retriever with FAISS support
  - ✅ All 4 utility scripts (download, build, evaluate, benchmark)
  - ✅ UI components library
  - ✅ Claude agents (search-expert, nlp-expert)
- **Status:** Ready for demo and experimentation
- **Next task:** Test all scripts end-to-end, optimize model selection

---

# Quick Reference

| Task | Command | Skill |
|------|---------|-------|
| Run web app | `streamlit run app/app.py` | run-webapp |
| Build indices | `python scripts/build_index.py` | N/A |
| Evaluate search | `python scripts/evaluate.py` | evaluate-search |
| Download dataset | `python scripts/download_data.py` | process-dataset |
| Benchmark system | `python scripts/benchmark.py` | benchmark |
| Test TF-IDF | `python -c "from core.engine import SearchEngine..."` | N/A |

---

# Key Decisions

1. **Modular Architecture**: Separated TextProcessor, Retrievers into distinct modules for maintainability
2. **Abstract BaseRetriever**: Provides consistent interface for both retrieval methods
3. **Shared Text Pipeline**: Both methods use identical preprocessing for fair comparison
4. **FAISS Optional**: Graceful fallback to NumPy if FAISS unavailable
5. **Model Choice**: `all-MiniLM-L6-v2` selected for speed/quality balance (384-dim, fast)
6. **TF-IDF Configuration**: max_features=20000, ngram_range=(1,2) for good coverage
7. **Streamlit for UI**: Rapid prototyping with components library for reusability
8. **Python Scripts**: Standalone utilities for data processing and benchmarking
9. **MS MARCO Dataset**: Standard benchmark dataset for eval, 1500 samples for demo
10. **Vietnamese Comments**: Project code includes Vietnamese comments for accessibility

##  Error Recovery

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FAISS not available` | Install: `pip install faiss-cpu` |
| `Dataset not found` | Run `python scripts/download_data.py` |
| `Embedding model fails` | Check internet connection |
| `Out of memory` | Reduce `NUM_PASSAGES` in .env |

## 🧪 Testing

## 🧪 Quick Test Commands

```bash
# Test TF-IDF retriever
python -c "from core.engine import SearchEngine; e=SearchEngine(); e.prepare(); result=e.search_tfidf('machine learning', 3); print(f'Top result: {result[\"results\"][0].text[:80]}...')"

# Test Semantic retriever
python -c "from core.engine import SearchEngine; e=SearchEngine(); e.prepare(); result=e.search_semantic('AI techniques', 3); print(f'Top result: {result[\"results\"][0].text[:80]}...')"

# Compare both methods
python -c "from core.engine import SearchEngine; e=SearchEngine(); e.prepare(); comp=e.compare('neural networks', 5); print(f'TF-IDF: {comp[\"baseline\"][\"inference_ms\"]:.2f}ms, Semantic: {comp[\"advanced\"][\"inference_ms\"]:.2f}ms')"

# Run evaluation
python scripts/evaluate.py --top-k 5

# Run benchmark
python scripts/benchmark.py --sample-sizes 1000 1500
```