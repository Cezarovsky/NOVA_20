# NOVA AI System

**Multi-Context AI Assistant cu arhitectură multi-agent și focus pe ML fundamentals**

## 📋 Overview

NOVA este un sistem AI modular care procesează și sintetizează informații din multiple surse folosind:
- 🧠 **Deep Learning Core**: Transformer architecture, Attention mechanisms, KV cache
- 🤖 **Multi-Agent System**: Agenți specializați coordonați de orchestrator
- 🔒 **Privacy-First**: Anthropic Claude + Mistral (fără OpenAI)
- 📊 **Vector Database**: ChromaDB pentru RAG (Retrieval-Augmented Generation)

## 🏗️ Architecture

```
Nova_20/
├── src/
│   ├── ml/                    # Machine Learning Core (PRIORITY)
│   │   ├── attention.py       # Multi-Head Attention
│   │   ├── transformer.py     # Transformer blocks
│   │   ├── embeddings.py      # Token & positional embeddings
│   │   ├── inference.py       # Inference engine + KV cache
│   │   ├── sampling.py        # Top-k, top-p sampling
│   │   └── optimization.py    # Performance optimizations
│   ├── rag/                   # RAG System (NEW!)
│   │   ├── embeddings.py      # Multi-strategy embeddings
│   │   ├── vector_store.py    # ChromaDB integration
│   │   ├── chunker.py         # Smart document chunking
│   │   ├── retriever.py       # Semantic search + re-ranking
│   │   ├── memory.py          # Multi-tier memory
│   │   └── rag_pipeline.py    # Complete RAG orchestration
│   ├── voice/                 # Voice Module
│   │   └── tts.py             # Text-to-speech (pyttsx3)
│   ├── core/
│   │   ├── llm_interface.py   # API-based LLM interface
│   │   └── vector_store.py    # ChromaDB wrapper
│   ├── agents/                # Multi-agent system
│   │   ├── base_agent.py
│   │   └── orchestrator.py
│   ├── utils/                 # Utilities
│   └── ui/                    # Streamlit interface
├── examples/                  # Demo scripts
│   ├── voice_demo.py          # TTS demonstrations
│   └── rag_demo.py            # RAG demonstrations
├── tests/                     # Unit & integration tests
├── data/                      # Vector DB & cache
├── logs/                      # Application logs
└── Documentation/             # Architecture & roadmap
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone/navigate to project
cd Nova_20

# Activate virtual environment
source venv/bin/activate

# Install dependencies (already done)
# pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configure API Keys

Edit `.env` file:
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
MISTRAL_API_KEY=your-mistral-key-here
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_ml/

# With coverage
pytest --cov=src tests/
```

### 4. Launch Application

```bash
# Run Streamlit UI
streamlit run src/ui/streamlit_app.py

# Or run Python scripts directly
python src/ml/attention.py

# Run Voice Demo
python examples/voice_demo.py

# Run RAG Demo (comprehensive test suite)
python examples/rag_demo.py
```

## 📚 Documentation

- **Architecture**: `Documentation/arhitectura_nova.md` - Comprehensive technical architecture
- **Implementation Roadmap**: `Documentation/implementation.md` - Step-by-step development plan (3600+ lines)
- **RAG Implementation**: `RAG_IMPLEMENTATION.md` - Complete RAG system documentation
- **API Docs**: Generate with `pdoc src/` (coming soon)

## 🧪 Key Features

### 🧠 RAG System (NEW!)
- ✅ **Advanced Retrieval-Augmented Generation**
- ✅ **3 Embedding Strategies**: Sentence transformers, NOVA custom, hybrid
- ✅ **ChromaDB Integration**: Persistent vector storage + multi-collection
- ✅ **Smart Chunking**: 5 strategies (fixed/sentence/paragraph/code/smart)
- ✅ **Semantic Search**: Re-ranking + MMR diversity algorithm
- ✅ **Multi-tier Memory**: Conversation (short), Knowledge (long), Working (context)
- ✅ **Multilingual**: Romanian + English support
- ✅ **Production Ready**: 2,400+ lines with comprehensive demos

### 🎙️ Voice Module
- ✅ **Text-to-Speech**: Offline TTS cu pyttsx3
- ✅ **Feminine Voice**: Elegant și expresivă
- ✅ **Multilingual**: Suport pentru română și engleză
- ✅ **Customizable**: Rate, volume, voice selection
- ✅ **5 Demo Scenarios**: Basic TTS, Romanian, English, customization, context-aware

### Deep Learning Core
- ✅ **Scaled Dot-Product Attention** cu causal masking
- ✅ **Multi-Head Attention** pentru parallel processing
- ✅ **KV Cache** pentru inference optimization (10-100x speedup)
- ✅ **Top-K & Top-P Sampling** pentru text generation quality
- ✅ **Sinusoidal & Learned Positional Encodings**
- ✅ **Feed-Forward Networks** cu GELU activation
- ✅ **Complete Transformer Encoder/Decoder** layers

### Inference Optimization
- **KV Cache**: Reduce complexity de la O(n²) la O(n)
- **Batch Processing**: Process multiple sequences în paralel
- **Memory Estimation**: Plan ahead pentru large models
- **Top-K/Top-P**: Balanced quality vs. diversity

### Multi-Modal Processing
- 📄 **Documents**: PDF, DOCX, TXT cu semantic chunking
- 🖼️ **Vision**: Image analysis cu Claude Vision (+ OCR)
- 🎵 **Audio**: Transcription cu Faster-Whisper (local)
- 🌐 **Web**: Scraping cu BeautifulSoup

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| LLM | Anthropic Claude | 3.5 Sonnet |
| LLM (Alt) | Mistral AI | Large/Small |
| Embeddings | Sentence Transformers | Multi-MiniLM-L12 |
| Embeddings (Alt) | Mistral Embed | 1024D |
| Vector DB | ChromaDB | 1.3.5 |
| Deep Learning | PyTorch | 2.0+ |
| TTS | pyttsx3 | 2.90 |
| PDF | PyPDF2 | 3.0.1 |
| UI | Streamlit | 1.39.0+ |
| Audio | Faster-Whisper | 1.0.0+ |
| Testing | pytest | Latest |

## 📊 Development Status

**Current Phase**: Advanced Features - RAG + Voice Integration (50% complete)

### Completed ✅
- [x] Project structure setup
- [x] Virtual environment & dependencies
- [x] Documentation (architecture + roadmap)
- [x] Environment configuration
- [x] **ML Core**: Complete transformer, attention, embeddings
- [x] **Training Pipeline**: Full training with mixed precision
- [x] **Validation & Metrics**: Comprehensive evaluation
- [x] **Advanced Training**: LR scheduling, gradient clipping
- [x] **Data Pipeline**: Multi-source data loading
- [x] **Inference Engine**: KV cache, beam search, streaming
- [x] **Voice Module**: TTS with pyttsx3, multilingual support
- [x] **RAG System**: Advanced retrieval with ChromaDB (2,400+ lines)

### In Progress 🔄
- [ ] Integration of RAG with main NOVA model
- [ ] Web interface with voice + RAG
- [ ] End-to-end conversational system

### Next Steps ⏳
- [ ] Multi-agent orchestration with RAG
- [ ] Advanced memory management
- [ ] Production deployment
- [ ] Performance optimization

## 🧑‍💻 Development

### Code Style
```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Testing
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/test_ml/test_attention.py -v

# Run with debugging
pytest tests/ -vv --pdb
```

### Performance Profiling
```bash
# Profile inference
python -m cProfile -o profile.stats src/ml/inference.py

# View with snakeviz
snakeviz profile.stats
```

## 📈 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Query Response Time | < 5s | ⏳ |
| Document Processing | < 30s | ⏳ |
| KV Cache Speedup | 10x+ | ⏳ |
| Test Coverage | > 80% | ⏳ |
| Memory Usage | < 8GB | ⏳ |

## 🤝 Contributing

Acest proiect este în development activ. Pentru contribuții:

1. Citește `Documentation/implementation.md` pentru roadmap
2. Creează un branch nou pentru feature-ul tău
3. Scrie teste pentru codul nou
4. Asigură-te că toate testele trec
5. Creează un pull request

## 📝 License

Acest proiect este pentru uz educațional și cercetare.

## 🙏 Acknowledgments

- **Anthropic** - Claude 3.5 Sonnet API
- **Mistral AI** - Mistral Large & embeddings
- **Vaswani et al.** - "Attention Is All You Need" (2017)
- **OpenAI** (research only) - GPT architecture inspiration

## 📞 Contact

Pentru întrebări despre arhitectură sau implementare, consultă documentația din `Documentation/`.

---

**Status**: 🚀 Active Development | **Version**: 0.5.0-beta | **Last Updated**: Dec 2024

**Latest Features**: RAG System (commit `650b1be`) + Voice Module (commit `01ed670`)
