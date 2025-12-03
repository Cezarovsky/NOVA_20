# Curs NOVA: De la Teorie la Practică

**Versiune**: 1.0  
**Data**: 3 Decembrie 2024  
**Autor**: NOVA Development Team

---

## Cuprins

1. [Fundamente Matematice](#1-fundamente-matematice)
2. [Arhitectura Transformer](#2-arhitectura-transformer)
3. [Training și Optimizare](#3-training-și-optimizare)
4. [Inference și Generare](#4-inference-și-generare)
5. [RAG și Memory Systems](#5-rag-și-memory-systems)
6. [Proiect Final](#6-proiect-final)

---

## 1. Fundamente Matematice

### 1.1 Vectori și Embeddings

**Teorie**: Text → Numere (vectori în spațiu N-dimensional)

**Matematică**:
```
Embedding: word → vector ∈ ℝᵈ
"cat" → [0.2, -0.5, 0.8, ..., 0.1]  (d=512)

Similaritate Cosine:
sim(A,B) = (A·B) / (||A|| × ||B||)
```

**Practică NOVA**:
```python
from src.ml.embeddings import NovaEmbeddings

# Create embeddings
embeddings = NovaEmbeddings(d_model=512, vocab_size=50000)

# Convert word IDs to vectors
word_ids = torch.tensor([42, 137, 891])  # [cat, is, cute]
vectors = embeddings(word_ids)  # Shape: (3, 512)

# Compute similarity
sim = torch.cosine_similarity(vectors[0], vectors[1], dim=0)
```

**Exercițiu**: Calculează similaritatea între "NOVA" și "AI assistant"

---

### 1.2 Matrici și Transformări Liniare

**Teorie**: Matricile transformă spații vectoriale

**Matematică**:
```
Y = XW + b
X: input (batch, seq_len, d_model)
W: weight matrix (d_model, d_model)
Y: output (batch, seq_len, d_model)
```

**Practică NOVA**:
```python
import torch.nn as nn

# Linear transformation in NOVA
linear = nn.Linear(512, 512)

# Forward pass
x = torch.randn(2, 10, 512)  # batch=2, seq=10, dim=512
y = linear(x)  # Same shape, transformed space
```

**Exercițiu**: Implementează o matrice de proiecție Q, K, V

---

### 1.3 Funcții de Activare

**Teorie**: Introducerea non-linearității

**Matematică**:
```
ReLU(x) = max(0, x)
GELU(x) ≈ x·Φ(x)  (Gaussian Error Linear Unit)
Softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

**Practică NOVA**:
```python
import torch.nn.functional as F

x = torch.tensor([-2, -1, 0, 1, 2])

# ReLU
relu_output = F.relu(x)  # [0, 0, 0, 1, 2]

# GELU (used in NOVA)
gelu_output = F.gelu(x)

# Softmax (for attention)
probs = F.softmax(x, dim=0)  # Sums to 1.0
```

**Exercițiu**: Plot GELU vs ReLU pentru x ∈ [-5, 5]

---

## 2. Arhitectura Transformer

### 2.1 Self-Attention Mechanism

**Teorie**: Fiecare token "vede" relația cu toate celelalte

**Matematică**:
```
Q = XWq, K = XWk, V = XWv

Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V

Exemplu:
Query:  "cat"  → ce caut?
Key:    "cute" → ce ofer?
Value:  "cute" → ce informație am?

Score = Q·Kᵀ / √512 = măsură de relevanță
```

**Practică NOVA**:
```python
from src.ml.attention import MultiHeadAttention

# Initialize attention (8 heads, 512 dim)
attention = MultiHeadAttention(
    d_model=512,
    num_heads=8,
    dropout=0.1
)

# Input sequence
x = torch.randn(2, 10, 512)  # batch=2, seq=10

# Self-attention
output, attn_weights = attention(x, x, x)

# Visualize attention
# attn_weights: (2, 8, 10, 10)
# [batch, heads, query_pos, key_pos]
```

**Exercițiu**: Calculează attention scores pentru "NOVA is smart"

---

### 2.2 Multi-Head Attention

**Teorie**: Paralelizăm atenția pe mai multe "subspații"

**Matematică**:
```
headᵢ = Attention(QWqⁱ, KWkⁱ, VWvⁱ)
MultiHead = Concat(head₁,...,headₕ)Wₒ

h = 8 heads
dₖ = d_model / h = 512 / 8 = 64 per head
```

**Vizualizare**:
```
Input (512 dim)
    ↓
Split into 8 heads (64 dim each)
    ↓
[Head1] [Head2] ... [Head8]
  ↓       ↓           ↓
Attention Attention Attention
  ↓       ↓           ↓
Concat all heads
    ↓
Output projection (512 dim)
```

**Practică NOVA**:
```python
# Already implemented in attention layer above
# Each head learns different patterns:
# - Head 1: syntax (subject-verb)
# - Head 2: semantics (word meanings)
# - Head 3: long-range dependencies
# etc.
```

**Exercițiu**: Vizualizează cele 8 heads pentru o propoziție

---

### 2.3 Positional Encoding

**Teorie**: Injectăm informație despre poziție (Transformers nu au ordine nativă)

**Matematică**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos = poziția în secvență (0, 1, 2, ...)
i = dimensiunea (0 la d_model/2)
```

**Practică NOVA**:
```python
from src.ml.embeddings import PositionalEncoding

# Create positional encoding
pos_enc = PositionalEncoding(d_model=512, max_len=5000)

# Apply to embeddings
embedded = embeddings(word_ids)  # (batch, seq, 512)
embedded_with_pos = pos_enc(embedded)

# Positional patterns are added automatically
```

**Exercițiu**: Plot PE pentru primele 50 poziții și 512 dimensiuni

---

### 2.4 Feed-Forward Network

**Teorie**: MLP care procesează fiecare poziție independent

**Matematică**:
```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂

d_model = 512 → d_ff = 2048 → d_model = 512
Expansion factor: 4x
```

**Practică NOVA**:
```python
from src.ml.feedforward import PositionwiseFeedForward

# FFN layer
ffn = PositionwiseFeedForward(
    d_model=512,
    d_ff=2048,
    dropout=0.1
)

# Forward pass
x = torch.randn(2, 10, 512)
output = ffn(x)  # Same shape: (2, 10, 512)
```

**Exercițiu**: Calculează numărul de parametri în FFN

---

### 2.5 Layer Normalization

**Teorie**: Stabilizează training-ul prin normalizare

**Matematică**:
```
LayerNorm(x) = γ·(x - μ) / √(σ² + ε) + β

μ = mean(x)
σ² = variance(x)
γ, β = parametri învățați
```

**Practică NOVA**:
```python
import torch.nn as nn

# Layer normalization
layer_norm = nn.LayerNorm(512)

x = torch.randn(2, 10, 512)
normalized = layer_norm(x)

# Properties:
# mean ≈ 0, std ≈ 1 for each feature dimension
```

**Exercițiu**: Compară LayerNorm cu BatchNorm

---

### 2.6 Transformer Block Complet

**Teorie**: Combinăm toate componentele

**Arhitectură**:
```
Input
  ↓
+ MultiHeadAttention
  ↓
LayerNorm
  ↓
+ FeedForward
  ↓
LayerNorm
  ↓
Output
```

**Practică NOVA**:
```python
from src.ml.transformer import TransformerBlock

# Single transformer block
block = TransformerBlock(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
)

# Forward pass
x = torch.randn(2, 10, 512)
output = block(x, mask=None)
```

**Exercițiu**: Calculează receptive field după N blocks

---

## 3. Training și Optimizare

### 3.1 Loss Function

**Teorie**: Cross-Entropy pentru predicție next token

**Matematică**:
```
Loss = -Σᵢ yᵢ log(ŷᵢ)

y = one-hot true token
ŷ = predicted probabilities

Example:
True: "cat" (token 42)
Pred: [0.1, 0.05, ..., 0.7, ..., 0.02]  (50k vocab)
Loss = -log(0.7) = 0.357
```

**Practică NOVA**:
```python
import torch.nn.functional as F

# Predictions (batch=2, seq=10, vocab=50000)
logits = model(input_ids)

# True labels (shifted by 1 position)
targets = input_ids[:, 1:]  # Next token

# Compute loss
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    targets.reshape(-1)
)
```

**Exercițiu**: Calculează loss pentru "NOVA is [MASK]"

---

### 3.2 Adam Optimizer

**Teorie**: Adaptive learning rate per parametru

**Matematică**:
```
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ         (momentum)
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²        (variance)
θₜ = θₜ₋₁ - η·mₜ/√(vₜ + ε)      (update)

β₁ = 0.9, β₂ = 0.999
η = learning rate
```

**Practică NOVA**:
```python
from src.training.trainer import NovaTrainer

# Training configuration
config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10,
    warmup_steps=1000
)

trainer = NovaTrainer(model, tokenizer, config)
```

**Exercițiu**: Plot learning rate cu warmup

---

### 3.3 Learning Rate Scheduling

**Teorie**: Warmup + Decay pentru convergență

**Matematică**:
```
Warmup: lr(t) = lr_max · (t / warmup_steps)
Decay:  lr(t) = lr_max · √(d_model) / √(max(t, warmup))
```

**Practică NOVA**:
```python
# Automatically handled by trainer
trainer.train(train_dataset)

# LR schedule:
# Steps 0-1000: Linear warmup
# Steps 1000+: Inverse sqrt decay
```

**Exercițiu**: Plot LR pentru 10k steps

---

### 3.4 Gradient Clipping

**Teorie**: Previne exploding gradients

**Matematică**:
```
g_clipped = g · min(1, max_norm / ||g||)

Dacă ||g|| > max_norm (ex: 1.0):
  Scale down gradient
```

**Practică NOVA**:
```python
# In training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Exercițiu**: Monitorizează grad norm pe 100 steps

---

### 3.5 Training Loop Complet

**Practică NOVA**:
```python
from src.training.trainer import NovaTrainer
from src.data.dataset import NovaDataset

# 1. Prepare data
dataset = NovaDataset("train.txt", tokenizer, max_length=512)

# 2. Configure training
config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10,
    save_every=1000,
    eval_every=500
)

# 3. Train
trainer = NovaTrainer(model, tokenizer, config)
history = trainer.train(dataset)

# 4. Monitor
print(f"Final loss: {history['train_loss'][-1]:.4f}")
```

**Exercițiu**: Train pe 1000 samples, plot loss curve

---

## 4. Inference și Generare

### 4.1 Greedy Decoding

**Teorie**: Alege mereu token-ul cu probabilitate maximă

**Matematică**:
```
xₜ = argmax P(x|x₁,...,xₜ₋₁)
```

**Practică NOVA**:
```python
from src.ml.inference import NovaInference

inference = NovaInference(model, tokenizer)

# Greedy generation
output = inference.generate(
    "NOVA is",
    max_length=50,
    strategy="greedy"
)
# Output: "NOVA is an advanced AI assistant..."
```

**Exercițiu**: Generate 5 continuări pentru "The weather is"

---

### 4.2 Beam Search

**Teorie**: Explorează top-K căi în paralel

**Matematică**:
```
Score(sequence) = Σₜ log P(xₜ|x₁,...,xₜ₋₁)
Keep top K sequences at each step
```

**Practică NOVA**:
```python
output = inference.generate(
    "NOVA is",
    max_length=50,
    strategy="beam_search",
    num_beams=5
)
# More diverse and coherent output
```

**Exercițiu**: Compară greedy vs beam (5 beams)

---

### 4.3 Sampling cu Temperatură

**Teorie**: Controlează randomness

**Matematică**:
```
P'(x) = softmax(logits / T)

T < 1: mai deterministă (confident)
T = 1: distribuție originală
T > 1: mai aleatorie (creative)
```

**Practică NOVA**:
```python
# Conservative (T=0.5)
output1 = inference.generate(
    "Once upon a time",
    temperature=0.5,
    strategy="sampling"
)

# Creative (T=1.5)
output2 = inference.generate(
    "Once upon a time",
    temperature=1.5,
    strategy="sampling"
)
```

**Exercițiu**: Generate cu T ∈ {0.3, 0.7, 1.0, 1.5}, compară

---

### 4.4 Top-K și Top-P (Nucleus) Sampling

**Teorie**: Filtrează token-uri improbabile

**Matematică**:
```
Top-K: Păstrează K token-uri cu prob. cea mai mare
Top-P: Păstrează token-uri până prob. cumulativă >= P

P(nucleus) >= p (ex: 0.9)
```

**Practică NOVA**:
```python
# Top-K sampling
output = inference.generate(
    "NOVA can",
    strategy="top_k",
    top_k=50
)

# Top-P (nucleus) sampling
output = inference.generate(
    "NOVA can",
    strategy="top_p",
    top_p=0.9
)
```

**Exercițiu**: Generează 10 outputs, calculează diversity score

---

### 4.5 KV Cache pentru Vitează

**Teorie**: Cache-uim key/value pentru token-uri generate

**Matematică**:
```
Fără cache: O(n²) attention pentru n tokens
Cu cache:   O(n) doar pentru ultimul token

Speedup: ~10x pentru secvențe lungi
```

**Practică NOVA**:
```python
# KV cache is automatic in NOVA inference
output = inference.generate(
    "Long prompt here...",
    max_length=500,
    use_cache=True  # Default
)
# Generates 500 tokens ~10x faster
```

**Exercițiu**: Benchmark cu/fără cache pentru 100 tokens

---

## 5. RAG și Memory Systems

### 5.1 Embeddings pentru Retrieval

**Teorie**: Reprezentări dense pentru similaritate semantică

**Matematică**:
```
Cosine Similarity:
sim(q, d) = (q·d) / (||q|| × ||d||)

query: "What is NOVA?"
docs: ["NOVA is AI", "Python code", "Weather"]
scores: [0.89, 0.23, 0.15]
```

**Practică NOVA**:
```python
from src.rag.embeddings import SentenceTransformerEmbeddings

# Initialize embedder
embedder = SentenceTransformerEmbeddings()

# Embed query and documents
query_emb = embedder.embed_query("What is NOVA?")
doc_embs = embedder.embed_documents([
    "NOVA is an AI assistant",
    "Python programming",
    "Weather forecast"
])

# Compute similarities
similarities = [
    torch.cosine_similarity(query_emb, doc_emb, dim=0)
    for doc_emb in doc_embs
]
```

**Exercițiu**: Creează 5 docs, find top-3 pentru query

---

### 5.2 Vector Store cu ChromaDB

**Teorie**: Database pentru search semantic rapid

**Practică NOVA**:
```python
from src.rag.vector_store import ChromaVectorStore

# Initialize persistent store
store = ChromaVectorStore(
    collection_name="my_knowledge",
    persist_directory="./chroma_db"
)

# Add documents
docs = ["NOVA is an AI", "Python is great"]
embeddings = embedder.embed_documents(docs)
store.add(docs, embeddings)

# Search
query_emb = embedder.embed_query("Tell me about NOVA")
results = store.search(query_emb, n_results=3)
```

**Exercițiu**: Build knowledge base cu 20 facts despre NOVA

---

### 5.3 Document Chunking

**Teorie**: Împărțim texte lungi în fragmente semantice

**Practică NOVA**:
```python
from src.rag.chunker import DocumentChunker

chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50,
    strategy='smart'
)

# Chunk a document
text = open("long_article.txt").read()
chunks = chunker.chunk_text(text)

# Each chunk: 450-550 chars with context overlap
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} chars")
```

**Exercițiu**: Chunk o carte, count chunks, plot distribution

---

### 5.4 Complete RAG Pipeline

**Practică NOVA**:
```python
from src.rag.rag_pipeline import RAGPipeline

# Initialize RAG system
rag = RAGPipeline(collection_name="knowledge")

# Add knowledge from file
rag.add_file("documentation.pdf")

# Query with context retrieval
result = rag.query(
    "How does NOVA handle Romanian?",
    n_results=5
)

# result contains:
# - Retrieved documents
# - Assembled context
# - Source citations
```

**Exercițiu**: Build RAG system cu documentația NOVA, test queries

---

### 5.5 Memory-Augmented Conversations

**Practică NOVA**:
```python
# RAG + Conversation Memory
rag = RAGPipeline(collection_name="chat")

# Turn 1
context1 = rag.chat("Who created NOVA?")
response1 = model.generate(context1)
rag.add_assistant_response(response1)

# Turn 2 (remembers context)
context2 = rag.chat("What can it do?")  # "it" = NOVA
response2 = model.generate(context2)
```

**Exercițiu**: Conversație multi-turn cu 5 întrebări

---

## 6. Proiect Final

### 6.1 Build Your Own NOVA Assistant

**Obiectiv**: Chatbot cu RAG și Voice pentru domeniu specific

**Pași**:

1. **Prepare Knowledge Base**
```python
# Collect domain documents
docs = ["doc1.pdf", "doc2.txt", "doc3.md"]

# Build RAG system
rag = RAGPipeline("my_domain")
for doc in docs:
    rag.add_file(doc)
```

2. **Setup Voice**
```python
from src.voice.tts import NovaVoice
voice = NovaVoice()
```

3. **Create Chat Loop**
```python
while True:
    user_input = input("You: ")
    
    # Retrieve context
    context = rag.query(user_input, n_results=3)
    
    # Generate response
    response = inference.generate(
        context,
        max_length=200,
        temperature=0.7
    )
    
    # Speak response
    print(f"NOVA: {response}")
    voice.speak(response)
    
    # Update memory
    rag.add_assistant_response(response)
```

**Exercițiu Final**: Creează chatbot pentru un domeniu la alegere (medicină, drept, IT, etc.)

---

## Resurse și Bibliografie

**Papers**:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT" (Devlin et al., 2018)
- "GPT-3" (Brown et al., 2020)

**NOVA Documentation**:
- `NOVA_MANUAL.md`: Complete implementation guide
- `arhitectura_nova.md`: Technical architecture
- `RAG_IMPLEMENTATION.md`: RAG system details

**Practice**:
- `examples/`: 8+ demo scripts
- `tests/`: Unit tests for all components

---

## Următorii Pași

1. ✅ Complete cursul teoretic
2. 🔄 Implementează fiecare exercițiu
3. 🎯 Build proiect final
4. 🚀 Deploy în producție

**Mult succes!** 🎓
