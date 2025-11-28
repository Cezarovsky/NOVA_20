# Arhitectura Proiectului NOVA - Sistem AI pentru Analiza Multi-Context

## 1. VIZIUNE GENERALĂ

### 1.1 Filosofia de Design
Proiectul NOVA este conceput ca un sistem AI modular și extensibil care respectă principiile clasice ale inteligenței artificiale:
- **Separarea preocupărilor** (Separation of Concerns)
- **Modularitate și reutilizabilitate**
- **Transparență și explicabilitate**
- **Scalabilitate și performanță**
- **Învățare incrementală**

### 1.2 Obiectivul Principal
Crearea unui asistent AI capabil să proceseze, analizeze și sintetizeze informații din multiple surse (documente, imagini, audio, web) într-un mod inteligent și contextual.

---

## 2. ARHITECTURA LA NIVEL ÎNALT

### 2.1 Straturile Sistemului

```
┌─────────────────────────────────────────────────────────┐
│           INTERFAȚĂ UTILIZATOR (UI Layer)                │
│          (Streamlit - Conversație & Vizualizare)         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│        ORCHESTRATOR AGENT (Coordination Layer)           │
│     (Routing, Planning, Multi-Agent Coordination)        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│          SPECIALIZED AGENTS (Processing Layer)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Document │ │  Vision  │ │  Audio   │ │   Web    │  │
│  │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           KNOWLEDGE BASE (Memory Layer)                  │
│     (Vector Store - ChromaDB cu Embeddings)              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│         TOOLS & UTILITIES (Infrastructure)               │
│  (File I/O, API Calls, Data Processing, Logging)        │
└─────────────────────────────────────────────────────────┘
```

---

## 3. PRINCIPII CLASICE DE AI IMPLEMENTATE

### 3.1 Reprezentarea Cunoștințelor (Knowledge Representation)
**Principiu clasic**: Sistemele AI trebuie să reprezinte cunoștințele într-un formatstructurat și interogabil.

**Implementare în NOVA**:
- **Vector Embeddings**: Transformarea textului în reprezentări vectoriale dense (1024 dimensiuni cu Mistral)
- **Metadata Structurată**: Fiecare chunk de informație are metadate (sursă, tip, timestamp, context)
- **Semantic Search**: Căutare bazată pe similaritate semantică, nu doar keyword matching
- **Persistent Memory**: Bază de date ChromaDB pentru păstrarea cunoștințelor între sesiuni

```python
# Exemplu de reprezentare
{
    "text": "Content extractat...",
    "embedding": [0.123, -0.456, ...],  # Vector 1024D (Mistral embed)
    "metadata": {
        "source": "document.pdf",
        "type": "document",
        "page": 5,
        "timestamp": "2025-11-28",
        "context": "Financial analysis"
    }
}
```

### 3.2 Reasoning & Planning (Raționament și Planificare)
**Principiu clasic**: AI-ul trebuie să poată planifica secvențe de acțiuni pentru a atinge obiective complexe.

**Implementare în NOVA**:
- **Chain-of-Thought Reasoning**: LLM-ul descompune probleme complexe în pași logici
- **Task Decomposition**: Orchestrator-ul descompune query-uri complexe în sub-task-uri
- **Multi-Step Processing**: Fiecare agent poate executa secvențe de operații
- **Context Awareness**: Deciziile se bazează pe istoricul conversației

```python
# Exemplu de planning
User Query: "Analizează acest PDF și compară cu informațiile din imagine"
↓
Orchestrator Planning:
1. Detectează tipuri de input (PDF + Image)
2. Rutează PDF → Document Agent
3. Rutează Image → Vision Agent
4. Așteaptă rezultatele
5. Sintetizează informațiile
6. Generează răspuns unificat
```

### 3.3 Learning & Adaptation (Învățare și Adaptare)
**Principiu clasic**: Sistemele AI trebuie să învețe din experiență și să se adapteze.

**Implementare în NOVA**:
- **Incremental Learning**: Noi documente se adaugă în knowledge base fără a șterge cele vechi
- **Context Accumulation**: Fiecare conversație îmbogățește memoria sistemului
- **Feedback Loop**: Sistemul poate fi îmbunătățit pe baza interacțiunilor
- **Transfer Learning**: Folosim modele pre-antrenate (Anthropic Claude, Mistral, Llama) care au învățat din miliarde de exemple

### 3.4 Multi-Modal Processing (Procesare Multi-Modală)
**Principiu clasic**: Inteligența umană procesează multiple tipuri de informații simultan.

**Implementare în NOVA**:
- **Vision Models**: Claude 3.5 Sonnet (vision capabilities) sau LLaVA (open-source) pentru analiza imaginilor
- **Text Processing**: Claude 3.5 Sonnet / Mistral Large pentru text și documente
- **Audio Processing**: Faster-Whisper (open-source) sau Whisper local pentru transcrierea audio
- **Web Scraping**: Extracție și procesare de conținut web
- **Unified Representation**: Toate modalitățile sunt convertite în embeddings comparabile

### 3.5 Explainability (Explicabilitate)
**Principiu clasic**: Sistemele AI trebuie să-și poată explica deciziile.

**Implementare în NOVA**:
- **Source Citation**: Fiecare răspuns citează sursele folosite
- **Chain of Reasoning**: Afișăm pașii intermediari ai raționamentului
- **Confidence Scores**: Similarity scores pentru relevanța surselor
- **Transparent Processing**: Logging detaliat al tuturor operațiilor

---

## 4. FUNDAMENTELE TEHNICE: DEEP LEARNING ȘI TRANSFORMERS

### 4.1 Arhitectura Transformer - Inima Sistemului

#### 4.1.1 De Ce Transformers?
Toate modelele moderne de limbaj (Claude, Mistral, Llama) se bazează pe arhitectura **Transformer** (Vaswani et al., 2017). Aceasta a revoluționat NLP-ul prin înlocuirea RNN-urilor și LSTM-urilor cu un mecanism mai eficient: **Attention**.

**Avantaje față de arhitecturi clasice**:
- **Paralelizare**: Poate procesa toate token-urile simultan (vs. secvențial în RNN)
- **Long-range dependencies**: Capturează relații între cuvinte distante
- **Scalabilitate**: Performanța crește cu dimensiunea modelului și datele
- **Transfer learning**: Pre-training + fine-tuning funcționează excepțional

#### 4.1.2 Mecanismul de Attention

**Intuiție**: Când procesăm un cuvânt, attention-ul ne spune "la ce alte cuvinte să ne uităm".

**Formula matematică**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Unde:
- **Q** (Query): "Ce caut?" - reprezentarea cuvântului curent
- **K** (Key): "Ce ofer?" - reprezentările tuturor cuvintelor
- **V** (Value): "Ce informație transmit?" - conținutul semantic
- $d_k$: dimensiunea key-urilor (pentru stabilizare numerică)

**Multi-Head Attention**:
```python
# Conceptual implementation
class MultiHeadAttention:
    def __init__(self, d_model=1024, num_heads=16):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64 per head
        
        # Linear projections pentru Q, K, V
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 1. Linear projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 3. Scaled Dot-Product Attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # 4. Concatenate heads și linear projection
        output = attention_output.view(batch_size, seq_len, d_model)
        return self.W_o(output)
```

**De ce Multiple Heads?**
- Fiecare head învață să se concentreze pe aspecte diferite (sintaxă, semantică, dependențe)
- Head 1: poate învăța subiect-verb agreement
- Head 2: poate învăța relații anafore (pronume → referent)
- Head 3: poate învăța relații semantice

**Self-Attention vs. Cross-Attention**:
- **Self-Attention**: Q, K, V vin din aceeași secvență (encoder/decoder)
- **Cross-Attention**: Q vine din decoder, K și V din encoder (folosit în traducere)

#### 4.1.3 Tipuri de Modele Transformer

##### A. Autoregressive Language Models (Causal LM)
**Arhitectură**: Doar decoder cu causal masking

**Reprezentanți**: GPT-3/4, Claude, Llama, Mistral

**Mecanism**:
```python
# Causal masking - token-ul curent vede doar trecutul
# Attention mask pentru secvența "The cat sat"
mask = [
    [1, 0, 0],  # "The" vede doar "The"
    [1, 1, 0],  # "cat" vede "The cat"
    [1, 1, 1],  # "sat" vede "The cat sat"
]
```

**Training Objective**: Next-token prediction
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

**Generare text**:
```python
def generate_autoregressive(prompt, model, max_tokens=100):
    tokens = tokenize(prompt)
    for _ in range(max_tokens):
        # Forward pass prin model
        logits = model(tokens)  # (seq_len, vocab_size)
        
        # Probabilități pentru următorul token
        next_token_logits = logits[-1, :]  # Ultimul token
        probs = softmax(next_token_logits / temperature)
        
        # Sampling (sau greedy: argmax)
        next_token = sample(probs)
        tokens.append(next_token)
        
        if next_token == EOS_TOKEN:
            break
    
    return detokenize(tokens)
```

**Caracteristici**:
✅ Excelent pentru generare creativă
✅ Conversație naturală
✅ Few-shot learning prin prompting
❌ Nu poate "vedea viitorul" (unidirectional)

##### B. Masked Language Models (Bidirectional)
**Arhitectură**: Doar encoder cu bidirectional attention

**Reprezentanți**: BERT, RoBERTa, ALBERT

**Mecanism**:
```python
# Training: Masking aleator 15% din tokens
# Input: "The cat sat on the [MASK]"
# Target: predict "mat"

# Bidirectional attention - fiecare token vede tot contextul
mask = [
    [1, 1, 1, 1, 1],  # Fiecare rând: toate token-urile se văd între ele
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
]
```

**Training Objective**: Masked Language Modeling (MLM)
$$\mathcal{L} = -\sum_{t \in \text{masked}} \log P(x_t | x_{\backslash t})$$

**Caracteristici**:
✅ Înțelegere profundă a contextului (bidirectional)
✅ Excelent pentru embeddings și clasificare
✅ Transfer learning pentru NLU tasks
❌ Nu generează text natural (nu e autoregressive)

##### C. Encoder-Decoder Models
**Arhitectură**: Encoder (bidirectional) + Decoder (autoregressive)

**Reprezentanți**: T5, BART, mT5

**Use cases**: Traducere, summarization, question answering

**Nu folosim în NOVA** (overkill pentru task-urile noastre), dar important de menționat.

#### 4.1.4 Embeddings și Reprezentări Vectoriale

**De la Cuvinte la Vectori**:

**1. Tokenization**:
```python
# Byte-Pair Encoding (BPE) sau SentencePiece
text = "Transformers are amazing!"
tokens = tokenizer.encode(text)
# Output: [8291, 388, 389, 6655, 0]  # IDs în vocabular
```

**2. Token Embeddings**:
```python
class TokenEmbedding:
    def __init__(self, vocab_size=50000, d_model=1024):
        # Lookup table: fiecare token ID → vector dense
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, token_ids):
        # token_ids: (batch, seq_len)
        return self.embedding(token_ids)  # (batch, seq_len, d_model)
```

**3. Positional Encoding**:
Transformers nu au noțiune de ordine → adăugăm informație pozițională.

**Sinusoidal Encoding** (BERT, GPT):
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Learned Positional Embeddings** (mai comun acum):
```python
self.position_embedding = nn.Embedding(max_seq_len, d_model)
```

**4. Final Input Representation**:
$$\text{Input} = \text{TokenEmb} + \text{PositionEmb} + \text{SegmentEmb (optional)}$$

**Proprietăți ale Embedding Space**:
- **Semantic similarity**: Cuvinte similare → vectori apropiați
- **Analogii**: "king" - "man" + "woman" ≈ "queen" (mai puțin robust în practice)
- **High-dimensional**: 1024-4096 dimensiuni pentru capturare nuanțe

#### 4.1.5 Feed-Forward Networks și Layer Normalization

**Position-wise Feed-Forward**:
```python
class FeedForward:
    def __init__(self, d_model=1024, d_ff=4096):
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()  # Gaussian Error Linear Unit
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.linear2(self.activation(self.linear1(x)))
```

**De ce d_ff = 4 × d_model?**
- Expansiune → compresie permite învățarea transformărilor non-liniare complexe
- "Bottleneck" architecture pentru regularizare

**Layer Normalization** (crucial pentru training stability):
$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

```python
# Pre-LN (mai stabil, folosit în GPT-3+)
def transformer_block(x):
    # Self-attention cu residual
    x = x + self_attention(layer_norm(x))
    # FFN cu residual
    x = x + feed_forward(layer_norm(x))
    return x
```

**Residual Connections**: Esențiale pentru training modele deep (100+ layers)

#### 4.1.6 Scaling Laws și Dimensiuni Practice

**Parametrii unui Model**:
```python
# Pentru un model Transformer autoregressive
num_params = (
    vocab_size * d_model +           # Token embeddings
    max_seq_len * d_model +          # Positional embeddings
    num_layers * (
        4 * d_model^2 +              # Attention (Q, K, V, O projections)
        2 * d_model * d_ff           # FFN (up + down projections)
    ) +
    d_model * vocab_size             # Output projection
)

# Aproximare: num_params ≈ 12 * num_layers * d_model^2
```

**Exemple Concrete**:
| Model | Layers | d_model | Heads | Params | Context |
|-------|--------|---------|-------|--------|---------|
| GPT-2 Small | 12 | 768 | 12 | 117M | 1024 |
| GPT-3 | 96 | 12288 | 96 | 175B | 2048 |
| Llama 3.1 405B | 126 | 16384 | 128 | 405B | 128K |
| Claude 3.5 | ~100 | ~16000 | ~128 | ~200B* | 200K |
| Mistral Large | ~80 | ~12000 | ~96 | ~123B | 128K |

*Estimări (nu sunt publice oficial)

**Scaling Laws** (Kaplan et al., 2020):
$$\text{Loss} \propto N^{-0.076} \cdot D^{-0.095} \cdot C^{-0.050}$$

Unde: N = params, D = data, C = compute

**Insight**: Performanța crește predictibil cu scaling, dar cu diminishing returns.

---

## 5. ARHITECTURA TEHNICĂ DETALIATĂ - IMPLEMENTARE NOVA

### 5.1 Layer 1: User Interface (Streamlit)

**Responsabilități**:
- Primirea input-urilor de la utilizator (text, fișiere, URL-uri)
- Afișarea răspunsurilor într-un format conversațional
- Gestionarea sesiunii și istoricului chat-ului
- Upload și preview pentru fișiere multiple

**Componente tehnice**:
```python
streamlit                  # Framework UI
st.session_state          # Persistență sesiune
st.file_uploader          # Upload fișiere
st.chat_message           # Display conversații
anthropic                  # Anthropic Claude API
mistralai                  # Mistral API (embeddings + LLM)
```

**Flow**:
1. User introduce query + fișiere opționale
2. UI validează input-urile
3. UI trimite request către Orchestrator
4. UI primește și afișează răspunsul progresiv

---

### 4.2 Layer 2: Orchestrator Agent (Coordination)

**Responsabilități**:
- Analizarea intent-ului utilizatorului
- Routing inteligent către agenți specializați
- Coordonarea execuției multi-agent
- Agregarea și sintetizarea rezultatelor
- Gestionarea context-ului conversațional

**Algoritm de Routing**:
```python
def route_query(query: str, files: List[File]) -> List[Agent]:
    """
    Routing bazat pe:
    1. Tipul fișierelor (PDF → DocAgent, JPG → VisionAgent)
    2. Keyword-uri în query (web, imagine, audio)
    3. Context istoric
    4. LLM-assisted routing pentru cazuri ambigue
    """
    agents_to_call = []
    
    # Routing bazat pe fișiere
    if has_documents(files):
        agents_to_call.append(DocumentAgent)
    if has_images(files):
        agents_to_call.append(VisionAgent)
    if has_audio(files):
        agents_to_call.append(AudioAgent)
    
    # Routing bazat pe query
    if "web" in query or "search" in query:
        agents_to_call.append(WebAgent)
    
    return agents_to_call
```

**Synthesis Strategy**:
```python
def synthesize_results(agent_results: Dict[str, str]) -> str:
    """
    Sinteză multi-sursă:
    1. Colectează toate rezultatele agenților
    2. Extrage informații relevante din vector DB
    3. Folosește LLM pentru a crea un răspuns coerent
    4. Include citări și surse
    """
    context = merge_contexts(agent_results)
    relevant_chunks = vector_db.search(query, context)
    
    final_response = llm.generate(
        prompt=synthesis_prompt,
        context=context,
        sources=relevant_chunks
    )
    
    return final_response
```

---

### 4.3 Layer 3: Specialized Agents

#### 4.3.1 Document Agent
**Specializare**: Procesarea documentelor text (PDF, DOCX, TXT, MD)

**Pipeline de procesare**:
```
Input: Fișier document
↓
1. EXTRACȚIE TEXT
   - PyMuPDF pentru PDF
   - python-docx pentru DOCX
   - Plain read pentru TXT/MD
↓
2. CHUNKING INTELIGENT
   - Split pe paragrafe/secțiuni
   - Chunk size: 1000 tokens
   - Overlap: 200 tokens (pentru context)
↓
3. EMBEDDING GENERATION
   - Model: mistral-embed (Mistral AI)
   - Dimensiuni: 1024
   - Normalizare: L2
↓
4. STORAGE
   - ChromaDB collection: "documents"
   - Metadata: {source, page, section, timestamp}
↓
5. RETRIEVAL
   - Semantic search cu query embedding
   - Top-k rezultate (k=5)
   - Re-ranking opțional
↓
Output: Text relevant + metadata
```

**Tehnici avansate**:
- **Semantic Chunking**: Folosim modele de sentence similarity pentru a grupa propoziții coerente
- **Hierarchical Indexing**: Creăm și un index la nivel de secțiune pentru overview
- **Citation Tracking**: Păstrăm exact pagina și poziția pentru fiecare chunk

**Deep Dive: Embeddings și Retrieval**

**1. Procesul de Embedding**:
```python
def create_embedding(text: str) -> np.ndarray:
    """
    Transformare text → vector semantic (1024D cu Mistral)
    
    Internal flow în mistral-embed:
    1. Tokenization (SentencePiece BPE)
    2. Token embeddings + positional encodings
    3. Multi-layer Transformer encoder (bidirectional)
    4. Mean pooling peste token embeddings
    5. L2 normalization
    """
    # API call
    response = mistral_client.embeddings(
        model="mistral-embed",
        input=[text]
    )
    embedding = np.array(response.data[0].embedding)
    
    # Normalizare pentru cosine similarity optimizată
    embedding = embedding / np.linalg.norm(embedding)
    return embedding  # Shape: (1024,)
```

**2. Semantic Search și Similarity**:
```python
def semantic_search(query: str, top_k: int = 5) -> List[Document]:
    """
    Cosine similarity în spațiul embedding
    
    Pentru vectori normalizați: cos(θ) = dot_product
    """
    query_embedding = create_embedding(query)
    
    # ChromaDB folosește HNSW (Hierarchical Navigable Small World)
    # Complexity: O(log N) vs O(N) brute-force
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Distance = 1 - cosine_similarity (ChromaDB convention)
    # Similarity = 1 - distance
    for i, dist in enumerate(results['distances'][0]):
        similarity = 1 - dist
        print(f"Result {i+1}: similarity = {similarity:.3f}")
    
    return results
```

**3. Advanced Chunking Strategy**:
```python
class IntelligentChunker:
    """
    Chunking semantic-aware cu overlap și context preservation
    """
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def semantic_chunk(self, text: str) -> List[Chunk]:
        """
        Chunking bazat pe similaritate semantică între propoziții
        """
        sentences = self.split_sentences(text)
        embeddings = self.sentence_model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0].split())
        
        for i in range(1, len(sentences)):
            # Similaritate cu chunk-ul curent
            chunk_emb = np.mean(embeddings[i-len(current_chunk):i], axis=0)
            similarity = cosine_similarity([chunk_emb], [embeddings[i]])[0][0]
            
            sent_size = len(sentences[i].split())
            
            # Adaugă la chunk dacă: similar semantic SAU chunk prea mic
            if (similarity > 0.7 and current_size + sent_size <= self.chunk_size) \
               or current_size < 100:
                current_chunk.append(sentences[i])
                current_size += sent_size
            else:
                # Finalizează chunk-ul curent
                chunks.append(self._create_chunk(current_chunk))
                
                # Start new chunk cu overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else []
                current_chunk = overlap_sentences + [sentences[i]]
                current_size = sum(len(s.split()) for s in current_chunk)
        
        chunks.append(self._create_chunk(current_chunk))
        return chunks
```

**4. RAG cu Re-ranking**:
```python
def rag_with_reranking(query: str, k: int = 5) -> str:
    """
    Retrieval-Augmented Generation cu cross-encoder re-ranking
    
    Pipeline:
    1. Bi-encoder (embeddings): retrieval rapid (top 20)
    2. Cross-encoder: re-ranking precis (top 5)
    3. LLM generation cu context optimizat
    """
    # Step 1: Fast retrieval cu bi-encoder
    candidates = semantic_search(query, top_k=20)
    
    # Step 2: Re-ranking cu cross-encoder (query + doc împreună)
    # Cross-encoder e mai lent dar mult mai precis
    reranked = []
    for doc in candidates['documents'][0]:
        # Model type: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        score = cross_encoder.predict([(query, doc)])
        reranked.append((score, doc))
    
    # Sort by score și păstrează top-k
    reranked.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in reranked[:k]]
    
    # Step 3: Context construction cu pozițional bias mitigation
    # LLM-urile sunt mai atenți la început și sfârșit
    context = reorder_for_attention(top_docs)
    
    # Step 4: Generation
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    return llm.generate(prompt)

def reorder_for_attention(docs: List[str]) -> str:
    """
    Mitigare 'Lost in the Middle' phenomenon
    Pune cele mai relevante docs la început și sfârșit
    """
    if len(docs) <= 2:
        return "\n\n".join(docs)
    
    # Pattern: [Best, 3rd, 5th, ..., 4th, 2nd]
    reordered = [docs[0]]  # Most relevant
    reordered.extend(docs[2::2])  # Odd positions
    reordered.extend(docs[3::2][::-1])  # Even reversed
    reordered.append(docs[1])  # Second best at end
    
    return "\n\n".join(reordered)
```

**5. Hybrid Search (Dense + Sparse)**:
```python
def hybrid_search(query: str, alpha: float = 0.7) -> List[Document]:
    """
    Combină semantic search (dense) cu BM25 (sparse)
    
    Dense: bun pentru semantic similarity
    Sparse: bun pentru exact keyword matches
    """
    # Dense retrieval (vector similarity)
    dense_results = vector_db.search(embed(query), n_results=20)
    dense_scores = {doc.id: doc.score for doc in dense_results}
    
    # Sparse retrieval (BM25 keyword matching)
    # BM25: Improved TF-IDF cu document length normalization
    bm25_results = bm25_index.search(query, top_k=20)
    bm25_scores = {doc.id: doc.score for doc in bm25_results}
    
    # Score fusion (RRF = Reciprocal Rank Fusion)
    all_doc_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
    hybrid_scores = {}
    
    for doc_id in all_doc_ids:
        # Normalize scores [0, 1]
        dense_norm = dense_scores.get(doc_id, 0) / max(dense_scores.values())
        bm25_norm = bm25_scores.get(doc_id, 0) / max(bm25_scores.values())
        
        # Weighted combination
        hybrid_scores[doc_id] = alpha * dense_norm + (1 - alpha) * bm25_norm
    
    # Return top-k
    sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return [get_document(doc_id) for doc_id, _ in sorted_docs[:5]]
```

#### 4.3.2 Vision Agent
**Specializare**: Analiza imaginilor și extracția informațiilor vizuale

**Deep Learning pentru Computer Vision în Context AI Agentic**

**1. Vision Transformers (ViT) - Arhitectura Modernă**

Claude 3.5 Sonnet și alte modele vision moderne folosesc **Vision Transformers**, nu CNN-uri clasice:

```python
class VisionTransformer:
    """
    Conceptual implementation a ViT
    
    Key idea: Tratează imaginea ca o secvență de patch-uri
    Similar cu tokens în NLP!
    """
    def __init__(self, img_size=224, patch_size=16, d_model=768, num_layers=12):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 14x14 = 196 patches
        
        # Linear projection pentru patch embeddings
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, d_model)
        
        # Positional embeddings (2D aware)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        # [CLS] token pentru classification/pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Standard Transformer encoder
        self.transformer = TransformerEncoder(d_model, num_layers, num_heads=12)
        
    def forward(self, image):
        # image: (batch, 3, 224, 224)
        batch_size = image.shape[0]
        
        # 1. Split image în patches
        # (batch, 3, 224, 224) → (batch, 196, 16*16*3)
        patches = self.split_into_patches(image)
        
        # 2. Linear projection: patches → embeddings
        # (batch, 196, 16*16*3) → (batch, 196, 768)
        patch_embeddings = self.patch_embedding(patches)
        
        # 3. Adaugă [CLS] token la început
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # (batch, 197, 768)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        # 4. Adaugă positional embeddings
        embeddings = embeddings + self.position_embedding
        
        # 5. Transformer encoding
        # Self-attention permite patch-urilor să "vadă" unele pe altele
        encoded = self.transformer(embeddings)
        
        # 6. Output: [CLS] token embedding pentru classification
        #            sau toate patch embeddings pentru dense tasks
        return encoded[:, 0]  # [CLS] token (batch, 768)
    
    def split_into_patches(self, image):
        """
        Split 224x224 image în 14x14 patches de 16x16
        """
        batch, c, h, w = image.shape
        p = self.patch_size
        
        # Reshape: (batch, 3, 224, 224) → (batch, 3, 14, 16, 14, 16)
        patches = image.reshape(batch, c, h//p, p, w//p, p)
        
        # Transpose: (batch, 14, 14, 3, 16, 16)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        
        # Flatten patches: (batch, 196, 3*16*16)
        patches = patches.reshape(batch, (h//p) * (w//p), c * p * p)
        
        return patches
```

**De ce ViT vs CNN?**
- **Scalability**: ViT scalează mai bine cu date și compute
- **Long-range dependencies**: Attention capturează relații între patch-uri distante
- **Transfer learning**: Pre-training pe imagini → fine-tune pe tasks specifice
- **Unified architecture**: Același Transformer pentru text și imagini

**2. Multimodal Embeddings - CLIP Architecture**

```python
class CLIPModel:
    """
    Contrastive Language-Image Pre-training
    
    Învață să alinieze embeddings text-imagine în același spațiu
    Folosit pentru: image-text similarity, zero-shot classification
    """
    def __init__(self):
        self.image_encoder = VisionTransformer()  # ViT
        self.text_encoder = TextTransformer()     # BERT-like
        
        # Projection heads pentru același dimensiuni
        self.image_projection = nn.Linear(768, 512)
        self.text_projection = nn.Linear(768, 512)
    
    def forward(self, images, texts):
        # Encode images și texts
        image_features = self.image_encoder(images)  # (batch, 768)
        text_features = self.text_encoder(texts)      # (batch, 768)
        
        # Project to shared embedding space
        image_embeds = self.image_projection(image_features)  # (batch, 512)
        text_embeds = self.text_projection(text_features)     # (batch, 512)
        
        # Normalize for cosine similarity
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        return image_embeds, text_embeds
    
    def contrastive_loss(self, image_embeds, text_embeds, temperature=0.07):
        """
        InfoNCE loss: maximizează similaritatea între perechi corecte
        """
        # Similarity matrix: (batch, batch)
        logits = torch.matmul(image_embeds, text_embeds.T) / temperature
        
        # Labels: diagonal (i-th image matches i-th text)
        labels = torch.arange(len(image_embeds))
        
        # Cross-entropy loss în ambele direcții
        loss_i2t = F.cross_entropy(logits, labels)      # Image → Text
        loss_t2i = F.cross_entropy(logits.T, labels)    # Text → Image
        
        return (loss_i2t + loss_t2i) / 2
```

**Utilizare în NOVA**:
```python
def analyze_image_with_context(image_path: str, query: str) -> str:
    """
    Vision analysis cu Claude 3.5 Sonnet
    
    Claude folosește arhitectură similară:
    - ViT pentru image encoding
    - Transformer decoder pentru text generation
    - Cross-attention între image patches și text tokens
    """
    # Load și encode imagine
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Prompt engineering pentru vision tasks
    prompt = f"""Analyze this image in detail. Focus on:

1. **Main objects and subjects**: What are the primary elements?
2. **Text content (OCR)**: Extract any visible text
3. **Context and meaning**: What does the image convey?
4. **Relevant details**: Colors, composition, style, quality

User question: {query}

Provide a comprehensive analysis."""

    # API call cu vision
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
    )
    
    analysis = response.content[0].text
    
    # Extract și store în vector DB
    image_embedding = create_embedding(analysis)  # Text embedding din descriere
    vector_db.add(
        embeddings=[image_embedding],
        documents=[analysis],
        metadatas=[{
            "type": "image",
            "source": image_path,
            "query": query
        }]
    )
    
    return analysis
```

**3. OCR și Text Detection în Imagini**

Pentru extracție text din imagini, folosim capability-ul OCR al modelelor vision:

```python
def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """
    OCR avansat cu position tracking
    """
    prompt = """Extract ALL text visible in this image.

For each text element, provide:
1. The exact text content
2. Its approximate position (top/middle/bottom, left/center/right)
3. Text properties (font size: large/medium/small, style: bold/normal)

Format as structured JSON:
{
    "text_elements": [
        {"text": "...", "position": "...", "size": "...", "style": "..."}
    ],
    "full_text": "Complete extracted text in reading order"
}"""

    response = claude_vision.analyze(image_path, prompt)
    
    # Parse JSON response
    extracted_data = json.loads(response)
    
    return extracted_data
```

**Pipeline de procesare**:
```
Input: Imagine (JPG, PNG, WEBP)
↓
1. IMAGE VALIDATION
   - Verifică format și dimensiune
   - Conversie la format standard
↓
2. VISUAL ANALYSIS (Claude 3.5 Sonnet Vision)
   - Descriere detaliată
   - Identificare obiecte
   - OCR pentru text în imagine
   - Analiză contextuală
↓
3. TEXT EXTRACTION
   - Extrage tot textul detectat
   - Structurează informațiile
↓
4. EMBEDDING & STORAGE
   - Text + descriere → embedding
   - Storage în ChromaDB
   - Metadata: {source, type:"image", analysis_timestamp}
↓
Output: Descriere + text extras + insight-uri
```

**Prompt Engineering pentru Vision**:
```python
vision_prompt = """
Analizează această imagine în detaliu și furnizează:

1. **Descriere generală**: Ce se vede în imagine
2. **Obiecte identificate**: Lista obiectelor principale
3. **Text vizibil**: Orice text scris în imagine (OCR)
4. **Context și interpretare**: Ce sugerează imaginea
5. **Detalii relevante**: Culori, compoziție, calitate

Fii precis și exhaustiv în analiză.
"""
```

#### 4.3.3 Audio Agent
**Specializare**: Transcrierea și analiza fișierelor audio

**Pipeline de procesare**:
```
Input: Fișier audio (MP3, WAV, M4A)
↓
1. AUDIO VALIDATION
   - Verifică format
   - Conversie la format compatibil (dacă necesar)
↓
2. TRANSCRIPTION (Faster-Whisper Local)
   - Model: whisper-large-v3 (local)
   - Output: Text + timestamps
   - Limba: Auto-detectată
↓
3. POST-PROCESSING
   - Curățare text
   - Segmentare pe topice
   - Identificare vorbitori (dacă posibil)
↓
4. EMBEDDING & STORAGE
   - Transcrierea → embedding
   - Metadata: {source, duration, language, speaker}
↓
Output: Transcriere + metadata
```

#### 4.3.4 Web Agent
**Specializare**: Extracția și procesarea conținutului web

**Pipeline de procesare**:
```
Input: URL sau query de căutare
↓
1. WEB SCRAPING
   - Beautiful Soup pentru HTML parsing
   - Requests pentru fetch
   - Extrage titlu, text principal, link-uri
↓
2. CONTENT CLEANING
   - Îndepărtare JavaScript, CSS
   - Extragere doar conținut relevant
   - Preservare structură (headers, lists)
↓
3. CHUNKING & EMBEDDING
   - Similar cu Document Agent
   - Metadata: {url, title, scrape_date}
↓
4. STORAGE
   - ChromaDB collection: "web_content"
↓
Output: Conținut web + metadata
```

**Extensie viitoare**: Integrare cu API-uri de search (Brave, Google) pentru rezultate mai bune.

---

### 4.4 Layer 4: Knowledge Base (Vector Database)

**Alegerea tehnologiei: ChromaDB**

**Motivație**:
- **Lightweight**: Nu necesită server separat
- **Persistent**: Salvează date local
- **Python-native**: Integrare perfectă
- **Open-source**: Gratuită și extensibilă
- **Performance**: Rapid pentru datasets mici-medii

**Structura bazei de date**:
```python
# Collections separate pentru fiecare tip de date
collections = {
    "documents": {
        "embeddings": List[float],  # 768D vectors
        "documents": List[str],      # Text content
        "metadatas": List[Dict],     # Metadata
        "ids": List[str]             # Unique IDs
    },
    "images": {...},
    "audio": {...},
    "web": {...}
}
```

**Operațiuni principale**:

1. **Add** (Inserare):
```python
collection.add(
    embeddings=embeddings,
    documents=texts,
    metadatas=metadata_list,
    ids=unique_ids
)
```

2. **Query** (Căutare semantică):
```python
results = collection.query(
    query_embeddings=query_vector,
    n_results=5,
    where={"source": "specific_doc.pdf"},  # Filtrare opțională
    include=["documents", "metadatas", "distances"]
)
```

3. **Update** (Actualizare):
```python
collection.update(
    ids=["id1"],
    metadatas=[{"updated": True}]
)
```

4. **Delete** (Ștergere):
```python
collection.delete(
    ids=["id1", "id2"]
)
```

**Strategii de căutare**:
- **Pure semantic search**: Bazată pe cosine similarity
- **Hybrid search**: Semantic + keyword filtering
- **Re-ranking**: Post-procesare cu LLM pentru a ordona rezultatele

---

### 4.5 Layer 5: Tools & Utilities

#### 4.5.1 LLM Interface
**Abstractizare pentru multiple modele**:
```python
class LLMInterface:
    """
    Interfață unificată pentru diferite LLM-uri
    Suportă: Anthropic Claude, Mistral, Llama (local), Qwen
    """
    def generate(self, prompt: str, model: str, **kwargs) -> str
    def embed(self, text: str, model: str) -> List[float]
    def vision_analyze(self, image: bytes, prompt: str) -> str
```

**Model Selection Strategy**:
- **Claude 3.5 Sonnet**: Pentru reasoning complex, synthesis și analiză avansată (primary)
- **Mistral Large**: Pentru task-uri complexe, alternative robustă
- **Mistral Small**: Pentru task-uri simple, cost-efficient
- **Claude 3.5 Sonnet (Vision)**: Pentru analiza imaginilor
- **Llama 3.1 405B** (local sau Groq): Pentru task-uri unde vrem independență totală
- **Embeddings**: mistral-embed (cost-performance balance) sau nomic-embed (open-source)

#### 4.5.2 File Processing
**Handlers specializați**:
```python
class FileProcessor:
    handlers = {
        ".pdf": PDFHandler,
        ".docx": DOCXHandler,
        ".txt": TextHandler,
        ".jpg": ImageHandler,
        ".png": ImageHandler,
        ".mp3": AudioHandler,
        ".wav": AudioHandler,
    }
    
    def process(self, file_path: str) -> ProcessedContent
```

#### 4.5.3 Logging & Monitoring
**Tracking pentru debugging și optimizare**:
```python
import logging

logger = logging.getLogger("NOVA")
logger.setLevel(logging.INFO)

# Log format: [TIMESTAMP] [LEVEL] [COMPONENT] Message
# Componente: UI, Orchestrator, DocumentAgent, VisionAgent, etc.
```

**Metrici urmărite**:
- Timp de procesare per agent
- Cost per request (API calls)
- Calitatea rezultatelor (user feedback)
- Număr de documente în vector DB

---

## 5. FLUXUL COMPLET AL UNEI INTEROGĂRI

### 5.1 Exemplu Concret: "Analizează acest PDF și imaginea și spune-mi ce diferențe există"

```
STEP 1: User Input (UI Layer)
├─ Query: "Analizează acest PDF și imaginea..."
├─ Files: [report.pdf, chart.png]
└─ Trimite către Orchestrator

STEP 2: Orchestrator Analysis
├─ Detectează: 2 tipuri de input (document + image)
├─ Planning:
│  ├─ Task 1: Procesează PDF cu DocumentAgent
│  ├─ Task 2: Procesează image cu VisionAgent
│  └─ Task 3: Compară rezultatele și sintetizează
└─ Execută în paralel Task 1 și 2

STEP 3: DocumentAgent Processing
├─ Extrage text din PDF (PyMuPDF)
├─ Split în chunks (15 chunks de ~1000 tokens)
├─ Generează embeddings pentru fiecare chunk
├─ Salvează în ChromaDB collection "documents"
├─ Query embeddings cu întrebarea utilizatorului
├─ Returnează top 5 chunks relevante
└─ Output: "Documentul descrie vânzări Q3 2024..."

STEP 4: VisionAgent Processing
├─ Încarcă imaginea chart.png
├─ Trimite către GPT-4 Vision cu prompt de analiză
├─ GPT-4V răspunde: "Grafic cu vânzări lunare..."
├─ Extrage text din imagine (OCR): "Q3 Total: $2.5M"
├─ Generează embedding pentru descriere + text
├─ Salvează în ChromaDB collection "images"
└─ Output: "Graficul arată evoluția vânzărilor..."

STEP 5: Orchestrator Synthesis
├─ Primește rezultate de la ambii agenți
├─ Query în vector DB pentru context suplimentar
├─ Construiește prompt de sinteză:
│  ├─ Context: Document result + Vision result
│  ├─ Task: "Compară informațiile și identifică diferențe"
│  └─ Sources: Include metadata pentru citări
├─ Trimite către Claude 3.5 Sonnet
├─ Primește răspuns sintetizat
└─ Output: "Pe baza analizei documentului și graficului..."

STEP 6: Response Display (UI Layer)
├─ Afișează răspunsul în chat
├─ Include citări: [Source: report.pdf, p.3] [Source: chart.png]
├─ Afișează timp de procesare
└─ Salvează în session state pentru context viitor
```

**Timp estimat**: 8-15 secunde
**Cost estimat**: $0.05-0.15 (depending on document size)

---

## 6. DESIGN PATTERNS ȘI BEST PRACTICES

### 6.1 Agent Pattern (Multi-Agent System)
**Pattern**: Sistem de agenți specializați coordonați de un orchestrator.

**Beneficii**:
- **Modularitate**: Fiecare agent are o responsabilitate clară
- **Scalabilitate**: Adăugăm ușor agenți noi
- **Paralelizare**: Agenții pot lucra simultan
- **Testabilitate**: Fiecare agent se testează independent

### 6.2 RAG Pattern (Retrieval-Augmented Generation)
**Pattern**: Combină retrieval din knowledge base cu generare de text.

**Beneficii**:
- **Acuratețe**: Răspunsuri bazate pe date reale
- **Actualizabilitate**: Adăugăm informații fără re-training
- **Citare surselor**: Transparență și verificabilitate
- **Cost-efficient**: Nu necesită fine-tuning

### 6.3 Chain-of-Thought Pattern
**Pattern**: Descompune probleme complexe în pași logici.

**Implementare**:
```python
prompt = """
Să rezolvăm această problemă pas cu pas:

1. Mai întâi, să identificăm informațiile din document
2. Apoi, să analizăm imaginea
3. În final, să comparăm cele două surse

Pas 1: Din document observăm...
"""
```

### 6.4 Prompt Engineering Best Practices
**Strategii folosite**:
- **Clear instructions**: Instrucțiuni specifice și clare
- **Few-shot examples**: Exemple de output dorit (când e cazul)
- **Structured output**: Cerem răspunsuri structurate (JSON, Markdown)
- **Role prompting**: "Tu ești un expert analyst..."
- **Chain-of-thought**: "Gândește pas cu pas..."

### 6.5 Error Handling & Resilience
**Strategii implementate**:
```python
@retry(max_attempts=3, backoff=2)
def call_llm_with_retry(prompt):
    """Retry cu exponential backoff"""
    try:
        return llm.generate(prompt)
    except RateLimitError:
        time.sleep(10)
        raise
    except APIError as e:
        logger.error(f"LLM API Error: {e}")
        raise
```

**Fallback mechanisms**:
- Dacă un agent eșuează, orchestratorul continuă cu alții
- Dacă vector DB nu returnează rezultate, folosim doar LLM knowledge
- Dacă un model e indisponibil, folosim un model alternativ

---

## 7. SCALABILITATE ȘI OPTIMIZĂRI

### 7.1 Performance Optimizations

**Caching**:
```python
@lru_cache(maxsize=128)
def get_embedding(text: str) -> List[float]:
    """Cache embeddings pentru texte folosite frecvent"""
    return embedding_model.embed(text)
```

**Batch Processing**:
```python
# În loc de:
for chunk in chunks:
    embedding = get_embedding(chunk)
    
# Folosim:
embeddings = get_embeddings_batch(chunks)  # Un singur API call
```

**Parallel Agent Execution**:
```python
import asyncio

async def process_multi_agent(query, files):
    tasks = [
        document_agent.process_async(files),
        vision_agent.process_async(files),
        web_agent.process_async(query)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### 7.2 Scalare la Volume Mari

**Pentru datasets mari (>10GB)**:
1. **Chunking agresiv**: Reducere chunk size pentru mai multă precizie
2. **Hierarchical indexing**: Index la nivel de document + chunk
3. **Pinecone/Weaviate**: Migrare la vector DB cloud-based
4. **Distributed processing**: Multiple workers pentru procesare

**Pentru multe utilizatori simultani**:
1. **Async processing**: FastAPI în loc de Streamlit
2. **Queue system**: Celery pentru task queue
3. **Load balancing**: Multiple instanțe ale aplicației
4. **Caching layer**: Redis pentru răspunsuri frecvente

### 7.3 Cost Optimization

**Strategii de reducere costuri**:
1. **Model selection**: Mistral Small pentru task-uri simple, Claude Haiku pentru rapiditate
2. **Embedding caching**: Refolosim embeddings existente
3. **Batch API calls**: Grupăm cereri pentru discount
4. **Token management**: Limităm context size
5. **Smart retrieval**: Doar top-k rezultate relevante
6. **Local models**: Llama/Whisper local pentru procesare offline

**Cost estimat per utilizator**:
- Document processing: $0.01-0.03 per document (Mistral)
- Image analysis: $0.01-0.08 per image (Claude Vision)
- Audio transcription: $0.00 (local Faster-Whisper)
- Chat interaction: $0.001-0.015 per mesaj (Claude/Mistral)

---

## 8. SECURITATE ȘI PRIVACY

### 8.1 Măsuri de Securitate

**API Keys Management**:
```python
from dotenv import load_dotenv
import os

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Nu hard-code niciodată
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
```

**Data Sanitization**:
- Validare input-uri (tip, dimensiune, format)
- Curățare conținut potențial periculos
- Rate limiting pentru API calls

**Access Control**:
- Utilizatori pot accesa doar propriile date
- Session-based isolation
- Ștergere date după închiderea sesiunii (opțional)

### 8.2 Privacy

**Principii**:
- **Data Minimization**: Stocăm doar ce e necesar
- **User Consent**: Informăm utilizatorii ce date procesăm
- **Local First**: ChromaDB local, nu cloud (default)
- **Right to Delete**: Utilizatorii pot șterge datele

**GDPR Compliance** (pentru deployment public):
- Explicăm ce date colectăm
- Oferim export de date
- Implementăm "right to be forgotten"

---

## 9. TESTING & QUALITY ASSURANCE

### 9.1 Strategii de Testare

**Unit Tests**:
```python
def test_document_chunking():
    text = "Sample document with multiple paragraphs..."
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(len(chunk) <= 120 for chunk in chunks)  # +overlap
```

**Integration Tests**:
```python
def test_document_agent_flow():
    agent = DocumentAgent()
    result = agent.process("test_document.pdf")
    assert result.status == "success"
    assert len(result.chunks) > 0
    assert result.embeddings is not None
```

**End-to-End Tests**:
```python
def test_full_conversation_flow():
    # Simulează o conversație completă
    response = orchestrator.process_query(
        query="Ce informații sunt în acest PDF?",
        files=["test.pdf"]
    )
    assert "test.pdf" in response
    assert len(response) > 50
```

### 9.2 Evaluation Metrics

**Pentru RAG System**:
- **Retrieval Precision**: Câte chunk-uri returnate sunt relevante?
- **Retrieval Recall**: Câte chunk-uri relevante sunt returnate?
- **Answer Relevance**: Răspunsul e relevant la întrebare?
- **Answer Faithfulness**: Răspunsul e fidel surselor?
- **Latency**: Timp de răspuns

**Measurement Framework**:
```python
def evaluate_rag_system(test_cases):
    results = []
    for case in test_cases:
        response = system.query(case.query)
        metrics = {
            "precision": calculate_precision(response, case.relevant_chunks),
            "recall": calculate_recall(response, case.relevant_chunks),
            "faithfulness": llm_judge_faithfulness(response, case.sources),
            "relevance": llm_judge_relevance(response, case.query),
            "latency": response.processing_time
        }
        results.append(metrics)
    return aggregate_metrics(results)
```

---

## 10. ROADMAP & EXTENSII VIITOARE

### 10.1 Faza Actuală (MVP)
✅ Arhitectură multi-agent de bază
✅ Document, Vision, Audio, Web agents
✅ ChromaDB pentru knowledge base
✅ Streamlit UI
✅ Orchestrator cu routing inteligent

### 10.2 Faza 2 (Îmbunătățiri Curente)
🔄 Optimizare prompts și chain-of-thought
🔄 Caching și performance improvements
🔄 Error handling robust
🔄 Logging și monitoring

### 10.3 Faza 3 (Extensii Planificate)
🔮 **Multi-User Support**: Autentificare și izolare date
🔮 **Cloud Vector DB**: Migrare la Pinecone/Weaviate pentru scalabilitate
🔮 **Fine-tuned Embeddings**: Embeddings custom pentru domenii specifice
🔮 **Graph RAG**: Integrare knowledge graphs pentru relații complexe
🔮 **Agent Memory**: Long-term memory pentru fiecare utilizator
🔮 **Multi-Language**: Suport pentru limbi multiple
🔮 **Voice Interface**: Input/output vocal
🔮 **Real-time Collaboration**: Multiple utilizatori pe același document

### 10.4 Faza 4 (Cercetare Avansată)
🔬 **Self-Improving Agents**: Agenți care învață din feedback
🔬 **Causal Reasoning**: Înțelegerea relațiilor cauză-efect
🔬 **Multimodal Fusion**: Fuziune avansată audio-video-text
🔬 **Federated Learning**: Învățare distribuită păstrând privacy-ul
🔬 **Neural-Symbolic AI**: Combinare rețele neurale + logică simbolică

---

## 11. CONCLUZIE

### 11.1 Rezumat Arhitectural

Proiectul NOVA implementează o arhitectură AI modernă bazată pe principii clasice:

1. **Reprezentare cunoștințe**: Vector embeddings + metadata structurată
2. **Raționament**: Chain-of-thought + multi-agent planning
3. **Învățare**: Incremental learning + transfer learning
4. **Multi-modalitate**: Procesare unificată text, imagine, audio, web
5. **Explicabilitate**: Citare surse + transparență procesare

### 11.2 Puncte Forte

✅ **Modularitate**: Agenți independenți, ușor de extins
✅ **Scalabilitate**: Poate crește de la prototype la production
✅ **Flexibilitate**: Suport multiple tipuri de date și surse
✅ **Transparență**: Sistem explicabil și auditabil
✅ **Cost-eficient**: Folosește API-uri, nu necesită GPU-uri proprii

### 11.3 Limitări Curente

⚠️ **Single-user**: Nu suportă multiple utilizatori simultan (încă)
⚠️ **Local storage**: ChromaDB local, nu distribuit
⚠️ **API dependency**: Dependență de API-uri externe (OpenAI)
⚠️ **No fine-tuning**: Folosim modele generice, nu specializate
⚠️ **Limited reasoning**: Raționament limitat la capabilitățile LLM-urilor

### 11.4 De Ce Această Arhitectură?

Această arhitectură reprezintă **best practice în 2025** pentru sisteme AI aplicative:

- **RAG over Fine-tuning**: Mai flexibil, mai ieftin, mai ușor de actualizat
- **Multi-agent over Monolithic**: Mai modular, mai scalabil
- **Vector DB over Traditional DB**: Căutare semantică, nu doar keyword
- **LLM Orchestration**: Folosim inteligența LLM-urilor pentru coordonare
- **Hybrid approach**: Combină symbolic AI (routing) cu neural AI (LLMs)

---

## 12. RESURSE ȘI REFERINȚE

### 12.1 Tehnologii Folosite
- **LLMs**: Anthropic Claude 3.5 Sonnet, Mistral Large/Small, Llama 3.1 (opțional local)
- **Embeddings**: mistral-embed sau nomic-embed (open-source)
- **Vision**: Claude 3.5 Sonnet (vision) sau LLaVA (open-source)
- **Audio**: Faster-Whisper (local, open-source)
- **Vector DB**: ChromaDB
- **UI**: Streamlit
- **Python**: 3.11+
- **Libraries**: LangChain, PyMuPDF, Beautiful Soup, anthropic-sdk, mistralai-sdk

### 12.2 Referințe Academice
- "Attention Is All You Need" (Vaswani et al., 2017) - Transformers
- "Retrieval-Augmented Generation" (Lewis et al., 2020) - RAG
- "Chain-of-Thought Prompting" (Wei et al., 2022) - Reasoning
- "REALM" (Guu et al., 2020) - Knowledge retrieval
- "Multi-Agent Systems" (Wooldridge, 2009) - Agent coordination

### 12.3 Best Practices Sources
- Anthropic Claude Documentation & Prompt Engineering Guide
- Mistral AI Documentation
- LangChain Documentation
- ChromaDB Documentation
- Hugging Face Model Hub (pentru modele open-source)

### 12.4 Alternative Open-Source Complete
**Pentru independență totală (fără API-uri externe)**:
- **LLM**: Llama 3.1 405B/70B (via Ollama local), Qwen 2.5, Mixtral
- **Embeddings**: nomic-embed-text, sentence-transformers
- **Vision**: LLaVA 1.6, Qwen-VL
- **Audio**: Faster-Whisper (deja open-source)
- **Deployment**: Poate rula complet offline pe hardware propriu

---

**Document Version**: 2.0 (OpenAI-Free)
**Date**: 28 Noiembrie 2025
**Autor**: Arhitectură NOVA AI System
**Status**: Living Document (se actualizează pe măsură ce proiectul evoluează)
