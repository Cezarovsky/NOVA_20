"""
ATTENTION MECHANISM DEMO
========================
Demonstrație pas-cu-pas a mecanismului Attention(Q, K, V) = softmax(QK^T / √d_k) V

Propoziție demo: "cainele da din coada"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 70)
print("ATTENTION MECHANISM - Demonstrație Completă")
print("=" * 70)

# ============================================================================
# SETARE: Propoziție și Vocabular
# ============================================================================

text = "cainele da din coada"
tokens = text.split()
print(f"\n📝 Text original: '{text}'")
print(f"🔤 Token-uri: {tokens}")

# Vocabular simplu
vocab = {
    "<PAD>": 0,
    "cainele": 1,
    "da": 2,
    "din": 3,
    "coada": 4,
}

token_ids = [vocab[token] for token in tokens]
print(f"🔢 Token IDs: {token_ids}")

# ============================================================================
# STEP 1: EMBEDDINGS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: EMBEDDINGS (dicționar static)")
print("=" * 70)

vocab_size = len(vocab)
d_model = 8  # 8 dimensiuni pentru demo (NOVA folosește 512)
seq_len = len(tokens)

embedding_layer = nn.Embedding(vocab_size, d_model)

# Setăm manual embeddings pentru claritate pedagogică
with torch.no_grad():
    embedding_layer.weight[1] = torch.tensor([0.8, 0.9, 0.2, -0.3, 0.7, 0.5, 0.4, 0.6])  # cainele
    embedding_layer.weight[2] = torch.tensor([0.3, 0.2, 0.9, 0.8, -0.2, 0.4, 0.1, 0.3])  # da
    embedding_layer.weight[3] = torch.tensor([0.1, 0.1, 0.3, 0.2, 0.1, -0.1, 0.2, 0.1])  # din
    embedding_layer.weight[4] = torch.tensor([0.7, 0.6, 0.3, -0.2, 0.8, 0.4, 0.5, 0.7])  # coada

# Lookup embeddings
input_ids = torch.tensor(token_ids)
embeddings = embedding_layer(input_ids)

print(f"\n📊 Shape embeddings: {embeddings.shape}  # [seq_len={seq_len}, d_model={d_model}]")
print(f"\nEmbeddings pentru fiecare token:")
for i, token in enumerate(tokens):
    print(f"  {token:8s}: {embeddings[i].detach().numpy()}")

# ============================================================================
# STEP 2: PROIECȚII Q, K, V
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: TRANSFORMĂRI Q, K, V")
print("=" * 70)

# Dimensiuni Attention
d_k = d_model  # dimensiunea pentru Query și Key
d_v = d_model  # dimensiunea pentru Value

# Matrici de transformare (în practică, învățate din antrenament)
W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_v, bias=False)

# Inițializare simplă pentru demo
with torch.no_grad():
    nn.init.eye_(W_q.weight)  # identitate pentru simplitate
    nn.init.eye_(W_k.weight)
    nn.init.eye_(W_v.weight)

# Calculăm Q, K, V
Q = W_q(embeddings)  # [seq_len, d_k]
K = W_k(embeddings)  # [seq_len, d_k]
V = W_v(embeddings)  # [seq_len, d_v]

print(f"\n📊 Shape Q: {Q.shape}  # Query  [seq_len={seq_len}, d_k={d_k}]")
print(f"📊 Shape K: {K.shape}  # Key    [seq_len={seq_len}, d_k={d_k}]")
print(f"📊 Shape V: {V.shape}  # Value  [seq_len={seq_len}, d_v={d_v}]")

print(f"\n🔍 Query pentru 'coada' (token 3):")
print(f"   Q[3] = {Q[3].detach().numpy()}")
print(f"\n🔑 Key pentru 'da' (token 1):")
print(f"   K[1] = {K[1].detach().numpy()}")
print(f"\n💎 Value pentru 'cainele' (token 0):")
print(f"   V[0] = {V[0].detach().numpy()}")

# ============================================================================
# STEP 3: CALCUL SCORURI (QK^T)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: COMPATIBILITATE - QK^T")
print("=" * 70)

# Q @ K^T: [seq_len, d_k] @ [d_k, seq_len] = [seq_len, seq_len]
scores = torch.matmul(Q, K.transpose(-2, -1))

print(f"\n📊 Shape scores: {scores.shape}  # [seq_len={seq_len}, seq_len={seq_len}]")
print(f"\n🎯 Matrice de compatibilitate (QK^T):")
print(f"     {' '.join([f'{t:>8s}' for t in tokens])}")
for i, token in enumerate(tokens):
    row = '  '.join([f'{scores[i][j].item():8.3f}' for j in range(seq_len)])
    print(f"{token:8s}: {row}")

print(f"\n💡 Interpretare:")
print(f"   - Valori mari = token-uri compatibile semantic")
print(f"   - scores[3][0] = compatibilitate între 'coada' și 'cainele': {scores[3][0].item():.3f}")
print(f"   - scores[3][2] = compatibilitate între 'coada' și 'din': {scores[3][2].item():.3f}")

# ============================================================================
# STEP 4: SCALARE (/ √d_k)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: SCALARE - împărțire la √d_k")
print("=" * 70)

sqrt_d_k = math.sqrt(d_k)
scaled_scores = scores / sqrt_d_k

print(f"\n📐 √d_k = √{d_k} = {sqrt_d_k:.3f}")
print(f"\n🎯 Scoruri scalate (QK^T / √d_k):")
print(f"     {' '.join([f'{t:>8s}' for t in tokens])}")
for i, token in enumerate(tokens):
    row = '  '.join([f'{scaled_scores[i][j].item():8.3f}' for j in range(seq_len)])
    print(f"{token:8s}: {row}")

print(f"\n💡 De ce scalăm?")
print(f"   - Pentru d_k mare (ex: 512), dot product-ul devine FOARTE mare")
print(f"   - Softmax devine instabil (gradienți foarte mici)")
print(f"   - Scalarea menține valorile într-un interval rezonabil")

# ============================================================================
# STEP 5: SOFTMAX (transformare în probabilități)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: SOFTMAX - transformare în probabilități")
print("=" * 70)

attention_weights = F.softmax(scaled_scores, dim=-1)

print(f"\n📊 Shape attention_weights: {attention_weights.shape}")
print(f"\n🎯 Atenție (probabilități) - fiecare rând sumează la 1.0:")
print(f"     {' '.join([f'{t:>8s}' for t in tokens])}")
for i, token in enumerate(tokens):
    row = '  '.join([f'{attention_weights[i][j].item():8.3f}' for j in range(seq_len)])
    suma = attention_weights[i].sum().item()
    print(f"{token:8s}: {row}  | Σ={suma:.3f}")

print(f"\n💡 Interpretare pentru 'coada' (rândul 3):")
print(f"   - Acordă {attention_weights[3][0].item()*100:.1f}% atenție la 'cainele'")
print(f"   - Acordă {attention_weights[3][1].item()*100:.1f}% atenție la 'da'")
print(f"   - Acordă {attention_weights[3][2].item()*100:.1f}% atenție la 'din'")
print(f"   - Acordă {attention_weights[3][3].item()*100:.1f}% atenție la sine ('coada')")

# ============================================================================
# STEP 6: APLICARE PE VALUES (× V)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: COMBINARE - attention_weights × V")
print("=" * 70)

# attention_weights @ V: [seq_len, seq_len] @ [seq_len, d_v] = [seq_len, d_v]
output = torch.matmul(attention_weights, V)

print(f"\n📊 Shape output: {output.shape}  # [seq_len={seq_len}, d_v={d_v}]")
print(f"\n🎯 Output după Attention:")
for i, token in enumerate(tokens):
    print(f"\n  {token:8s} (înainte): {embeddings[i].detach().numpy()}")
    print(f"  {token:8s} (după):    {output[i].detach().numpy()}")

print(f"\n💡 Ce s-a întâmplat?")
print(f"   Embedding-ul pentru 'coada' ERA: {embeddings[3].detach().numpy()}")
print(f"   DUPĂ Attention devine:           {output[3].detach().numpy()}")
print(f"\n   Diferența:")
diff = output[3] - embeddings[3]
print(f"   {diff.detach().numpy()}")
print(f"\n   'coada' a ABSORBIT informații din:")
print(f"   - 'cainele' ({attention_weights[3][0].item()*100:.1f}%)")
print(f"   - 'da' ({attention_weights[3][1].item()*100:.1f}%)")
print(f"   - 'din' ({attention_weights[3][2].item()*100:.1f}%)")
print(f"   Acum 'coada' înseamnă 'coada UNUI CÂINE care DAĂ din ea'!")

# ============================================================================
# STEP 7: VIZUALIZARE ATENȚIE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 7: VIZUALIZARE - Heatmap Atenție")
print("=" * 70)

print(f"\n🔥 Intensitatea atenției (0.0 = ignoră, 1.0 = focus maxim):\n")
print(f"        → Către:")
print(f"        {' '.join([f'{t:>8s}' for t in tokens])}")
print("    ↓ De la:")

for i, token_from in enumerate(tokens):
    row = []
    for j in range(seq_len):
        val = attention_weights[i][j].item()
        if val > 0.4:
            symbol = "██"  # atenție mare
        elif val > 0.2:
            symbol = "▓▓"  # atenție medie
        elif val > 0.1:
            symbol = "▒▒"  # atenție mică
        else:
            symbol = "░░"  # atenție minimă
        row.append(f"{symbol}")
    print(f"{token_from:8s}: {' '.join(row)}")

# ============================================================================
# REZUMAT FINAL
# ============================================================================

print("\n" + "=" * 70)
print("REZUMAT: FLUXUL COMPLET")
print("=" * 70)

print(f"""
1. EMBEDDINGS:        Text → Vectori statici [4, 8]
2. PROIECȚII Q,K,V:   Transformări liniare → Q, K, V
3. COMPATIBILITATE:   QK^T → matrice [4, 4] de scoruri
4. SCALARE:           / √{d_k} → stabilizare numerică
5. SOFTMAX:           → probabilități (suma = 1.0)
6. COMBINARE:         × V → output cu context!

Rezultat: Fiecare token și-a "citit" vecinii și a absorbit informații relevante!

Token 'coada' acum ȘTIE că:
  - Face parte dintr-un context cu 'cainele'
  - E asociată cu acțiunea 'da'
  - Embedding-ul său NU mai e ambiguu - e "coada DE CÂINE care se mișcă"!
""")

print("=" * 70)
print("✅ Demo complet! Rulează din nou pentru a revedea pașii.")
print("=" * 70)
