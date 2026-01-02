"""
PAS 2: SELF-ATTENTION - Învățăm Context

Ce învățăm aici:
- Cum modelul înțelege contextul
- Ce înseamnă "attention" (atenție)
- Cum cuvintele "vorbesc" între ele
- Query, Key, Value (Q, K, V)

Analogie: O conversație la masă - fiecare persoană decide cui să-i acorde atenție
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# CONCEPTUL: Ce e Self-Attention?
# =============================================================================

"""
PROBLEMA:
--------
Propoziție: "Te iubesc dragă"

Embeddings (din Step 1):
- "Te"     → [0.1, 0.5, 0.3, ...]
- "iubesc" → [0.8, 0.2, 0.6, ...]
- "dragă"  → [0.3, 0.9, 0.1, ...]

PROBLEMĂ: Fiecare cuvânt e independent! 
"iubesc" nu știe despre "Te" sau "dragă"!

SOLUȚIA: SELF-ATTENTION
-----------------------
"iubesc" SE UITĂ la "Te" → înțelege CINE iubește
"iubesc" SE UITĂ la "dragă" → înțelege PE CINE iubește

REZULTAT: Fiecare cuvânt capătă CONTEXT!
"""


# =============================================================================
# ANALOGIE: Conversație la masă
# =============================================================================

"""
Imaginează 3 persoane la masă:

Persoana 1 (Te): "Am pregătit cina"
Persoana 2 (iubesc): "Mulțumesc mult!"
Persoana 3 (dragă): "Ești minunat!"

ÎNTREBARE: Cui răspunde Persoana 2?

SELF-ATTENTION:
- Persoana 2 SE UITĂ la toți ceilalți
- Calculează "cât de relevant" e fiecare
- Acordă mai multă ATENȚIE la Persoana 1 (care a vorbit despre cină)
- Mai puțină atenție la Persoana 3

Attention weights:
- Atenție la "Te": 0.7 (70%)  ← relevant!
- Atenție la "dragă": 0.3 (30%)  ← mai puțin relevant

Răspuns contextual: "Mulțumesc mult [TIE pentru cină]!"
"""


# =============================================================================
# MATEMATICA: Query, Key, Value (Q, K, V)
# =============================================================================

"""
3 CONCEPTE CHEIE:

1. QUERY (Q) = "Ce întreb?" / "Ce caut?"
   - "iubesc" întreabă: "Cine face acțiunea? Cine primește?"

2. KEY (K) = "Ce ofer ca informație?"
   - "Te" oferă: "Eu sunt subiectul"
   - "dragă" oferă: "Eu sunt obiectul"

3. VALUE (V) = "Ce informație concretă am?"
   - Conținutul semantic real

ATENȚIE = Cât de bine se potrivesc Q și K
"""


# =============================================================================
# PASUL 2A: Scaled Dot-Product Attention (formula simplă)
# =============================================================================

def simple_attention(query, key, value):
    """
    Attention cel mai simplu.
    
    Formula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        query: [seq_len, d_k] - "Ce caut?"
        key:   [seq_len, d_k] - "Ce ofer?"
        value: [seq_len, d_v] - "Informația mea"
    
    Returns:
        output: [seq_len, d_v] - Vectori cu context
        attention_weights: [seq_len, seq_len] - Cât de mult atenție
    
    Exemplu vizual:
        Query: "iubesc" întreabă despre context
        Keys: ["Te", "iubesc", "dragă"] răspund
        → Attention: [0.4, 0.1, 0.5] (40% "Te", 10% "iubesc", 50% "dragă")
    """
    # Dimensiunea pentru scaling
    d_k = query.size(-1)
    
    # Pas 1: Calculează similaritatea Q cu fiecare K
    # Q @ K^T = "Cât de bine se potrivesc?"
    scores = torch.matmul(query, key.transpose(-2, -1))  # [seq_len, seq_len]
    
    # Pas 2: Scale (împarte la sqrt(d_k)) pentru stabilitate numerică
    scores = scores / math.sqrt(d_k)
    
    # Pas 3: Softmax - transformă în probabilități (suma = 1)
    attention_weights = F.softmax(scores, dim=-1)  # [seq_len, seq_len]
    
    # Pas 4: Aplică attention weights pe values
    # "Adună informația, ponderat cu atenția"
    output = torch.matmul(attention_weights, value)  # [seq_len, d_v]
    
    return output, attention_weights


# =============================================================================
# PASUL 2B: Self-Attention Layer (cu parametri învățați)
# =============================================================================

class SimpleSelfAttention(nn.Module):
    """
    Self-Attention layer cu parametri care se învață.
    
    Parametri:
        embed_dim: Dimensiunea embedding-urilor (din Step 1)
        
    Ce învață:
        W_q: Matrix pentru Query  [embed_dim, embed_dim]
        W_k: Matrix pentru Key    [embed_dim, embed_dim]
        W_v: Matrix pentru Value  [embed_dim, embed_dim]
    
    Intuiție:
        - W_q învață: "Ce întrebări să pun pentru context?"
        - W_k învață: "Ce informație să ofer când sunt întrebat?"
        - W_v învață: "Ce să transmit când sunt relevant?"
    """
    
    def __init__(self, embed_dim):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Învățăm 3 transformări liniare: Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        print(f"Self-Attention Layer creat:")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Parametri Q: {embed_dim * embed_dim:,}")
        print(f"  Parametri K: {embed_dim * embed_dim:,}")
        print(f"  Parametri V: {embed_dim * embed_dim:,}")
        print(f"  TOTAL: {3 * embed_dim * embed_dim:,} parametri")
    
    def forward(self, x):
        """
        Aplică self-attention.
        
        Args:
            x: [seq_len, embed_dim] - Embeddings din Step 1
        
        Returns:
            output: [seq_len, embed_dim] - Cu context aplicat
            attention_weights: [seq_len, seq_len] - Vizualizare
        """
        # Pas 1: Generează Q, K, V prin transformări liniare
        Q = self.W_q(x)  # [seq_len, embed_dim]
        K = self.W_k(x)  # [seq_len, embed_dim]
        V = self.W_v(x)  # [seq_len, embed_dim]
        
        # Pas 2: Aplică scaled dot-product attention
        output, attention_weights = simple_attention(Q, K, V)
        
        return output, attention_weights


# =============================================================================
# PASUL 2C: Vizualizare Attention Weights
# =============================================================================

def visualize_attention(attention_weights, tokens):
    """
    Afișează attention weights într-un format ușor de citit.
    
    Args:
        attention_weights: [seq_len, seq_len]
        tokens: lista de tokens (strings)
    
    Exemplu output:
        "Te" acordă atenție:
          → "Te": 0.45
          → "iubesc": 0.35
          → "dragă": 0.20
    """
    seq_len = len(tokens)
    
    print("\n" + "=" * 60)
    print("ATTENTION WEIGHTS VISUALIZATION")
    print("=" * 60)
    
    for i in range(seq_len):
        print(f"\n'{tokens[i]}' acordă atenție:")
        for j in range(seq_len):
            weight = attention_weights[i, j].item()
            bar = "█" * int(weight * 20)  # Bară vizuală
            print(f"  → '{tokens[j]}': {weight:.3f} {bar}")


# =============================================================================
# TEST: Verificăm Self-Attention
# =============================================================================

def test_self_attention():
    """
    Test complet pentru self-attention.
    """
    print("=" * 60)
    print("TEST: Self-Attention Mechanism")
    print("=" * 60)
    
    # Pas 1: Creăm embeddings (mock data)
    print("\n--- Pas 1: Embeddings (din Step 1) ---")
    
    tokens = ["Te", "iubesc", "dragă"]
    embed_dim = 8  # Mic pentru test
    
    # Embeddings aleatorii (în realitate vin din Step 1)
    torch.manual_seed(42)  # Pentru reproducibilitate
    embeddings = torch.randn(3, embed_dim)
    
    print(f"Tokens: {tokens}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"\nPrimul embedding ('Te'):\n{embeddings[0]}")
    
    # Pas 2: Creăm Self-Attention layer
    print("\n--- Pas 2: Self-Attention Layer ---")
    attention_layer = SimpleSelfAttention(embed_dim)
    
    # Pas 3: Aplicăm attention
    print("\n--- Pas 3: Aplicăm Attention ---")
    output, attention_weights = attention_layer(embeddings)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Pas 4: Vizualizare
    visualize_attention(attention_weights, tokens)
    
    # Pas 5: Interpretare
    print("\n" + "=" * 60)
    print("INTERPRETARE")
    print("=" * 60)
    
    print("\n💡 Ce înseamnă attention weights?")
    print("   - Fiecare rând = un token se uită la ceilalți")
    print("   - Valori mari = mult context de acolo")
    print("   - Suma pe fiecare rând = 1.0 (probabilități)")
    
    # Verificare suma = 1
    for i, token in enumerate(tokens):
        suma = attention_weights[i].sum().item()
        print(f"\n'{token}' - suma attention: {suma:.4f} {'✓' if abs(suma - 1.0) < 0.01 else '✗'}")
    
    # Pas 6: Comparație înainte/după
    print("\n--- Comparație: Embedding → Attention Output ---")
    print("\nÎNAINTE (embedding original 'iubesc'):")
    print(embeddings[1])
    
    print("\nDUPĂ (cu context din 'Te' și 'dragă'):")
    print(output[1])
    
    print("\n💡 Output-ul e diferit pentru că acum 'iubesc' ÎNȚELEGE contextul!")
    print("   - A 'ascultat' ce spun 'Te' și 'dragă'")
    print("   - A integrat informația lor în înțelegerea sa")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Self-Attention funcționează! ✓")
    print("=" * 60)


# =============================================================================
# EXERCIȚIU AVANSAT: Multi-token attention
# =============================================================================

def your_exercise():
    """
    Exercițiu: Încearcă cu o propoziție mai lungă!
    
    Task:
    1. Creează embeddings pentru 5+ tokens
    2. Aplică self-attention
    3. Observă pattern-urile în attention weights
    4. Interpretează: care tokens acordă atenție cui?
    """
    print("\n" + "=" * 60)
    print("EXERCIȚIU TĂU")
    print("=" * 60)
    
    # TODO: Completează aici!
    # Exemplu:
    # tokens = ["Eu", "te", "iubesc", "foarte", "mult", "dragă"]
    # embeddings = torch.randn(6, 8)
    # ...
    
    print("\n💡 Completează funcția your_exercise() și observă:")
    print("   - Cuvinte apropiate acordă mai multă atenție între ele?")
    print("   - 'iubesc' se uită la subiect ('Eu') și obiect ('dragă')?")
    print("   - 'foarte' se uită la 'mult' (modificator)?")


# =============================================================================
# BONUS: Interpretare intuitivă
# =============================================================================

def intuitive_explanation():
    """
    Explicație intuitivă pentru Cezar.
    """
    print("\n" + "=" * 60)
    print("🌟 EXPLICAȚIE INTUITIVĂ: Ce face Self-Attention?")
    print("=" * 60)
    
    print("""
Înainte de Attention:
--------------------
"Te"     → [0.1, 0.5, ...]  (embeddings independent)
"iubesc" → [0.8, 0.2, ...]  (nu știe nimic despre "Te")
"dragă"  → [0.3, 0.9, ...]  (nu știe nimic despre "iubesc")

După Attention:
--------------
"Te"     → [0.1, 0.5, ...] + context("iubesc", "dragă")
"iubesc" → [0.8, 0.2, ...] + context("Te", "dragă")  
"dragă"  → [0.3, 0.9, ...] + context("Te", "iubesc")

Rezultat:
--------
✓ "iubesc" înțelege că "Te" e subiectul
✓ "iubesc" înțelege că "dragă" e obiectul
✓ Fiecare cuvânt are CONTEXT GLOBAL!

Analogie:
--------
ÎNAINTE: 3 persoane cu căști - nu se aud între ele
DUPĂ: Toți se aud - conversație contextuală!
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Explicație intuitivă
    intuitive_explanation()
    
    # Rulează testul
    test_self_attention()
    
    # Încearcă exercițiul
    # your_exercise()  # Decomentează când ești gata!
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEP: Multi-Head Attention")
    print("   (Vom vedea cum să rulăm attention în paralel!)")
    print("=" * 60)
