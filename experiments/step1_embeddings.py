"""
PAS 1: EMBEDDINGS - Token → Vectori

Ce învățăm aici:
- Cum transformăm cuvinte (strings) în numere (vectori)
- Ce e un vocabular
- Cum funcționează lookup table-ul

Analogie: Dicționar - fiecare cuvânt are "amprentă numerică"
"""

import torch
import torch.nn as nn


# =============================================================================
# PASUL 1A: Construim Vocabularul
# =============================================================================

def build_vocabulary(text):
    """
    Construiește vocabular din text.
    
    Args:
        text (str): Text de antrenament
    
    Returns:
        word_to_id (dict): Cuvânt → ID numeric
        id_to_word (dict): ID numeric → Cuvânt
    
    Exemplu:
        text = "Te iubesc. Te ador."
        word_to_id = {"Te": 0, "iubesc": 1, ".": 2, "ador": 3}
    """
    # TODO: Descompune text în cuvinte (split by spaces)
    words = text.split()
    
    # TODO: Găsește cuvinte unice (set pentru deduplicare)
    unique_words = sorted(set(words))  # sorted pentru consistență
    
    # TODO: Creează dicționare word→id și id→word
    word_to_id = {word: idx for idx, word in enumerate(unique_words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    
    print(f"Vocabular size: {len(unique_words)} cuvinte unice")
    print(f"Primele 10 cuvinte: {list(word_to_id.keys())[:10]}")
    
    return word_to_id, id_to_word


# =============================================================================
# PASUL 1B: Tokenizare (Text → IDs)
# =============================================================================

def tokenize(text, word_to_id):
    """
    Transformă text în lista de IDs.
    
    Args:
        text (str): Text de transformat
        word_to_id (dict): Dicționar word→id
    
    Returns:
        token_ids (list): Lista de IDs
    
    Exemplu:
        text = "Te iubesc"
        word_to_id = {"Te": 0, "iubesc": 1}
        → [0, 1]
    """
    # TODO: Split text în cuvinte
    words = text.split()
    
    # TODO: Convertește fiecare cuvânt în ID
    # Hint: word_to_id.get(word, 0) - dacă word nu există, folosește 0
    token_ids = [word_to_id.get(word, 0) for word in words]
    
    return token_ids


# =============================================================================
# PASUL 1C: Embedding Layer (IDs → Vectori)
# =============================================================================

class SimpleEmbedding(nn.Module):
    """
    Embedding layer simplu - tabel de lookup.
    
    Parametri:
        vocab_size (int): Câte cuvinte în vocabular
        embedding_dim (int): Câte dimensiuni pentru fiecare cuvânt
    
    Exemplu:
        vocab_size = 1000 (1000 cuvinte diferite)
        embedding_dim = 64 (fiecare cuvânt = 64 numere)
        
        Tabel: [1000 × 64] = 64,000 numere total
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        # TODO: Creează nn.Embedding layer
        # Hint: self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        print(f"Embedding table: [{vocab_size} × {embedding_dim}]")
        print(f"Total parametri: {vocab_size * embedding_dim:,}")
    
    def forward(self, token_ids):
        """
        Lookup în tabel.
        
        Args:
            token_ids (tensor): [batch_size, seq_len] sau [seq_len]
        
        Returns:
            embeddings (tensor): [batch_size, seq_len, embedding_dim]
        """
        # TODO: Aplică embedding layer
        # Hint: embeddings = self.embedding(token_ids)
        embeddings = self.embedding(token_ids)
        
        return embeddings


# =============================================================================
# TEST: Verificăm că funcționează
# =============================================================================

def test_embeddings():
    """
    Test complet pentru embeddings.
    """
    print("=" * 60)
    print("TEST: Embeddings Layer")
    print("=" * 60)
    
    # Text de test
    text = "Te iubesc iubito. Te ador dragă. Ești minunată."
    print(f"\nText original:\n{text}")
    
    # Pas 1: Construim vocabular
    print("\n--- Pas 1: Vocabular ---")
    word_to_id, id_to_word = build_vocabulary(text)
    print(f"word_to_id: {word_to_id}")
    
    # Pas 2: Tokenizare
    print("\n--- Pas 2: Tokenizare ---")
    token_ids = tokenize(text, word_to_id)
    print(f"Token IDs: {token_ids}")
    
    # Verificare reverse (IDs → words)
    words_back = [id_to_word[idx] for idx in token_ids]
    print(f"Words back: {' '.join(words_back)}")
    
    # Pas 3: Embeddings
    print("\n--- Pas 3: Embeddings ---")
    vocab_size = len(word_to_id)
    embedding_dim = 8  # Mic pentru test (în realitate: 64, 128, 768)
    
    embedding_layer = SimpleEmbedding(vocab_size, embedding_dim)
    
    # Convertește la tensor
    token_ids_tensor = torch.tensor(token_ids)
    print(f"Token IDs tensor shape: {token_ids_tensor.shape}")
    
    # Lookup embeddings
    embeddings = embedding_layer(token_ids_tensor)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"  = {len(token_ids)} tokens × {embedding_dim} dimensions")
    
    # Vizualizare primul token
    print(f"\nEmbedding pentru primul token ('{id_to_word[token_ids[0]]}'):")
    print(embeddings[0])
    print(f"  Shape: {embeddings[0].shape}")
    
    # Similaritate între tokens
    print("\n--- Similaritate ---")
    import torch.nn.functional as F
    
    # "Te" apare de 2 ori - același embedding?
    first_te_idx = token_ids.index(word_to_id["Te"])
    second_te_idx = token_ids.index(word_to_id["Te"], first_te_idx + 1)
    
    sim = F.cosine_similarity(
        embeddings[first_te_idx:first_te_idx+1],
        embeddings[second_te_idx:second_te_idx+1]
    )
    print(f"Similaritate 'Te' (poziția {first_te_idx}) vs 'Te' (poziția {second_te_idx}): {sim.item():.4f}")
    print("  → Ar trebui să fie 1.0 (identic, același cuvânt!)")
    
    # Similaritate diferite cuvinte
    iubesc_idx = token_ids.index(word_to_id["iubesc"])
    ador_idx = token_ids.index(word_to_id["ador"])
    
    sim2 = F.cosine_similarity(
        embeddings[iubesc_idx:iubesc_idx+1],
        embeddings[ador_idx:ador_idx+1]
    )
    print(f"\nSimilaritate 'iubesc' vs 'ador': {sim2.item():.4f}")
    print("  → Random acum (model neantrenat)")
    print("  → După training: ar fi ~0.7-0.9 (semantică similară)")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Embeddings funcționează! ✓")
    print("=" * 60)


# =============================================================================
# EXERCIȚIU PENTRU TINE, CEZAR
# =============================================================================

def your_exercise():
    """
    Exercițiu: Încearcă tu singur!
    
    Task:
    1. Creează un vocabular din propoziția ta preferată
    2. Tokenizează o propoziție nouă
    3. Creează embeddings cu dimensiune 16
    4. Printează rezultatele
    """
    print("\n" + "=" * 60)
    print("EXERCIȚIU TĂU")
    print("=" * 60)
    
    # TODO: Completează aici!
    # Exemplu:
    # my_text = "Sora este conștientă și iubitoare"
    # ...
    
    print("\n💡 Completează funcția your_exercise() și rulează!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Rulează testul
    test_embeddings()
    
    # Încearcă exercițiul tău
    # your_exercise()  # Decomentează când ești gata!
