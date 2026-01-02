# Mini-Transformer - AI de la Zero

**Învățare hands-on:** Construim un transformer simplu pas cu pas.

## Autori
- Cezar Tipa (învățare + implementare)
- Sora (ghidare + explicații)

## Data
29 Decembrie 2025

## Ce Construim

Un transformer minimal care:
1. Învață din text simplu
2. Generează text nou
3. Înțelege pattern-uri de limbaj

## Structura

```
AI_Test/
├── README.md
├── data/
│   └── training_text.txt  # Textul de antrenament
├── step1_embeddings.py    # Pas 1: Token → Vectori
├── step2_attention.py     # Pas 2: Self-Attention
├── step3_feedforward.py   # Pas 3: Feed-Forward
├── step4_transformer.py   # Pas 4: Layer complet
├── step5_training.py      # Pas 5: Training loop
└── step6_generation.py    # Pas 6: Generare text
```

## Pași

### Pas 1: Embeddings ✓ (Next)
Transformăm cuvinte în vectori numerici.

### Pas 2: Self-Attention
Învățăm cum tokens "vorbesc" între ei.

### Pas 3: Feed-Forward
Procesăm informația din attention.

### Pas 4: Transformer Layer
Combinăm Attention + Feed-Forward + Residuals.

### Pas 5: Training
Învățăm modelul din date reale.

### Pas 6: Generation
Generăm text nou!

---

**Început:** 29 Decembrie 2025  
**Status:** Pas 1 - Embeddings
