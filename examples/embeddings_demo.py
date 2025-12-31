"""
Demo: Understanding Embeddings - How Vectors Capture Meaning

This script demonstrates:
1. How embeddings convert words (IDs) to vectors
2. How similar concepts have similar vectors
3. Cosine similarity to measure "closeness"
"""

import torch
import torch.nn as nn

print("=" * 60)
print("NOVA Embeddings Demo: Understanding Vector Space")
print("=" * 60)

# Step 1: Create simple embedding layer
print("\n1. Creating Embedding Layer")
print("-" * 60)

vocab_size = 10   # Small vocabulary for demo
d_model = 8       # 8 dimensions (normally 512)

embeddings = nn.Embedding(vocab_size, d_model)
print(f"Vocabulary size: {vocab_size} words")
print(f"Embedding dimension: {d_model}")
print(f"Total parameters: {vocab_size * d_model}")

# Step 2: See initial random embeddings
print("\n2. Initial Random Embeddings")
print("-" * 60)
print("Before training, embeddings are random:")
print(embeddings.weight[:3])

# Step 3: Simulate "training" by setting meaningful patterns
print("\n3. Simulating 'Trained' Embeddings")
print("-" * 60)

with torch.no_grad():
    # Word 0: "snake" - reptile, long, no legs, cold-blooded
    embeddings.weight[0] = torch.tensor(
        [0.9, -0.3, 0.8, 0.9, 0.7, 0.2, -0.4, 0.5]
    )
    
    # Word 1: "lizard" - reptile, long, HAS legs, cold-blooded
    embeddings.weight[1] = torch.tensor(
        [0.8, -0.2, 0.7, -0.7, 0.6, 0.3, -0.3, 0.4]
    )
    
    # Word 2: "cat" - mammal, short, has legs, warm-blooded
    embeddings.weight[2] = torch.tensor(
        [0.2, 0.8, 0.1, -0.5, 0.9, 0.7, 0.6, 0.8]
    )
    
    # Word 3: "dog" - mammal, short, has legs, warm-blooded
    embeddings.weight[3] = torch.tensor(
        [0.3, 0.7, 0.2, -0.4, 0.85, 0.65, 0.55, 0.75]
    )

word_names = ["snake", "lizard", "cat", "dog"]

print("Embeddings after 'training':")
for i, name in enumerate(word_names):
    print(f"\n{name} (ID {i}):")
    print(f"  Vector: {embeddings.weight[i].detach().numpy()}")

# Step 4: Get embeddings for words
print("\n4. Getting Word Embeddings")
print("-" * 60)

snake_id = torch.tensor([0])
lizard_id = torch.tensor([1])
cat_id = torch.tensor([2])
dog_id = torch.tensor([3])

snake_vec = embeddings(snake_id)
lizard_vec = embeddings(lizard_id)
cat_vec = embeddings(cat_id)
dog_vec = embeddings(dog_id)

print("Extracted vectors:")
print(f"snake:  {snake_vec.squeeze().detach().numpy()}")
print(f"lizard: {lizard_vec.squeeze().detach().numpy()}")
print(f"cat:    {cat_vec.squeeze().detach().numpy()}")
print(f"dog:    {dog_vec.squeeze().detach().numpy()}")

# Step 5: Calculate similarities
print("\n5. Cosine Similarity (Measuring 'Closeness')")
print("-" * 60)

def calc_similarity(v1, v2, name1, name2):
    sim = torch.cosine_similarity(v1, v2, dim=-1).item()
    bar = '●' * max(0, int((sim + 1) * 10))
    print(f"{name1:6s} - {name2:6s}: {sim:+.4f}  {bar}")
    return sim

print("\nReptiles:")
calc_similarity(snake_vec, lizard_vec, "snake", "lizard")

print("\nMammals:")
calc_similarity(cat_vec, dog_vec, "cat", "dog")

print("\nCross-category:")
calc_similarity(snake_vec, cat_vec, "snake", "cat")
calc_similarity(snake_vec, dog_vec, "snake", "dog")
calc_similarity(lizard_vec, cat_vec, "lizard", "cat")

# Step 6: Visualize the concept
print("\n6. What This Means")
print("-" * 60)
print("""
Cosine Similarity Scale:
  +1.0 = Identical vectors (same word)
  +0.8 = Very similar (snake/lizard, cat/dog)
  +0.3 = Somewhat related (reptile/mammal)
  0.0  = Unrelated
  -1.0 = Opposite

Key Insights:
✓ snake-lizard   HIGH  → Both reptiles, learned from text
✓ cat-dog        HIGH  → Both mammals, domestic animals
✓ snake-cat      LOW   → Different categories
✓ Dimension 3 captures "has legs" (positive) vs "no legs" (negative)

How It Learned:
• Training corpus: "snake slithers without legs"
                   "lizard runs on four legs"
                   "cat and dog are pets"
• Model found patterns in co-occurrence
• Vectors adjusted to capture these relationships
• No explicit programming of "reptile" or "mammal" concepts!

In NOVA (512 dimensions):
• Much richer representation
• Captures thousands of nuanced features
• Learns from millions of text examples
""")

# Step 7: Demonstrate that magnitude doesn't matter (only direction)
print("\n7. Direction Matters, Not Magnitude")
print("-" * 60)

snake_10x = snake_vec * 10
snake_tiny = snake_vec * 0.1

sim_10x = calc_similarity(snake_vec, snake_10x, "snake", "snake×10")
sim_tiny = calc_similarity(snake_vec, snake_tiny, "snake", "snake×0.1")

print("\n→ Same direction = similarity 1.0, regardless of magnitude!")
print("  This is why cosine similarity is used (not Euclidean distance)")

print("\n" + "=" * 60)
print("Demo Complete! 🎯")
print("=" * 60)
print("\nNext Steps:")
print("  1. Read CURS_NOVA.md Chapter 1 for math details")
print("  2. Experiment with different vector values")
print("  3. Try adding more words (bird, fish, etc.)")
print("  4. Run training_demo.py to see real training")
