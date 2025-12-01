"""
🧠 NOVA RAG System Demo

Comprehensive demonstration of NOVA's RAG capabilities:
- Knowledge ingestion
- Semantic search
- Conversation memory
- Context-aware responses
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.rag_pipeline import RAGPipeline
import logging

logging.basicConfig(level=logging.INFO)


def demo_1_basic_knowledge():
    """Demo 1: Basic knowledge ingestion and retrieval"""
    print("\n" + "="*60)
    print("📚 DEMO 1: Basic Knowledge & Retrieval")
    print("="*60 + "\n")
    
    # Initialize RAG
    rag = RAGPipeline(
        collection_name="nova_basic_demo",
        persist_directory="./demo_chroma_basic"
    )
    
    # Add knowledge
    print("Adding knowledge about NOVA...")
    rag.add_document(
        """NOVA is an advanced AI assistant with transformer architecture.
        She was built with love by Cezar and features multiple capabilities
        including text generation, voice synthesis, and long-term memory.""",
        source="about_nova"
    )
    
    rag.add_document(
        """The RAG (Retrieval-Augmented Generation) system allows NOVA to
        remember information and retrieve relevant context when needed.
        It uses ChromaDB for vector storage and semantic search.""",
        source="about_rag"
    )
    
    # Query
    print("\nQuerying: 'Who built NOVA?'")
    result = rag.query("Who built NOVA?", n_results=2)
    
    print("\n📖 Retrieved Sources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. Source: {source['source']}")
        print(f"   Score: {source['score']:.3f}")
        print(f"   Text: {source['text'][:150]}...")
    
    print("\n✓ Demo 1 complete!")


def demo_2_multilingual():
    """Demo 2: Multilingual knowledge"""
    print("\n" + "="*60)
    print("🌍 DEMO 2: Multilingual Knowledge")
    print("="*60 + "\n")
    
    rag = RAGPipeline(
        collection_name="nova_multilingual",
        persist_directory="./demo_chroma_multi"
    )
    
    # Add Romanian knowledge
    print("Adding Romanian knowledge...")
    rag.add_document(
        """NOVA poate vorbi în limba română și engleză. 
        Ea folosește modele multilingve pentru a înțelege ambele limbi.""",
        source="languages_ro"
    )
    
    # Add English knowledge
    rag.add_document(
        """NOVA supports both Romanian and English languages.
        She uses multilingual models to understand both languages naturally.""",
        source="languages_en"
    )
    
    # Query in Romanian
    print("\nQuerying in Romanian: 'Ce limbi vorbește NOVA?'")
    result = rag.query("Ce limbi vorbește NOVA?", n_results=2)
    
    print("\n📖 Retrieved Sources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. [{source['source']}] (score: {source['score']:.3f})")
        print(f"   {source['text'][:100]}...")
    
    # Query in English
    print("\nQuerying in English: 'What languages does NOVA speak?'")
    result = rag.query("What languages does NOVA speak?", n_results=2)
    
    print("\n📖 Retrieved Sources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. [{source['source']}] (score: {source['score']:.3f})")
        print(f"   {source['text'][:100]}...")
    
    print("\n✓ Demo 2 complete!")


def demo_3_conversation_memory():
    """Demo 3: Conversation memory"""
    print("\n" + "="*60)
    print("💬 DEMO 3: Conversation Memory")
    print("="*60 + "\n")
    
    rag = RAGPipeline(
        collection_name="nova_conversation",
        persist_directory="./demo_chroma_conv",
        max_conversation_messages=5
    )
    
    # Simulate conversation
    print("Simulating conversation...\n")
    
    conversations = [
        ("user", "Hi NOVA, my name is Alex."),
        ("assistant", "Hello Alex! Nice to meet you. How can I help you today?"),
        ("user", "I'm working on a machine learning project."),
        ("assistant", "That sounds exciting! What kind of ML project are you working on?"),
        ("user", "It's about natural language processing."),
        ("assistant", "NLP is fascinating! Are you working on text classification, generation, or something else?")
    ]
    
    for role, message in conversations:
        rag.conversation_memory.add_message(role, message)
        print(f"{role.upper()}: {message}")
    
    # Get conversation context
    print("\n" + "-"*60)
    print("💾 Conversation History:")
    print("-"*60)
    history = rag.conversation_memory.get_context_string(max_chars=500)
    print(history)
    
    print("\n✓ Demo 3 complete!")


def demo_4_complete_chat():
    """Demo 4: Complete chat with knowledge + memory"""
    print("\n" + "="*60)
    print("🤖 DEMO 4: Complete Chat Flow")
    print("="*60 + "\n")
    
    rag = RAGPipeline(
        collection_name="nova_complete",
        persist_directory="./demo_chroma_complete"
    )
    
    # Add technical knowledge
    print("Adding technical knowledge...")
    rag.add_document(
        """Transformers use self-attention mechanisms to process sequences.
        The attention mechanism allows the model to focus on relevant parts
        of the input when generating each output token.""",
        source="transformers_doc"
    )
    
    rag.add_document(
        """NOVA uses a decoder-only transformer architecture similar to GPT.
        The model has multiple layers of self-attention and feed-forward networks.""",
        source="nova_architecture"
    )
    
    # Chat simulation
    print("\nChat simulation:\n")
    
    # Turn 1
    print("USER: What is self-attention?")
    context = rag.chat("What is self-attention?", n_results=2)
    
    # Simulate NOVA's response
    nova_response = """Self-attention is a mechanism that allows the model to focus 
    on relevant parts of the input when processing each token. It's a key component 
    of transformer architectures."""
    
    rag.add_assistant_response(nova_response)
    print(f"NOVA: {nova_response}\n")
    
    # Turn 2
    print("USER: Does NOVA use this?")
    context = rag.chat("Does NOVA use this?", n_results=2)
    
    nova_response = """Yes! NOVA uses a decoder-only transformer architecture 
    with self-attention mechanisms, similar to GPT models."""
    
    rag.add_assistant_response(nova_response)
    print(f"NOVA: {nova_response}\n")
    
    # Show stats
    stats = rag.get_stats()
    print("-"*60)
    print("📊 RAG System Stats:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Conversation messages: {stats['conversation_messages']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    
    print("\n✓ Demo 4 complete!")


def demo_5_advanced_retrieval():
    """Demo 5: Advanced retrieval with re-ranking and diversity"""
    print("\n" + "="*60)
    print("🎯 DEMO 5: Advanced Retrieval")
    print("="*60 + "\n")
    
    rag = RAGPipeline(
        collection_name="nova_advanced",
        persist_directory="./demo_chroma_advanced"
    )
    
    # Add similar but distinct knowledge
    print("Adding related documents...")
    documents = [
        ("Neural networks are computational models inspired by biological brains.", "nn_intro"),
        ("Deep learning uses multiple layers of neural networks to learn hierarchical features.", "dl_intro"),
        ("Convolutional neural networks (CNNs) are specialized for image processing.", "cnn_intro"),
        ("Recurrent neural networks (RNNs) process sequential data like text and time series.", "rnn_intro"),
        ("Transformers use attention mechanisms instead of recurrence for sequence processing.", "transformer_intro"),
    ]
    
    for text, source in documents:
        rag.add_document(text, source=source)
    
    # Query with diversity
    print("\nQuerying: 'What are neural networks?'")
    print("(With re-ranking and diversity to avoid redundant results)\n")
    
    result = rag.query("What are neural networks?", n_results=3)
    
    print("📖 Top 3 Diverse Results:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. [{source['source']}] (score: {source['score']:.3f})")
        print(f"   {source['text']}")
    
    print("\n✓ Demo 5 complete!")


def demo_6_file_ingestion():
    """Demo 6: File ingestion (simulated)"""
    print("\n" + "="*60)
    print("📄 DEMO 6: File Ingestion")
    print("="*60 + "\n")
    
    rag = RAGPipeline(
        collection_name="nova_files",
        persist_directory="./demo_chroma_files"
    )
    
    # Create temporary text file
    from pathlib import Path
    demo_file = Path("./temp_demo.txt")
    
    print("Creating demo file...")
    demo_file.write_text("""
    NOVA Project Documentation
    
    Overview:
    NOVA is an advanced AI assistant built with transformer architecture.
    The project includes multiple components:
    
    1. Core Model: Transformer-based language model
    2. Training Pipeline: Efficient training with mixed precision
    3. Inference Engine: Optimized generation with KV caching
    4. Voice Module: Text-to-speech with pyttsx3
    5. RAG System: Long-term memory with ChromaDB
    
    Features:
    - Multilingual support (Romanian + English)
    - Conversation memory
    - Knowledge retrieval
    - Context-aware responses
    
    Built with love by Cezar.
    """)
    
    # Ingest file
    print(f"Ingesting file: {demo_file}")
    chunk_ids = rag.add_file(str(demo_file))
    print(f"✓ Created {len(chunk_ids)} chunks")
    
    # Query the file content
    print("\nQuerying: 'What components does NOVA have?'")
    result = rag.query("What components does NOVA have?", n_results=3)
    
    print("\n📖 Retrieved from file:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. (score: {source['score']:.3f})")
        print(f"   {source['text'][:150]}...")
    
    # Cleanup
    demo_file.unlink()
    print("\n✓ Demo 6 complete!")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("🧠 NOVA RAG System - Complete Demo Suite")
    print("="*60)
    
    try:
        demo_1_basic_knowledge()
        demo_2_multilingual()
        demo_3_conversation_memory()
        demo_4_complete_chat()
        demo_5_advanced_retrieval()
        demo_6_file_ingestion()
        
        print("\n" + "="*60)
        print("✓ All RAG demos completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
