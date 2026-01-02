#!/usr/bin/env python3
"""
📦 Export Training Data from Semantic Cache

Exports Q&A pairs from Nova's semantic cache to JSONL format
for fine-tuning local LLMs (Mistral, Llama, etc.)

Usage:
    python tools/export_training_data.py --output data/training/qa_pairs.jsonl
    python tools/export_training_data.py --min-length 100 --format chat
"""

import sys
from pathlib import Path
import json
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_qa_text(qa_text: str) -> dict:
    """
    Parse Q&A text format from semantic cache.
    
    Format: "Q: question\nA: answer"
    
    Returns:
        Dict with 'question' and 'answer', or None if invalid
    """
    try:
        lines = qa_text.strip().split('\n', 1)
        
        if len(lines) < 2:
            return None
        
        question_line = lines[0]
        answer_line = lines[1]
        
        if not question_line.startswith('Q: ') or not answer_line.startswith('A: '):
            return None
        
        return {
            'question': question_line[3:].strip(),
            'answer': answer_line[3:].strip()
        }
    
    except Exception as e:
        logger.error(f"Failed to parse Q&A: {e}")
        return None


def export_training_data(
    persist_directory: str = "data/chroma_db",
    collection_name: str = "nova_knowledge",
    output_file: str = "data/training/qa_pairs.jsonl",
    format_type: str = "instruction",
    min_answer_length: int = 50,
    min_quality_score: float = 0.0
):
    """
    Export Q&A pairs from semantic cache.
    
    Args:
        persist_directory: ChromaDB persist directory
        collection_name: Collection name
        output_file: Output JSONL file path
        format_type: Output format ('instruction', 'chat', 'completion')
        min_answer_length: Minimum answer length to export
        min_quality_score: Minimum quality score (0-1)
    """
    logger.info("📦 Starting training data export...")
    logger.info(f"  Source: {persist_directory}/{collection_name}")
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Format: {format_type}")
    logger.info(f"  Min answer length: {min_answer_length} chars")
    
    # Initialize vector store
    try:
        vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize vector store: {e}")
        return
    
    # Get all cached Q&A documents
    logger.info("🔍 Fetching cached Q&A pairs...")
    
    try:
        # Get all documents with cached_qa metadata
        results = vector_store.get(
            where={"type": "cached_qa"}
        )
        
        if not results or not results.get('documents'):
            logger.warning("⚠️  No cached Q&A pairs found!")
            logger.info("💡 Tip: Chat with Nova first to build up the cache")
            return
        
        total_docs = len(results['documents'])
        logger.info(f"✓ Found {total_docs} cached Q&A pairs")
        
    except Exception as e:
        # Fallback: Get all documents if metadata filtering fails
        logger.warning(f"Metadata filtering failed, getting all documents: {e}")
        results = vector_store.get()
        total_docs = len(results['documents']) if results else 0
        logger.info(f"✓ Found {total_docs} total documents (some may not be Q&A)")
    
    # Parse and filter Q&A pairs
    qa_pairs = []
    skipped = 0
    
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
        # Parse Q&A format
        qa = parse_qa_text(doc)
        
        if not qa:
            skipped += 1
            continue
        
        # Filter by answer length
        if len(qa['answer']) < min_answer_length:
            logger.debug(f"Skipped (too short): {qa['question'][:40]}...")
            skipped += 1
            continue
        
        # Add metadata
        qa['metadata'] = metadata
        qa['export_date'] = datetime.now().isoformat()
        
        qa_pairs.append(qa)
    
    logger.info(f"✓ Parsed {len(qa_pairs)} valid Q&A pairs")
    if skipped > 0:
        logger.info(f"  Skipped {skipped} items (invalid or too short)")
    
    if not qa_pairs:
        logger.warning("⚠️  No valid Q&A pairs to export after filtering")
        return
    
    # Format for fine-tuning
    training_data = []
    
    for qa in qa_pairs:
        if format_type == "instruction":
            # Instruction format for Mistral/Llama
            training_data.append({
                "prompt": qa['question'],
                "completion": qa['answer'],
                "metadata": qa.get('metadata', {})
            })
        
        elif format_type == "chat":
            # Chat format with system message
            training_data.append({
                "messages": [
                    {"role": "system", "content": "You are NOVA, a helpful AI assistant."},
                    {"role": "user", "content": qa['question']},
                    {"role": "assistant", "content": qa['answer']}
                ],
                "metadata": qa.get('metadata', {})
            })
        
        elif format_type == "completion":
            # Simple completion format
            training_data.append({
                "text": f"<|user|>\n{qa['question']}\n<|assistant|>\n{qa['answer']}"
            })
        
        else:
            logger.error(f"Unknown format: {format_type}")
            return
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL
    logger.info(f"💾 Writing to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Export complete!")
    logger.info(f"  Total Q&A pairs: {len(training_data)}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Show sample
    if training_data:
        logger.info("\n📋 Sample (first item):")
        sample = training_data[0]
        
        if format_type == "instruction":
            logger.info(f"  Prompt: {sample['prompt'][:80]}...")
            logger.info(f"  Completion: {sample['completion'][:80]}...")
        elif format_type == "chat":
            logger.info(f"  User: {sample['messages'][1]['content'][:80]}...")
            logger.info(f"  Assistant: {sample['messages'][2]['content'][:80]}...")
        else:
            logger.info(f"  Text: {sample['text'][:120]}...")
    
    # Statistics
    if training_data:
        avg_prompt_len = sum(len(item.get('prompt', '')) for item in training_data) / len(training_data)
        avg_completion_len = sum(len(item.get('completion', '')) for item in training_data) / len(training_data)
        
        logger.info(f"\n📊 Statistics:")
        logger.info(f"  Avg prompt length: {avg_prompt_len:.0f} chars")
        logger.info(f"  Avg completion length: {avg_completion_len:.0f} chars")
        
        # Check language distribution (simple heuristic)
        romanian_count = sum(1 for item in training_data 
                           if any(word in item.get('completion', '').lower() 
                                  for word in ['este', 'sunt', 'pentru', 'cu', 'la']))
        
        logger.info(f"  Romanian pairs: ~{romanian_count} ({romanian_count/len(training_data)*100:.0f}%)")
        logger.info(f"  English pairs: ~{len(training_data)-romanian_count} ({(len(training_data)-romanian_count)/len(training_data)*100:.0f}%)")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Export training data from Nova's semantic cache"
    )
    
    parser.add_argument(
        '--persist-dir',
        type=str,
        default='data/chroma_db',
        help='ChromaDB persist directory (default: data/chroma_db)'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        default='nova_knowledge',
        help='Collection name (default: nova_knowledge)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='data/training/qa_pairs.jsonl',
        help='Output JSONL file (default: data/training/qa_pairs.jsonl)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['instruction', 'chat', 'completion'],
        default='instruction',
        help='Output format (default: instruction)'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Minimum answer length in characters (default: 50)'
    )
    
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.0,
        help='Minimum quality score 0-1 (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    export_training_data(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        output_file=args.output,
        format_type=args.format,
        min_answer_length=args.min_length,
        min_quality_score=args.min_quality
    )


if __name__ == "__main__":
    main()
