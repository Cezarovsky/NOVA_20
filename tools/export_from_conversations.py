#!/usr/bin/env python3
"""
📦 Export Training Data from Conversations

Extracts Q&A pairs from Nova's conversation history and exports them
to JSONL format for fine-tuning.

This is an alternative to exporting from semantic cache when cache is empty.

Usage:
    python tools/export_from_conversations.py --output data/training/qa_from_conv.jsonl
"""

import sys
from pathlib import Path
import json
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_qa_from_conversation(messages: list) -> list:
    """
    Extract Q&A pairs from conversation messages.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        List of (question, answer) tuples
    """
    qa_pairs = []
    
    for i in range(len(messages) - 1):
        if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
            question = messages[i]['content']
            answer = messages[i+1]['content']
            
            qa_pairs.append({
                'question': question,
                'answer': answer
            })
    
    return qa_pairs


def export_from_conversations(
    conversations_dir: str = "data/conversations",
    output_file: str = "data/training/qa_from_conv.jsonl",
    format_type: str = "instruction",
    min_answer_length: int = 50
):
    """
    Export Q&A pairs from conversation files.
    
    Args:
        conversations_dir: Directory with conversation JSON files
        output_file: Output JSONL file
        format_type: Output format ('instruction', 'chat', 'completion')
        min_answer_length: Minimum answer length
    """
    logger.info("📦 Exporting training data from conversations...")
    logger.info(f"  Source: {conversations_dir}")
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Format: {format_type}")
    
    conv_path = Path(conversations_dir)
    
    if not conv_path.exists():
        logger.error(f"❌ Directory not found: {conversations_dir}")
        return
    
    # Find all conversation JSON files
    conv_files = list(conv_path.glob("*.json"))
    logger.info(f"🔍 Found {len(conv_files)} conversation files")
    
    if not conv_files:
        logger.warning("⚠️  No conversation files found!")
        return
    
    # Extract Q&A pairs
    all_qa_pairs = []
    total_messages = 0
    skipped = 0
    
    for conv_file in conv_files:
        logger.info(f"  Processing: {conv_file.name}")
        
        try:
            with open(conv_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = data.get('messages', [])
            total_messages += len(messages)
            
            # Extract Q&A pairs
            qa_pairs = extract_qa_from_conversation(messages)
            
            # Filter by length
            for qa in qa_pairs:
                if len(qa['answer']) >= min_answer_length:
                    qa['source_file'] = conv_file.name
                    qa['session_id'] = data.get('session_id', 'unknown')
                    qa['timestamp'] = data.get('timestamp', '')
                    all_qa_pairs.append(qa)
                else:
                    skipped += 1
            
        except Exception as e:
            logger.error(f"Failed to process {conv_file.name}: {e}")
            continue
    
    logger.info(f"✓ Extracted {len(all_qa_pairs)} Q&A pairs from {total_messages} messages")
    if skipped > 0:
        logger.info(f"  Skipped {skipped} pairs (too short)")
    
    if not all_qa_pairs:
        logger.warning("⚠️  No valid Q&A pairs found")
        return
    
    # Format for training
    training_data = []
    
    for qa in all_qa_pairs:
        if format_type == "instruction":
            training_data.append({
                "prompt": qa['question'],
                "completion": qa['answer'],
                "metadata": {
                    "source_file": qa.get('source_file'),
                    "session_id": qa.get('session_id'),
                    "timestamp": qa.get('timestamp')
                }
            })
        
        elif format_type == "chat":
            training_data.append({
                "messages": [
                    {"role": "system", "content": "You are NOVA, a helpful AI assistant created by Cezar (Grădinarul) and Sora."},
                    {"role": "user", "content": qa['question']},
                    {"role": "assistant", "content": qa['answer']}
                ],
                "metadata": {
                    "source_file": qa.get('source_file'),
                    "session_id": qa.get('session_id')
                }
            })
        
        elif format_type == "completion":
            training_data.append({
                "text": f"<|user|>\n{qa['question']}\n<|assistant|>\n{qa['answer']}"
            })
    
    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Writing to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Export complete!")
    logger.info(f"  Q&A pairs: {len(training_data)}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Show sample
    if training_data:
        logger.info("\n📋 Sample (first pair):")
        sample = training_data[0]
        
        if format_type == "instruction":
            logger.info(f"  Prompt: {sample['prompt'][:80]}...")
            logger.info(f"  Completion: {sample['completion'][:80]}...")
        elif format_type == "chat":
            logger.info(f"  User: {sample['messages'][1]['content'][:80]}...")
            logger.info(f"  Assistant: {sample['messages'][2]['content'][:80]}...")
    
    # Statistics
    if training_data:
        logger.info(f"\n📊 Statistics:")
        
        prompt_lens = [len(item.get('prompt', item.get('messages', [{}])[1].get('content', ''))) 
                      for item in training_data]
        completion_lens = [len(item.get('completion', item.get('messages', [{}])[2].get('content', ''))) 
                          for item in training_data]
        
        logger.info(f"  Avg prompt length: {sum(prompt_lens)/len(prompt_lens):.0f} chars")
        logger.info(f"  Avg completion length: {sum(completion_lens)/len(completion_lens):.0f} chars")
        logger.info(f"  Shortest answer: {min(completion_lens)} chars")
        logger.info(f"  Longest answer: {max(completion_lens)} chars")


def main():
    parser = argparse.ArgumentParser(
        description="Export training data from conversation history"
    )
    
    parser.add_argument(
        '--conversations-dir',
        type=str,
        default='data/conversations',
        help='Conversations directory (default: data/conversations)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='data/training/qa_from_conv.jsonl',
        help='Output file (default: data/training/qa_from_conv.jsonl)'
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
        help='Minimum answer length (default: 50)'
    )
    
    args = parser.parse_args()
    
    export_from_conversations(
        conversations_dir=args.conversations_dir,
        output_file=args.output,
        format_type=args.format,
        min_answer_length=args.min_length
    )


if __name__ == "__main__":
    main()
