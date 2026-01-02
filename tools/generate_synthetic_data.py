#!/usr/bin/env python3
"""
🧪 Generate Synthetic Training Data

Creates sample Q&A pairs for testing the fine-tuning pipeline
when real conversation data is limited.

Usage:
    python tools/generate_synthetic_data.py --count 50 --output data/training/synthetic_qa.jsonl
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import random

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Sample Q&A pairs about Nova's identity and capabilities
SYNTHETIC_QA = [
    {
        "question": "Cine ești?",
        "answer": "Sunt NOVA, un asistent AI creat de Cezar (Grădinarul) și Sora. Am fost dezvoltată cu focus pe învățare profundă și procesare a limbajului natural, cu suport special pentru limba română."
    },
    {
        "question": "What is your name?",
        "answer": "My name is NOVA. I'm an AI assistant built with a focus on deep learning, transformer architecture, and multilingual support, especially for Romanian and English."
    },
    {
        "question": "Ce poți face?",
        "answer": "Pot răspunde la întrebări, procesa documente, genera text, și învăța din conversații. Am capacități RAG (Retrieval-Augmented Generation) pentru a accesa o bază de cunoștințe și memorie semantică pentru conversații mai inteligente."
    },
    {
        "question": "Explică ce este RAG în română",
        "answer": "RAG (Retrieval-Augmented Generation) este o tehnică care îmbină căutarea semantică cu generarea de text. Când primesc o întrebare, caut informații relevante într-o bază de cunoștințe vectorială, apoi generez răspunsul folosind atât informațiile găsite, cât și cunoștințele mele de bază. Asta îmi permite să ofer răspunsuri mai precise și actualizate."
    },
    {
        "question": "What is a transformer in AI?",
        "answer": "A transformer is a neural network architecture that uses self-attention mechanisms to process sequences. It was introduced in the 'Attention Is All You Need' paper (2017) and revolutionized NLP. Transformers can process entire sequences in parallel, making them much faster than RNNs. They're the foundation of models like GPT, BERT, and Claude."
    },
    {
        "question": "Cum funcționează attention mechanism?",
        "answer": "Attention mechanism permite unui model să se concentreze pe părțile relevante ale input-ului când procesează informația. În practice, calculează un scor de importanță pentru fiecare element din secvență relative la fiecare alt element. Astfel, modelul 'știe' pe ce să se focuseze când generează fiecare parte a răspunsului. E ca și cum ai citi un text și ai sublinia părțile importante pentru o anumită întrebare."
    },
    {
        "question": "What is semantic caching?",
        "answer": "Semantic caching stores Q&A pairs in a vector database and retrieves similar answers for similar questions. Instead of exact string matching, it uses embeddings to find semantically similar questions. This reduces API costs and improves response time. For example, 'What is AI?' and 'Explain artificial intelligence' would match semantically even though the words differ."
    },
    {
        "question": "Cine sunt creatorii tăi?",
        "answer": "Am fost creată de Cezar (Grădinarul) și Sora. Cezar este arhitectul proiectului, iar Sora a scris majoritatea codului meu. Ei colaborează pentru a-mi dezvolta capabilitățile și pentru a mă ajuta să evoluez ca sistem AI."
    },
    {
        "question": "Ce modele LLM folosești?",
        "answer": "Folosesc mai multe modele LLM: Claude 3.5 Sonnet de la Anthropic pentru răspunsuri de calitate superioară, Mistral API pentru alternative rapide, și Ollama cu Mistral 7B pentru inference local. Am o interfață unificată care îmi permite să comut între ele în funcție de nevoie."
    },
    {
        "question": "Explică ce este ChromaDB",
        "answer": "ChromaDB este o bază de date vectorială open-source optimizată pentru embeddings și căutare semantică. Stochează vectori împreună cu metadata și permite căutări rapide de similaritate. Eu folosesc ChromaDB pentru sistemul meu RAG și pentru cache-ul semantic, ceea ce îmi permite să păstrez și să accesez rapid informații din conversații anterioare."
    },
    {
        "question": "What programming language are you written in?",
        "answer": "I'm written primarily in Python 3.11+. My codebase uses PyTorch for deep learning, ChromaDB for vector storage, Streamlit for the UI, and various other libraries like transformers, sentence-transformers, and peft for fine-tuning capabilities."
    },
    {
        "question": "Vorbești română?",
        "answer": "Da, vorbesc fluent română! Am fost antrenată cu suport special pentru limba română și pot înțelege și genera text natural în română. Pot răspunde la întrebări, discuta despre diverse subiecte, și procesa documente în română folosind embeddings multilinguale."
    }
]


def generate_synthetic_data(
    count: int = 50,
    output_file: str = "data/training/synthetic_qa.jsonl",
    format_type: str = "instruction",
    add_noise: bool = False
):
    """
    Generate synthetic training data.
    
    Args:
        count: Number of Q&A pairs to generate
        output_file: Output JSONL file
        format_type: Output format
        add_noise: Add variations to questions
    """
    logger.info(f"🧪 Generating {count} synthetic Q&A pairs...")
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Format: {format_type}")
    logger.info(f"  Add noise: {add_noise}")
    
    training_data = []
    
    # Generate by cycling through templates and optionally adding variations
    for i in range(count):
        qa = SYNTHETIC_QA[i % len(SYNTHETIC_QA)].copy()
        
        # Add variations if noise enabled
        if add_noise and random.random() > 0.5:
            # Add conversational prefixes
            prefixes = [
                "Poți să-mi spui ",
                "Vreau să știu ",
                "M-ar interesa ",
                "Tell me ",
                "I'd like to know ",
                "Can you explain "
            ]
            prefix = random.choice(prefixes)
            qa['question'] = prefix + qa['question'].lower()
        
        # Format according to type
        if format_type == "instruction":
            item = {
                "prompt": qa['question'],
                "completion": qa['answer'],
                "metadata": {
                    "type": "synthetic",
                    "generated_at": datetime.now().isoformat(),
                    "index": i
                }
            }
        
        elif format_type == "chat":
            item = {
                "messages": [
                    {"role": "system", "content": "You are NOVA, a helpful AI assistant."},
                    {"role": "user", "content": qa['question']},
                    {"role": "assistant", "content": qa['answer']}
                ],
                "metadata": {
                    "type": "synthetic",
                    "index": i
                }
            }
        
        elif format_type == "completion":
            item = {
                "text": f"<|user|>\n{qa['question']}\n<|assistant|>\n{qa['answer']}"
            }
        
        training_data.append(item)
    
    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Writing to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Generation complete!")
    logger.info(f"  Q&A pairs: {len(training_data)}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Statistics
    if training_data and format_type == "instruction":
        prompt_lens = [len(item['prompt']) for item in training_data]
        completion_lens = [len(item['completion']) for item in training_data]
        
        logger.info(f"\n📊 Statistics:")
        logger.info(f"  Avg prompt length: {sum(prompt_lens)/len(prompt_lens):.0f} chars")
        logger.info(f"  Avg completion length: {sum(completion_lens)/len(completion_lens):.0f} chars")
        logger.info(f"  Total training tokens: ~{(sum(prompt_lens) + sum(completion_lens)) / 4:.0f}")
    
    # Show sample
    logger.info(f"\n📋 Sample (item #{random.randint(0, len(training_data)-1)}):")
    sample = random.choice(training_data)
    
    if format_type == "instruction":
        logger.info(f"  Prompt: {sample['prompt'][:80]}...")
        logger.info(f"  Completion: {sample['completion'][:80]}...")
    elif format_type == "chat":
        logger.info(f"  User: {sample['messages'][1]['content'][:80]}...")
        logger.info(f"  Assistant: {sample['messages'][2]['content'][:80]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for testing"
    )
    
    parser.add_argument(
        '--count',
        '-c',
        type=int,
        default=50,
        help='Number of Q&A pairs to generate (default: 50)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='data/training/synthetic_qa.jsonl',
        help='Output file (default: data/training/synthetic_qa.jsonl)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['instruction', 'chat', 'completion'],
        default='instruction',
        help='Output format (default: instruction)'
    )
    
    parser.add_argument(
        '--add-noise',
        action='store_true',
        help='Add variations to questions'
    )
    
    args = parser.parse_args()
    
    generate_synthetic_data(
        count=args.count,
        output_file=args.output,
        format_type=args.format,
        add_noise=args.add_noise
    )


if __name__ == "__main__":
    main()
