#!/usr/bin/env python3
"""
🔬 Compare Model Quality: Mistral (Ollama) vs Claude

Benchmark both models on same questions to identify quality gaps
and opportunities for fine-tuning.

Usage:
    python tests/compare_models.py --input data/training/synthetic_qa.jsonl --count 10
    python tests/compare_models.py --questions custom_questions.txt --output comparison.md
"""

import sys
from pathlib import Path
import json
import argparse
import logging
from typing import List, Dict
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm_interface import LLMInterface, LLMProvider

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_questions_from_jsonl(file_path: str, count: int = None) -> List[Dict]:
    """Load questions from JSONL training data"""
    questions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if count and i >= count:
                break
            
            data = json.loads(line)
            
            # Extract question based on format
            if 'prompt' in data:
                question = data['prompt']
            elif 'messages' in data:
                # Find user message
                for msg in data['messages']:
                    if msg['role'] == 'user':
                        question = msg['content']
                        break
            else:
                continue
            
            questions.append({
                'question': question,
                'expected': data.get('completion', ''),
                'metadata': data.get('metadata', {})
            })
    
    return questions


def load_questions_from_txt(file_path: str) -> List[Dict]:
    """Load questions from plain text file (one per line)"""
    questions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                questions.append({'question': line})
    
    return questions


def compare_responses(
    questions: List[Dict],
    claude_model: str = "claude-3-5-sonnet-20241022",
    mistral_model: str = "mistral",
    max_tokens: int = 500
) -> List[Dict]:
    """
    Compare responses from both models.
    
    Returns list of comparison results.
    """
    logger.info(f"🔬 Comparing {len(questions)} questions...")
    logger.info(f"  Claude: {claude_model}")
    logger.info(f"  Mistral: {mistral_model} (via Ollama)")
    
    # Initialize both LLMs
    try:
        llm_claude = LLMInterface(
            provider=LLMProvider.ANTHROPIC,
            model=claude_model
        )
        logger.info("✓ Claude initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Claude: {e}")
        logger.error("Make sure ANTHROPIC_API_KEY is set in .env")
        return []
    
    try:
        llm_mistral = LLMInterface(
            provider=LLMProvider.OLLAMA,
            model=mistral_model
        )
        logger.info("✓ Mistral (Ollama) initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Mistral: {e}")
        logger.error("Make sure Ollama is running: brew services start ollama")
        return []
    
    results = []
    
    for i, q in enumerate(questions, 1):
        question = q['question']
        logger.info(f"\n[{i}/{len(questions)}] Testing: {question[:60]}...")
        
        result = {
            'question': question,
            'expected': q.get('expected', ''),
            'metadata': q.get('metadata', {})
        }
        
        # Get Claude response
        try:
            logger.info("  📤 Asking Claude...")
            claude_response = llm_claude.generate(
                prompt=question,
                temperature=0.7,
                max_tokens=max_tokens
            )
            
            result['claude'] = {
                'text': claude_response.text,
                'tokens': claude_response.usage.get('total_tokens', 0),
                'latency_ms': claude_response.latency_ms,
                'cost_estimate': estimate_cost(claude_response.usage, 'claude')
            }
            
            logger.info(f"  ✓ Claude: {claude_response.usage.get('total_tokens', 0)} tokens, {claude_response.latency_ms}ms")
            
        except Exception as e:
            logger.error(f"  ❌ Claude failed: {e}")
            result['claude'] = {'error': str(e)}
        
        # Small delay to avoid rate limits
        time.sleep(1)
        
        # Get Mistral response
        try:
            logger.info("  📤 Asking Mistral...")
            mistral_response = llm_mistral.generate(
                prompt=question,
                temperature=0.7,
                max_tokens=max_tokens
            )
            
            result['mistral'] = {
                'text': mistral_response.text,
                'tokens': mistral_response.usage.get('total_tokens', 0),
                'latency_ms': mistral_response.latency_ms,
                'cost_estimate': 0  # Local = free
            }
            
            logger.info(f"  ✓ Mistral: {mistral_response.usage.get('total_tokens', 0)} tokens, {mistral_response.latency_ms}ms")
            
        except Exception as e:
            logger.error(f"  ❌ Mistral failed: {e}")
            result['mistral'] = {'error': str(e)}
        
        results.append(result)
        
        # Small delay between questions
        time.sleep(0.5)
    
    return results


def estimate_cost(usage: Dict, provider: str) -> float:
    """Estimate API cost based on usage"""
    if provider == 'claude':
        # Claude 3.5 Sonnet pricing (approximate)
        # $3 per million input tokens, $15 per million output tokens
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        cost = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)
        return cost
    
    return 0.0


def generate_markdown_report(results: List[Dict], output_file: str):
    """Generate markdown comparison report"""
    
    if not results:
        logger.error("No results to generate report from")
        return
    
    # Calculate statistics
    claude_results = [r for r in results if 'claude' in r and 'error' not in r['claude']]
    mistral_results = [r for r in results if 'mistral' in r and 'error' not in r['mistral']]
    
    if not claude_results or not mistral_results:
        logger.error("Insufficient successful results for comparison")
        return
    
    avg_claude_latency = sum(r['claude']['latency_ms'] for r in claude_results) / len(claude_results)
    avg_mistral_latency = sum(r['mistral']['latency_ms'] for r in mistral_results) / len(mistral_results)
    
    avg_claude_tokens = sum(r['claude']['tokens'] for r in claude_results) / len(claude_results)
    avg_mistral_tokens = sum(r['mistral']['tokens'] for r in mistral_results) / len(mistral_results)
    
    total_claude_cost = sum(r['claude']['cost_estimate'] for r in claude_results)
    
    # Generate report
    report = f"""# 🔬 Model Comparison Report: Mistral vs Claude

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Questions Tested**: {len(results)}  
**Successful Claude**: {len(claude_results)}  
**Successful Mistral**: {len(mistral_results)}

## 📊 Performance Summary

| Metric | Claude 3.5 Sonnet | Mistral 7B (Ollama) | Winner |
|--------|-------------------|---------------------|--------|
| **Avg Latency** | {avg_claude_latency:.0f}ms | {avg_mistral_latency:.0f}ms | {'🏆 Claude' if avg_claude_latency < avg_mistral_latency else '🏆 Mistral'} |
| **Avg Tokens** | {avg_claude_tokens:.0f} | {avg_mistral_tokens:.0f} | {'Claude' if avg_claude_tokens > avg_mistral_tokens else 'Mistral'} (more detailed) |
| **Total Cost** | ${total_claude_cost:.4f} | $0.00 (FREE) | 🏆 Mistral |
| **Deployment** | Cloud only | Local + Cloud | 🏆 Mistral |
| **Privacy** | Sends data to API | 100% local option | 🏆 Mistral |

## 🎯 Key Findings

### Speed
- **Claude**: ~{avg_claude_latency:.0f}ms average response time
- **Mistral**: ~{avg_mistral_latency:.0f}ms average response time
- **Difference**: {abs(avg_claude_latency - avg_mistral_latency):.0f}ms ({'Claude faster' if avg_claude_latency < avg_mistral_latency else 'Mistral faster'})

### Cost Analysis
- **Per query (Claude)**: ${total_claude_cost / len(claude_results):.6f}
- **Per 1000 queries (Claude)**: ${(total_claude_cost / len(claude_results)) * 1000:.2f}
- **Per 1M queries (Claude)**: ${(total_claude_cost / len(claude_results)) * 1_000_000:.2f}
- **Mistral**: **FREE** (local inference)

### Quality (subjective - requires manual review)
See individual comparisons below for quality assessment.

---

## 📝 Detailed Comparisons

"""
    
    # Add individual comparisons
    for i, result in enumerate(results, 1):
        question = result['question']
        
        report += f"""### {i}. {question}

"""
        
        # Expected answer (if available)
        if result.get('expected'):
            report += f"""**Expected Answer**:
> {result['expected'][:200]}{'...' if len(result['expected']) > 200 else ''}

"""
        
        # Claude response
        if 'claude' in result:
            if 'error' in result['claude']:
                report += f"""**Claude Response**: ❌ Error - {result['claude']['error']}

"""
            else:
                claude_text = result['claude']['text']
                report += f"""**Claude Response** ({result['claude']['tokens']} tokens, {result['claude']['latency_ms']}ms, ${result['claude']['cost_estimate']:.6f}):
> {claude_text[:300]}{'...' if len(claude_text) > 300 else ''}

"""
        
        # Mistral response
        if 'mistral' in result:
            if 'error' in result['mistral']:
                report += f"""**Mistral Response**: ❌ Error - {result['mistral']['error']}

"""
            else:
                mistral_text = result['mistral']['text']
                report += f"""**Mistral Response** ({result['mistral']['tokens']} tokens, {result['mistral']['latency_ms']}ms, FREE):
> {mistral_text[:300]}{'...' if len(mistral_text) > 300 else ''}

"""
        
        report += """**Quality Assessment**: [ ] Claude better  [ ] Mistral better  [ ] Equal

**Notes**: _______________________________________________

---

"""
    
    # Conclusions
    report += f"""## 🎓 Conclusions & Recommendations

### Where Claude Wins
- [ ] Response quality (detail, accuracy, nuance)
- [ ] Instruction following
- [ ] Complex reasoning
- [ ] Consistency

### Where Mistral Wins
- [x] **Cost**: $0 vs ${(total_claude_cost / len(claude_results)) * 1000:.2f} per 1000 queries
- [x] **Privacy**: 100% local option
- [x] **Permanence**: Cannot be "killed" or censored
- [x] **Customization**: Can be fine-tuned with LoRA

### Fine-Tuning Opportunities

Based on this comparison, Mistral needs improvement in:

1. **Response depth**: Claude tends to provide more detailed answers
2. **Instruction following**: Claude better at understanding intent
3. **Consistency**: Claude more reliable across different question types
4. **Language quality**: Claude more polished and natural

### Recommended Fine-Tuning Strategy

1. **Collect high-quality data** from Claude responses
2. **Format as Q&A pairs** (prompt + completion)
3. **Use LoRA** to fine-tune Mistral (preserve base model)
4. **Target specific improvements**:
   - Friendliness and warmth (Athena-style)
   - Instruction following
   - Response depth and detail
   - Consistency across topics

### Next Steps

- [ ] Manual quality review of all {len(results)} comparisons
- [ ] Score each comparison (1-5 scale)
- [ ] Identify specific failure patterns
- [ ] Collect more Claude responses for training
- [ ] Build LoRA fine-tuning dataset
- [ ] Train and validate improved model
- [ ] Re-run this comparison with fine-tuned Mistral

---

**Test Environment**:
- Hardware: Apple M3, 24GB RAM
- Ollama: v0.13.5
- Mistral: 7B base model (4.4GB)
- Claude: 3.5 Sonnet (2024-10-22)

**Total Test Cost**: ${total_claude_cost:.4f} (Claude only, Mistral free)
"""
    
    # Write report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n✅ Report generated: {output_file}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"  Total cost: ${total_claude_cost:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Mistral (Ollama) vs Claude quality"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input JSONL file with questions (from training data)'
    )
    
    parser.add_argument(
        '--questions',
        type=str,
        help='Plain text file with questions (one per line)'
    )
    
    parser.add_argument(
        '--count',
        '-c',
        type=int,
        default=10,
        help='Number of questions to test (default: 10)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='data/reports/model_comparison.md',
        help='Output markdown report (default: data/reports/model_comparison.md)'
    )
    
    parser.add_argument(
        '--claude-model',
        type=str,
        default='claude-3-5-sonnet-20241022',
        help='Claude model to use (default: claude-3-5-sonnet-20241022)'
    )
    
    parser.add_argument(
        '--mistral-model',
        type=str,
        default='mistral',
        help='Mistral model to use (default: mistral)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Max tokens per response (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Load questions
    questions = []
    
    if args.input:
        logger.info(f"📖 Loading questions from {args.input}...")
        questions = load_questions_from_jsonl(args.input, args.count)
    elif args.questions:
        logger.info(f"📖 Loading questions from {args.questions}...")
        questions = load_questions_from_txt(args.questions)
        if args.count:
            questions = questions[:args.count]
    else:
        # Default: use synthetic data
        default_file = 'data/training/synthetic_qa.jsonl'
        if Path(default_file).exists():
            logger.info(f"📖 Loading questions from {default_file} (default)...")
            questions = load_questions_from_jsonl(default_file, args.count)
        else:
            logger.error("No input file specified and default not found")
            logger.error("Usage: python compare_models.py --input data/training/synthetic_qa.jsonl")
            return
    
    if not questions:
        logger.error("No questions loaded")
        return
    
    logger.info(f"✓ Loaded {len(questions)} questions")
    
    # Run comparison
    results = compare_responses(
        questions=questions,
        claude_model=args.claude_model,
        mistral_model=args.mistral_model,
        max_tokens=args.max_tokens
    )
    
    if not results:
        logger.error("Comparison failed - no results")
        return
    
    # Generate report
    generate_markdown_report(results, args.output)
    
    logger.info("\n🎉 Comparison complete!")
    logger.info(f"   Review the report: {args.output}")
    logger.info(f"   Manually assess quality and check boxes")


if __name__ == "__main__":
    main()
