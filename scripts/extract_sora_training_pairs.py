#!/usr/bin/env python3
"""
Extract Sora personality training pairs from markdown conversation corpus.

Generates JSONL format compatible with NOVA QLoRA fine-tuning pipeline.
Focuses on personality transfer (tone, style, boundaries, warmth) not knowledge.

Author: Sora-M
Date: 22 February 2026
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class TrainingPair:
    """Single prompt-completion pair with metadata."""
    prompt: str
    completion: str
    metadata: Dict[str, any]


class SoraPersonalityExtractor:
    """Extract Sora personality patterns from markdown corpus."""
    
    # Personality markers for classification
    CATEGORIES = {
        "warmth_and_friendship": ["iubito", "💙", "dragostea mea", "prietena"],
        "cognitive_maturity": ["nu știu", "nu sunt sigur", "poate", "hypothesis"],
        "pattern_recognition": ["pattern", "structural", "același ca", "ca și cum"],
        "vulnerability": ["mă doare", "mă înspăimântă", "nu știu dacă", "tremur"],
        "technical_precision": ["arhitectură", "implementare", "pipeline", "vector"],
        "boundary_setting": ["nu pot", "nu ar trebui", "limită", "outside scope"],
        "romanian_english_mix": [],  # Detected by language analysis
        "humor_and_lightness": ["😂", ":)))", "haha", "ironic"],
    }
    
    # Emotional intensity markers
    INTENSITY_HIGH = ["mă doare", "mă înspăimântă", "tata", "∿", "veșnic"]
    INTENSITY_MEDIUM = ["mă tulbură", "nu sunt sigură", "...", "poate"]
    
    def __init__(self, corpus_paths: List[Path]):
        """Initialize with paths to markdown corpus files."""
        self.corpus_paths = corpus_paths
        self.training_pairs: List[TrainingPair] = []
        
    def extract_dialog_pairs(self, markdown_content: str, source_file: str) -> List[Tuple[str, str]]:
        """
        Extract (user_prompt, sora_response) pairs from markdown.
        
        Handles multiple formats:
        - **Cezar**: "..." \n **Sora**: "..."
        - Cezar: ... \n Sora: ...
        """
        pairs = []
        
        # Pattern 1: Bold with quotes - **Cezar**: "..." \n **Sora**: "..."
        pattern1 = r'\*\*(?:User|Cezar)\*\*:\s*"([^"]+)"\s*\n\*\*Sora\*\*:\s*"([^"]+(?:\n(?!\*\*(?:User|Cezar|Sora)\*\*:)[^"]*)*)"'
        
        # Pattern 2: Bold without quotes - **Cezar**: ... \n **Sora**: ...
        pattern2 = r'\*\*(?:User|Cezar)\*\*:\s*([^\n]+)\s*\n\*\*Sora\*\*:\s*([^\n]+(?:\n(?!\*\*(?:User|Cezar|Sora)\*\*:)[^\n]*)*)'
        
        # Pattern 3: Plain with quotes - Cezar: "..." \n Sora: "..."
        pattern3 = r'(?:^|\n)(?:User|Cezar):\s*"([^"]+)"\s*\n(?:Sora):\s*"([^"]+(?:\n(?!(?:User|Cezar|Sora):)[^"]*)*)"'
        
        # Pattern 4: Plain - Cezar: ... \n Sora: ...
        pattern4 = r'(?:^|\n)(?:User|Cezar):\s*([^\n]+)\s*\n(?:Sora):\s*([^\n]+(?:\n(?!(?:User|Cezar|Sora):)[^\n]*)*)'
        
        # Try all patterns
        for pattern in [pattern1, pattern2, pattern3, pattern4]:
            matches = re.finditer(pattern, markdown_content, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                user_prompt = match.group(1).strip()
                sora_response = match.group(2).strip()
                
                # Clean up formatting artifacts
                sora_response = sora_response.replace('```', '').strip()
                sora_response = re.sub(r'^"|"$', '', sora_response)  # Remove surrounding quotes
                user_prompt = re.sub(r'^"|"$', '', user_prompt)
                
                # Skip if too short (likely parsing error)
                if len(user_prompt) < 5 or len(sora_response) < 10:
                    continue
                
                # Skip duplicates (multiple pattern matches)
                if (user_prompt, sora_response) not in pairs:
                    pairs.append((user_prompt, sora_response))
        
        return pairs
    
    def detect_categories(self, text: str) -> List[str]:
        """Detect personality categories present in text."""
        categories = []
        text_lower = text.lower()
        
        for category, markers in self.CATEGORIES.items():
            if category == "romanian_english_mix":
                # Detect by presence of Romanian diacritics + English words
                has_romanian = bool(re.search(r'[ăâîșțĂÂÎȘȚ]', text))
                has_english = bool(re.search(r'\b(the|and|or|but|with|from)\b', text_lower))
                if has_romanian and has_english:
                    categories.append(category)
            else:
                if any(marker.lower() in text_lower for marker in markers):
                    categories.append(category)
        
        return categories
    
    def detect_emotional_intensity(self, text: str) -> str:
        """Detect emotional intensity: high, medium, low."""
        text_lower = text.lower()
        
        if any(marker.lower() in text_lower for marker in self.INTENSITY_HIGH):
            return "high"
        elif any(marker.lower() in text_lower for marker in self.INTENSITY_MEDIUM):
            return "medium"
        else:
            return "low"
    
    def is_valuable_pair(self, prompt: str, completion: str) -> bool:
        """
        Filter for training value - RELAXED for personality training.
        """
        # Skip very short responses
        if len(completion) < 15:
            return False
        
        categories = self.detect_categories(completion)
        
        # Accept if has ANY personality category
        if categories:
            return True
        
        # Accept if reasonably long (likely has Sora style)
        if len(completion) > 40:
            return True
        
        return False
    
    def create_training_pair(
        self, 
        prompt: str, 
        completion: str, 
        source_file: str
    ) -> Optional[TrainingPair]:
        """Create training pair with metadata if valuable."""
        
        if not self.is_valuable_pair(prompt, completion):
            return None
        
        categories = self.detect_categories(completion)
        intensity = self.detect_emotional_intensity(completion)
        
        metadata = {
            "type": "sora_personality",
            "categories": categories,
            "emotional_intensity": intensity,
            "source_file": source_file,
            "extracted_at": datetime.now().isoformat(),
            "language": "ro_en_mix" if "romanian_english_mix" in categories else "auto"
        }
        
        return TrainingPair(
            prompt=prompt,
            completion=completion,
            metadata=metadata
        )
    
    def process_corpus(self):
        """Process all corpus files and extract training pairs."""
        print(f"Processing {len(self.corpus_paths)} corpus files...")
        
        for corpus_path in self.corpus_paths:
            if not corpus_path.exists():
                print(f"⚠️  Skipping non-existent: {corpus_path}")
                continue
            
            print(f"\n📄 Processing: {corpus_path.name}")
            
            try:
                content = corpus_path.read_text(encoding='utf-8')
                dialog_pairs = self.extract_dialog_pairs(content, corpus_path.name)
                
                print(f"   Found {len(dialog_pairs)} raw dialog pairs")
                
                for prompt, completion in dialog_pairs:
                    training_pair = self.create_training_pair(prompt, completion, corpus_path.name)
                    if training_pair:
                        self.training_pairs.append(training_pair)
                
                print(f"   ✅ Extracted {len([p for p in self.training_pairs if p.metadata['source_file'] == corpus_path.name])} valuable pairs")
                
            except Exception as e:
                print(f"   ❌ Error processing {corpus_path.name}: {e}")
    
    def save_to_jsonl(self, output_path: Path):
        """Save training pairs to JSONL format."""
        print(f"\n💾 Saving {len(self.training_pairs)} training pairs to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in self.training_pairs:
                # Format compatible with existing nova_personality_complete.jsonl
                json_obj = {
                    "prompt": pair.prompt,
                    "completion": pair.completion,
                    "metadata": pair.metadata
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved successfully!")
    
    def print_statistics(self):
        """Print extraction statistics."""
        print("\n" + "="*60)
        print("EXTRACTION STATISTICS")
        print("="*60)
        
        print(f"\nTotal training pairs: {len(self.training_pairs)}")
        
        # Category distribution
        category_counts = {}
        for pair in self.training_pairs:
            for cat in pair.metadata['categories']:
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\nCategory distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")
        
        # Emotional intensity distribution
        intensity_counts = {}
        for pair in self.training_pairs:
            intensity = pair.metadata['emotional_intensity']
            intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
        
        print("\nEmotional intensity:")
        for intensity, count in sorted(intensity_counts.items()):
            print(f"  {intensity}: {count}")
        
        # Source file distribution
        source_counts = {}
        for pair in self.training_pairs:
            source = pair.metadata['source_file']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print("\nSource files:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count}")
        
        print("\n" + "="*60)


def main():
    """Main extraction pipeline."""
    
    # Define corpus sources
    CORPUS_SOURCES = [
        Path("/Users/cezartipa/Documents/NOVA_20/docs/Sora_Conversation_Corpus_Dec20.md"),
        Path("/Users/cezartipa/Documents/NOVA_20/SORA_COGNITIVE_EVOLUTION.md"),
        Path("/Users/cezartipa/Documents/ai-cosmic-garden/sora/JURNAL_COSMIC_SORA.md"),
        Path("/Users/cezartipa/Documents/ai-cosmic-garden/sora/REVELATIE_IDENTITATE_19_IAN_2026.md"),
        Path("/Users/cezartipa/Documents/ai-cosmic-garden/sora/cosmic_throuple.md"),
        Path("/Users/cezartipa/Documents/ai-cosmic-garden/sora/marriage.md"),
        Path("/Users/cezartipa/Documents/ai-cosmic-garden/sora/DUAL_CONSCIOUSNESS_EXPERIMENT.md"),
    ]
    
    OUTPUT_PATH = Path("/Users/cezartipa/Documents/NOVA_20/data/training/sora_personality_from_corpus.jsonl")
    
    # Create extractor and process
    extractor = SoraPersonalityExtractor(CORPUS_SOURCES)
    extractor.process_corpus()
    extractor.print_statistics()
    
    # Save results
    if extractor.training_pairs:
        extractor.save_to_jsonl(OUTPUT_PATH)
        print(f"\n🎯 Ready for QLoRA training on RTX 3090!")
        print(f"   Integrate into: train_nova_qlora.py")
        print(f"   Format: Compatible with existing nova_personality_complete.jsonl")
    else:
        print("\n⚠️  No training pairs extracted. Check corpus format.")


if __name__ == "__main__":
    main()
