"""
Data Augmentation for NOVA

Techniques to increase dataset diversity:
- Back-translation
- Paraphrasing
- Domain-specific augmentations
"""

import random
from typing import List, Optional, Callable
import re


class DataAugmentor:
    """
    Base class for data augmentation.
    
    Provides framework for various augmentation strategies.
    """
    
    def __init__(self, augmentation_prob: float = 0.5, seed: Optional[int] = None):
        """
        Initialize augmentor.
        
        Args:
            augmentation_prob: Probability of applying augmentation
            seed: Random seed for reproducibility
        """
        self.augmentation_prob = augmentation_prob
        
        if seed is not None:
            random.seed(seed)
    
    def augment(self, text: str) -> str:
        """
        Augment text.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        if random.random() < self.augmentation_prob:
            return self._apply_augmentation(text)
        return text
    
    def _apply_augmentation(self, text: str) -> str:
        """Override in subclasses."""
        return text
    
    def augment_batch(self, texts: List[str]) -> List[str]:
        """Augment multiple texts."""
        return [self.augment(text) for text in texts]


class SynonymReplacer(DataAugmentor):
    """
    Replace words with synonyms.
    
    Simple augmentation using predefined synonym dictionaries.
    """
    
    def __init__(
        self,
        synonyms: Optional[dict] = None,
        replacement_prob: float = 0.3,
        **kwargs
    ):
        """
        Initialize synonym replacer.
        
        Args:
            synonyms: Dictionary of word -> [synonyms]
            replacement_prob: Probability of replacing each word
        """
        super().__init__(**kwargs)
        self.replacement_prob = replacement_prob
        
        # Default physics/math synonyms
        self.synonyms = synonyms or {
            'energy': ['power', 'force'],
            'velocity': ['speed'],
            'acceleration': ['rate of change'],
            'function': ['map', 'mapping'],
            'theorem': ['proposition', 'result'],
            'proof': ['demonstration'],
            'calculate': ['compute', 'determine'],
            'equation': ['formula', 'expression'],
        }
    
    def _apply_augmentation(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            if word_lower in self.synonyms and random.random() < self.replacement_prob:
                synonym = random.choice(self.synonyms[word_lower])
                
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                
                words[i] = synonym
        
        return ' '.join(words)


class BackTranslationAugmentor(DataAugmentor):
    """
    Back-translation augmentation.
    
    Simulates translation to another language and back.
    In real implementation, would use translation APIs.
    For now, we simulate with paraphrasing.
    """
    
    def __init__(self, intermediate_language: str = 'fr', **kwargs):
        """
        Initialize back-translation augmentor.
        
        Args:
            intermediate_language: Language code for intermediate translation
        """
        super().__init__(**kwargs)
        self.intermediate_language = intermediate_language
    
    def _apply_augmentation(self, text: str) -> str:
        """
        Simulate back-translation.
        
        In production, would use translation API:
        1. Translate text -> intermediate language
        2. Translate back -> original language
        
        For now, we do simple transformations.
        """
        # Simulate translation artifacts
        
        # 1. Reorder some clauses
        sentences = text.split('. ')
        if len(sentences) > 2 and random.random() < 0.5:
            # Swap two sentences
            i = random.randint(0, len(sentences) - 2)
            sentences[i], sentences[i+1] = sentences[i+1], sentences[i]
        
        text = '. '.join(sentences)
        
        # 2. Replace some words with paraphrases
        replacements = {
            'because': 'since',
            'however': 'but',
            'therefore': 'thus',
            'large': 'big',
            'small': 'little',
            'quickly': 'rapidly',
            'slowly': 'gradually',
        }
        
        for word, replacement in replacements.items():
            if random.random() < 0.3:
                text = re.sub(rf'\b{word}\b', replacement, text, flags=re.IGNORECASE)
        
        return text


class ParaphraseAugmentor(DataAugmentor):
    """
    Paraphrase text using rule-based transformations.
    
    In production, could use T5 or GPT for paraphrasing.
    """
    
    def __init__(self, paraphrase_templates: Optional[List[tuple]] = None, **kwargs):
        """
        Initialize paraphrase augmentor.
        
        Args:
            paraphrase_templates: List of (pattern, replacement) tuples
        """
        super().__init__(**kwargs)
        
        # Default paraphrase patterns
        self.templates = paraphrase_templates or [
            (r'(\w+) is equal to (\w+)', r'\1 equals \2'),
            (r'the (\w+) of (\w+)', r"\2's \1"),
            (r'in order to (\w+)', r'to \1'),
            (r'due to the fact that', 'because'),
            (r'at this point in time', 'now'),
            (r'has the ability to', 'can'),
        ]
    
    def _apply_augmentation(self, text: str) -> str:
        """Apply paraphrase transformations."""
        for pattern, replacement in self.templates:
            if random.random() < 0.5:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


class DomainAugmentor(DataAugmentor):
    """
    Domain-specific augmentation.
    
    Applies transformations specific to physics, math, or code.
    """
    
    def __init__(self, domain: str = 'general', **kwargs):
        """
        Initialize domain augmentor.
        
        Args:
            domain: Target domain (physics, math, code, general)
        """
        super().__init__(**kwargs)
        self.domain = domain
    
    def _apply_augmentation(self, text: str) -> str:
        """Apply domain-specific augmentations."""
        if self.domain == 'physics':
            return self._augment_physics(text)
        elif self.domain == 'math':
            return self._augment_math(text)
        elif self.domain == 'code':
            return self._augment_code(text)
        else:
            return text
    
    def _augment_physics(self, text: str) -> str:
        """Augment physics text."""
        # Replace variable names
        var_replacements = {
            'v': 'u',  # velocity
            'a': 'α',  # acceleration
            'F': 'f',  # force
            'm': 'M',  # mass
        }
        
        for old, new in var_replacements.items():
            if random.random() < 0.3:
                # Only replace standalone variables
                text = re.sub(rf'\b{old}\b', new, text)
        
        # Add/remove units
        if random.random() < 0.5:
            # Convert between unit systems occasionally
            text = text.replace(' m/s', ' m·s⁻¹')
            text = text.replace(' kg*m', ' kg·m')
        
        return text
    
    def _augment_math(self, text: str) -> str:
        """Augment math text."""
        # Replace notation
        replacements = {
            '>=': '≥',
            '<=': '≤',
            '!=': '≠',
            '~=': '≈',
            'sqrt': '√',
            'infinity': '∞',
        }
        
        for old, new in replacements.items():
            if random.random() < 0.5:
                text = text.replace(old, new)
        
        # Swap equivalent expressions
        if 'x + y' in text and random.random() < 0.5:
            text = text.replace('x + y', 'y + x')
        
        return text
    
    def _augment_code(self, text: str) -> str:
        """Augment code text."""
        # Rename variables
        var_names = ['x', 'y', 'i', 'j', 'n', 'k', 'temp', 'result']
        
        for var in var_names:
            if f' {var} ' in text and random.random() < 0.3:
                new_var = var + '_val'
                text = re.sub(rf'\b{var}\b', new_var, text)
        
        # Add/remove comments
        if random.random() < 0.3:
            lines = text.split('\n')
            if lines:
                idx = random.randint(0, len(lines) - 1)
                if not lines[idx].strip().startswith('#'):
                    lines[idx] = f"# {lines[idx]}"
                text = '\n'.join(lines)
        
        return text


class MixupAugmentor(DataAugmentor):
    """
    Mixup augmentation for text.
    
    Combines two examples by mixing their representations.
    Note: This is conceptual - actual mixup happens at embedding level.
    """
    
    def __init__(self, alpha: float = 0.2, **kwargs):
        """
        Initialize mixup augmentor.
        
        Args:
            alpha: Mixup interpolation strength
        """
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def augment_pair(self, text1: str, text2: str) -> tuple:
        """
        Mix two texts.
        
        Returns:
            (mixed_text, lambda) where lambda is mixing coefficient
        """
        # Sample mixing coefficient
        lam = random.betavariate(self.alpha, self.alpha)
        
        # For text, we can't directly interpolate
        # In practice, mixup is applied at embedding level
        # Here we simulate by selecting sentences from both
        
        sentences1 = text1.split('. ')
        sentences2 = text2.split('. ')
        
        # Select sentences based on lambda
        n1 = int(len(sentences1) * lam)
        n2 = len(sentences2) - n1
        
        mixed_sentences = sentences1[:n1] + sentences2[:n2]
        random.shuffle(mixed_sentences)
        
        mixed_text = '. '.join(mixed_sentences)
        
        return mixed_text, lam


class CompositeAugmentor(DataAugmentor):
    """
    Compose multiple augmentors.
    
    Applies multiple augmentation strategies in sequence.
    """
    
    def __init__(self, augmentors: List[DataAugmentor], **kwargs):
        """
        Initialize composite augmentor.
        
        Args:
            augmentors: List of augmentors to apply
        """
        super().__init__(**kwargs)
        self.augmentors = augmentors
    
    def _apply_augmentation(self, text: str) -> str:
        """Apply all augmentors in sequence."""
        for augmentor in self.augmentors:
            text = augmentor.augment(text)
        
        return text


def create_augmentation_pipeline(
    domain: str = 'general',
    use_synonyms: bool = True,
    use_paraphrase: bool = True,
    use_back_translation: bool = False,
    augmentation_prob: float = 0.5,
) -> CompositeAugmentor:
    """
    Create augmentation pipeline for domain.
    
    Args:
        domain: Target domain
        use_synonyms: Include synonym replacement
        use_paraphrase: Include paraphrasing
        use_back_translation: Include back-translation
        augmentation_prob: Overall augmentation probability
        
    Returns:
        Composite augmentor
    """
    augmentors = []
    
    if use_synonyms:
        augmentors.append(SynonymReplacer(augmentation_prob=augmentation_prob))
    
    if use_paraphrase:
        augmentors.append(ParaphraseAugmentor(augmentation_prob=augmentation_prob))
    
    if use_back_translation:
        augmentors.append(BackTranslationAugmentor(augmentation_prob=augmentation_prob))
    
    # Always include domain-specific augmentation
    augmentors.append(DomainAugmentor(domain=domain, augmentation_prob=augmentation_prob))
    
    return CompositeAugmentor(augmentors=augmentors, augmentation_prob=1.0)
