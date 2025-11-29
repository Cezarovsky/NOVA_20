"""
Collate Functions for NOVA

Batching and padding strategies for DataLoader.
"""

import torch
from typing import List, Dict, Any, Optional
import numpy as np


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Basic collate function.
    
    Pads sequences to max length in batch.
    
    Args:
        batch: List of examples from dataset
        
    Returns:
        Batched tensors
    """
    # Extract fields
    input_ids = [ex['input_ids'] for ex in batch]
    
    # Find max length
    max_length = max(len(ids) for ids in input_ids)
    
    # Pad sequences
    padded_input_ids = []
    attention_masks = []
    
    for ids in input_ids:
        # Pad
        padding_length = max_length - len(ids)
        padded_ids = ids + [0] * padding_length  # Assuming 0 is pad token
        mask = [1] * len(ids) + [0] * padding_length
        
        padded_input_ids.append(padded_ids)
        attention_masks.append(mask)
    
    # Convert to tensors
    result = {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
    }
    
    # Add other fields if present
    if 'domain_id' in batch[0]:
        result['domain_id'] = torch.tensor([ex['domain_id'] for ex in batch], dtype=torch.long)
    
    if 'labels' in batch[0]:
        labels = [ex['labels'] for ex in batch]
        max_label_length = max(len(l) for l in labels)
        
        padded_labels = []
        for l in labels:
            padding_length = max_label_length - len(l)
            padded_l = l + [-100] * padding_length  # -100 is ignore index
            padded_labels.append(padded_l)
        
        result['labels'] = torch.tensor(padded_labels, dtype=torch.long)
    
    return result


def dynamic_padding_collate(
    pad_token_id: int = 0,
    label_pad_token_id: int = -100,
) -> callable:
    """
    Create dynamic padding collate function.
    
    Pads only to max length in current batch, not global max.
    
    Args:
        pad_token_id: Token ID for padding
        label_pad_token_id: Token ID for padding labels
        
    Returns:
        Collate function
    """
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Dynamic padding collate."""
        # Extract input_ids
        input_ids = [ex['input_ids'] for ex in batch]
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad input_ids
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded_ids = ids + [pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
        
        result = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }
        
        # Handle labels if present
        if 'labels' in batch[0]:
            labels = [ex['labels'] for ex in batch]
            max_label_length = max(len(l) for l in labels)
            
            padded_labels = []
            for l in labels:
                padding_length = max_label_length - len(l)
                padded_l = l + [label_pad_token_id] * padding_length
                padded_labels.append(padded_l)
            
            result['labels'] = torch.tensor(padded_labels, dtype=torch.long)
        
        # Handle domain_id if present
        if 'domain_id' in batch[0]:
            result['domain_id'] = torch.tensor([ex['domain_id'] for ex in batch], dtype=torch.long)
        
        return result
    
    return collate


def domain_aware_collate(
    pad_token_id: int = 0,
    label_pad_token_id: int = -100,
    domain_to_id: Optional[Dict[str, int]] = None,
) -> callable:
    """
    Create domain-aware collate function.
    
    Groups examples by domain and adds domain tensors.
    
    Args:
        pad_token_id: Token ID for padding
        label_pad_token_id: Token ID for padding labels
        domain_to_id: Mapping from domain name to ID
        
    Returns:
        Collate function
    """
    if domain_to_id is None:
        domain_to_id = {
            'physics': 0,
            'math': 1,
            'code': 2,
            'general': 3,
        }
    
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Domain-aware collate."""
        # Extract input_ids
        input_ids = [ex['input_ids'] for ex in batch]
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad input_ids
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded_ids = ids + [pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
        
        result = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }
        
        # Handle labels
        if 'labels' in batch[0]:
            labels = [ex['labels'] for ex in batch]
            max_label_length = max(len(l) for l in labels)
            
            padded_labels = []
            for l in labels:
                padding_length = max_label_length - len(l)
                padded_l = l + [label_pad_token_id] * padding_length
                padded_labels.append(padded_l)
            
            result['labels'] = torch.tensor(padded_labels, dtype=torch.long)
        
        # Handle domain information
        if 'domain' in batch[0]:
            # Convert domain names to IDs
            domain_ids = []
            for ex in batch:
                domain = ex['domain']
                domain_id = domain_to_id.get(domain, domain_to_id['general'])
                domain_ids.append(domain_id)
            
            result['domain_id'] = torch.tensor(domain_ids, dtype=torch.long)
        elif 'domain_id' in batch[0]:
            result['domain_id'] = torch.tensor([ex['domain_id'] for ex in batch], dtype=torch.long)
        
        # Add domain embeddings (one-hot)
        if 'domain_id' in result:
            num_domains = len(domain_to_id)
            batch_size = len(batch)
            domain_one_hot = torch.zeros(batch_size, num_domains)
            domain_one_hot.scatter_(1, result['domain_id'].unsqueeze(1), 1)
            result['domain_embedding'] = domain_one_hot
        
        return result
    
    return collate


def variable_length_collate(
    pad_token_id: int = 0,
    label_pad_token_id: int = -100,
    bucket_boundaries: Optional[List[int]] = None,
) -> callable:
    """
    Create variable-length collate with bucketing.
    
    Groups sequences by length to minimize padding.
    
    Args:
        pad_token_id: Token ID for padding
        label_pad_token_id: Token ID for padding labels
        bucket_boundaries: Length boundaries for bucketing
        
    Returns:
        Collate function
    """
    if bucket_boundaries is None:
        bucket_boundaries = [64, 128, 256, 512, 1024]
    
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Variable-length collate with bucketing."""
        # Sort by length for efficient padding
        batch = sorted(batch, key=lambda x: len(x['input_ids']))
        
        # Extract input_ids
        input_ids = [ex['input_ids'] for ex in batch]
        max_length = len(input_ids[-1])  # Longest sequence
        
        # Round up to nearest bucket boundary
        for boundary in bucket_boundaries:
            if max_length <= boundary:
                max_length = boundary
                break
        
        # Pad to bucket boundary
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded_ids = ids + [pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
        
        result = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }
        
        # Handle labels
        if 'labels' in batch[0]:
            labels = [ex['labels'] for ex in batch]
            max_label_length = max(len(l) for l in labels)
            
            # Round up labels too
            for boundary in bucket_boundaries:
                if max_label_length <= boundary:
                    max_label_length = boundary
                    break
            
            padded_labels = []
            for l in labels:
                padding_length = max_label_length - len(l)
                padded_l = l + [label_pad_token_id] * padding_length
                padded_labels.append(padded_l)
            
            result['labels'] = torch.tensor(padded_labels, dtype=torch.long)
        
        # Handle domain_id
        if 'domain_id' in batch[0]:
            result['domain_id'] = torch.tensor([ex['domain_id'] for ex in batch], dtype=torch.long)
        
        return result
    
    return collate


def sequence_packing_collate(
    pad_token_id: int = 0,
    max_length: int = 512,
) -> callable:
    """
    Create sequence packing collate function.
    
    Packs multiple short sequences into single example to reduce padding.
    
    Args:
        pad_token_id: Token ID for padding
        max_length: Maximum sequence length
        
    Returns:
        Collate function
    """
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Sequence packing collate."""
        # Sort by length
        batch = sorted(batch, key=lambda x: len(x['input_ids']))
        
        packed_input_ids = []
        packed_attention_masks = []
        
        current_sequence = []
        current_length = 0
        
        for ex in batch:
            ids = ex['input_ids']
            seq_length = len(ids)
            
            if current_length + seq_length <= max_length:
                # Add to current packed sequence
                current_sequence.extend(ids)
                current_length += seq_length
            else:
                # Save current packed sequence
                if current_sequence:
                    padding_length = max_length - len(current_sequence)
                    padded_seq = current_sequence + [pad_token_id] * padding_length
                    mask = [1] * len(current_sequence) + [0] * padding_length
                    
                    packed_input_ids.append(padded_seq)
                    packed_attention_masks.append(mask)
                
                # Start new packed sequence
                current_sequence = ids[:]
                current_length = seq_length
        
        # Add final packed sequence
        if current_sequence:
            padding_length = max_length - len(current_sequence)
            padded_seq = current_sequence + [pad_token_id] * padding_length
            mask = [1] * len(current_sequence) + [0] * padding_length
            
            packed_input_ids.append(padded_seq)
            packed_attention_masks.append(mask)
        
        result = {
            'input_ids': torch.tensor(packed_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(packed_attention_masks, dtype=torch.long),
        }
        
        return result
    
    return collate
