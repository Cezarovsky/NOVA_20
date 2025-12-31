"""
🔍 NOVA Semantic Retriever

Advanced retrieval with:
- Semantic search using embeddings
- Re-ranking for improved relevance
- Hybrid search (dense + sparse)
- Filtering and post-processing
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Advanced semantic retrieval with re-ranking.
    
    Features:
    - Vector similarity search
    - Re-ranking by relevance
    - Diversity-aware retrieval
    - Metadata filtering
    """
    
    def __init__(
        self,
        vector_store,
        embeddings,
        rerank: bool = True,
        diversity_weight: float = 0.0
    ):
        """
        Initialize semantic retriever.
        
        Args:
            vector_store: ChromaVectorStore instance
            embeddings: Embedding model
            rerank: Enable re-ranking
            diversity_weight: Weight for diversity (0-1, 0=no diversity)
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.rerank = rerank
        self.diversity_weight = diversity_weight
        
        logger.info(f"✓ Semantic retriever initialized")
        logger.info(f"  Re-ranking: {rerank}")
        logger.info(f"  Diversity weight: {diversity_weight}")
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filters: Metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of result dicts with 'text', 'metadata', 'score'
        """
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Search vector store (get more for re-ranking)
        search_n = n_results * 3 if self.rerank else n_results
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=search_n,
            where=filters
        )
        
        # Convert to result format
        retrieved = []
        for i in range(len(results['ids'])):
            # Convert distance to similarity score (1 - distance for cosine)
            score = 1.0 - results['distances'][i]
            
            retrieved.append({
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i],
                'score': score
            })
        
        # Filter by score threshold
        if score_threshold is not None:
            retrieved = [r for r in retrieved if r['score'] >= score_threshold]
        
        # Re-rank if enabled
        if self.rerank and len(retrieved) > n_results:
            retrieved = self._rerank(query, retrieved, n_results)
        else:
            retrieved = retrieved[:n_results]
        
        # Apply diversity if requested
        if self.diversity_weight > 0 and len(retrieved) > 0:
            retrieved = self._diversify(retrieved, n_results)
        
        # Log results
        if len(retrieved) > 0:
            logger.info(f"✓ Retrieved {len(retrieved)} documents (score range: {retrieved[0]['score']:.3f}-{retrieved[-1]['score']:.3f})")
        else:
            logger.warning("⚠ No documents retrieved - knowledge base may be empty")
        
        return retrieved
    
    def _rerank(
        self,
        query: str,
        candidates: List[Dict],
        n_results: int
    ) -> List[Dict]:
        """
        Re-rank candidates by computing exact relevance.
        
        Uses cross-encoder style scoring with query-document pairs.
        """
        # Simple re-ranking: compute overlap with query terms
        query_terms = set(query.lower().split())
        
        for candidate in candidates:
            doc_terms = set(candidate['text'].lower().split())
            overlap = len(query_terms & doc_terms)
            term_score = overlap / max(len(query_terms), 1)
            
            # Combine with original score
            candidate['score'] = 0.7 * candidate['score'] + 0.3 * term_score
        
        # Sort by new score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:n_results]
    
    def _diversify(
        self,
        results: List[Dict],
        n_results: int
    ) -> List[Dict]:
        """
        Diversify results using MMR (Maximal Marginal Relevance).
        """
        if len(results) <= n_results:
            return results
        
        selected = [results[0]]  # Start with top result
        remaining = results[1:]
        
        while len(selected) < n_results and remaining:
            # Compute MMR scores
            mmr_scores = []
            for candidate in remaining:
                # Relevance to query (already in score)
                relevance = candidate['score']
                
                # Similarity to already selected (simple text overlap)
                max_sim = 0
                for selected_doc in selected:
                    sim = self._text_similarity(candidate['text'], selected_doc['text'])
                    max_sim = max(max_sim, sim)
                
                # MMR = λ * relevance - (1-λ) * max_similarity
                mmr = (
                    (1 - self.diversity_weight) * relevance -
                    self.diversity_weight * max_sim
                )
                mmr_scores.append((mmr, candidate))
            
            # Select best MMR score
            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            best = mmr_scores[0][1]
            selected.append(best)
            remaining.remove(best)
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity between texts."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0
    
    def retrieve_with_context(
        self,
        query: str,
        n_results: int = 5,
        context_window: int = 1
    ) -> List[Dict]:
        """
        Retrieve documents with surrounding context chunks.
        
        Args:
            query: Search query
            n_results: Number of results
            context_window: Number of adjacent chunks to include
            
        Returns:
            List of results with context
        """
        results = self.retrieve(query, n_results)
        
        # Add context from adjacent chunks
        for result in results:
            metadata = result['metadata']
            
            if 'chunk_index' in metadata:
                chunk_idx = metadata['chunk_index']
                source = metadata.get('source', '')
                
                # Get adjacent chunks
                context_chunks = []
                for offset in range(-context_window, context_window + 1):
                    if offset == 0:
                        continue
                    
                    target_idx = chunk_idx + offset
                    # Query for adjacent chunk
                    adjacent = self.vector_store.get(
                        where={
                            'source': source,
                            'chunk_index': target_idx
                        },
                        limit=1
                    )
                    
                    if adjacent['documents']:
                        context_chunks.append({
                            'offset': offset,
                            'text': adjacent['documents'][0]
                        })
                
                result['context'] = context_chunks
        
        return results


if __name__ == "__main__":
    print("🔍 Semantic Retriever Demo")
    print("Note: Run after setting up vector store and embeddings")
    print("✓ Retriever module ready")
