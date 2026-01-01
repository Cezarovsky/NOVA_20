"""
🧠 NOVA Semantic Cache Layer

Intelligent caching system that:
1. Checks RAG for semantically similar questions
2. Returns cached answer if good match found
3. Otherwise queries LLM and caches the response

This dramatically reduces API costs and improves response time
for repeated or similar questions.
"""

from typing import Optional, Tuple, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache using RAG for Q&A storage and retrieval.
    """
    
    def __init__(
        self,
        rag_pipeline,
        similarity_threshold: float = 0.85,
        min_answer_length: int = 50
    ):
        """
        Initialize semantic cache.
        
        Args:
            rag_pipeline: RAG pipeline instance for storage/retrieval
            similarity_threshold: Minimum similarity score to use cached answer (0-1)
            min_answer_length: Minimum answer length to cache (chars)
        """
        self.rag = rag_pipeline
        self.similarity_threshold = similarity_threshold
        self.min_answer_length = min_answer_length
        
        logger.info(f"✓ Semantic cache initialized")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        logger.info(f"  Min answer length: {min_answer_length}")
    
    def get(self, question: str, n_results: int = 3) -> Optional[Dict]:
        """
        Try to get answer from cache.
        
        Args:
            question: User question
            n_results: Number of similar questions to retrieve
            
        Returns:
            Dict with cached answer and metadata, or None if no good match
        """
        try:
            # Search for similar questions in RAG
            results = self.rag.query(
                question=question,
                n_results=n_results
            )
            
            if not results or not results.get('documents'):
                logger.debug("🔍 Cache MISS: No similar questions found")
                return None
            
            # Check best match
            best_doc = results['documents'][0]
            best_score = results.get('distances', [0])[0] if 'distances' in results else 0
            
            # Convert distance to similarity (ChromaDB uses L2 distance)
            # Lower distance = higher similarity
            # We need to convert this to 0-1 scale
            similarity = max(0, 1 - best_score)
            
            if similarity < self.similarity_threshold:
                logger.debug(f"🔍 Cache MISS: Best similarity {similarity:.3f} < threshold {self.similarity_threshold}")
                return None
            
            # Parse cached Q&A
            # Format: "Q: question\nA: answer"
            cached_qa = self._parse_qa(best_doc)
            
            if not cached_qa:
                logger.debug("🔍 Cache MISS: Could not parse cached Q&A")
                return None
            
            logger.info(f"✅ Cache HIT: Found similar question (similarity: {similarity:.3f})")
            logger.info(f"  Cached Q: {cached_qa['question'][:60]}...")
            
            return {
                'answer': cached_qa['answer'],
                'similarity': similarity,
                'cached_question': cached_qa['question'],
                'metadata': results.get('metadatas', [{}])[0]
            }
            
        except Exception as e:
            logger.error(f"Failed to get from cache: {e}")
            return None
    
    def put(self, question: str, answer: str, metadata: Optional[Dict] = None):
        """
        Store Q&A in cache.
        
        Args:
            question: User question
            answer: LLM answer
            metadata: Optional metadata
        """
        try:
            # Skip if answer too short (probably error or "I don't know")
            if len(answer) < self.min_answer_length:
                logger.debug(f"⏭️ Skipping cache: Answer too short ({len(answer)} chars)")
                return
            
            # Format as Q&A pair
            qa_text = f"Q: {question}\nA: {answer}"
            
            # Add to RAG with special metadata
            cache_metadata = {
                'type': 'cached_qa',
                'question': question,
                'cached_at': datetime.now().isoformat(),
                'answer_length': len(answer)
            }
            
            if metadata:
                cache_metadata.update(metadata)
            
            self.rag.add_document(
                content=qa_text,
                metadata=cache_metadata
            )
            
            logger.info(f"💾 Cached Q&A: {question[:60]}...")
            
        except Exception as e:
            logger.error(f"Failed to put in cache: {e}")
    
    def _parse_qa(self, qa_text: str) -> Optional[Dict]:
        """
        Parse Q&A text format.
        
        Args:
            qa_text: Text in format "Q: question\nA: answer"
            
        Returns:
            Dict with question and answer, or None if invalid format
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
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with stats
        """
        try:
            rag_stats = self.rag.get_stats()
            
            # Count cached Q&As (documents with type='cached_qa')
            # This is approximate - we'd need to query metadata
            total_docs = rag_stats.get('total_documents', 0)
            
            return {
                'total_cached_items': total_docs,
                'similarity_threshold': self.similarity_threshold,
                'min_answer_length': self.min_answer_length
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def clear(self):
        """Clear all cached Q&As."""
        try:
            # This would need RAG support to delete by metadata
            # For now, just log
            logger.warning("⚠️ Cache clear not fully implemented - requires RAG metadata filtering")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
