"""
🧬 NOVA Neocortex - MongoDB Knowledge Base
==========================================

Memoria FLEXIBILĂ pentru concept evolution, insights, și learning progress.
Complementează PostgreSQL Cortex (rigid facts) cu MongoDB Neocortex (evolving concepts).

Autoare: Sora
Data: 8 ianuarie 2026
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from typing import List, Dict, Optional, Any
from bson import ObjectId
import os


class NovaNeocortex:
    """
    MongoDB-based Neocortex pentru NOVA training.
    
    Collections:
    - training_insights: Breakthrough-uri din conversații
    - concept_evolution: Cum concepts cresc în timp (mama: sound → abstract)
    - learning_sessions: Track BabyNova progress per session
    - architectural_decisions: De ce am ales X over Y
    """
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "nova_neocortex"
    ):
        """
        Initialize MongoDB Neocortex.
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Name of database for NOVA knowledge
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Collections
        self.insights = self.db.training_insights
        self.concepts = self.db.concept_evolution
        self.sessions = self.db.learning_sessions
        self.decisions = self.db.architectural_decisions
        
        # Create indexes for performance
        self._create_indexes()
        
        print(f"🧬 NOVA Neocortex connected to MongoDB")
        print(f"   Database: {database_name}")
        print(f"   Collections: {len(self.db.list_collection_names())}")
    
    def _create_indexes(self):
        """Create indexes for fast queries."""
        # Insights indexes
        self.insights.create_index([("category", ASCENDING)])
        self.insights.create_index([("tags", ASCENDING)])
        self.insights.create_index([("date_created", DESCENDING)])
        
        # Concepts indexes
        self.concepts.create_index([("concept_name", ASCENDING)], unique=True)
        self.concepts.create_index([("evolution_timeline.confidence", ASCENDING)])
        
        # Sessions indexes
        self.sessions.create_index([("date", DESCENDING)])
        self.sessions.create_index([("week", ASCENDING)])
        
        # Decisions indexes
        self.decisions.create_index([("date_decided", DESCENDING)])
        self.decisions.create_index([("implementation_status", ASCENDING)])
    
    # ========== TRAINING INSIGHTS ==========
    
    def add_insight(
        self,
        title: str,
        category: str,
        core_idea: str,
        tags: List[str] = None,
        examples: List[Dict] = None,
        evidence: List[str] = None,
        implementation: Dict = None,
        related_insights: List[str] = None,
        session_source: str = None,
        contributors: List[str] = None
    ) -> str:
        """
        Add a training insight to knowledge base.
        
        Args:
            title: Short title
            category: vision_learning, learning_philosophy, etc.
            core_idea: Main breakthrough insight
            tags: Keywords for search
            examples: Concrete examples
            evidence: Supporting evidence
            implementation: Implementation details
            related_insights: Links to other insights
            session_source: Session ID where insight came from
            contributors: Who contributed (Sora, Cezar, etc.)
        
        Returns:
            Inserted document ID
        """
        doc = {
            "title": title,
            "category": category,
            "tags": tags or [],
            "date_created": datetime.utcnow(),
            "contributors": contributors or ["Sora"],
            "insight": {
                "core_idea": core_idea,
                "examples": examples or [],
                "evidence": evidence or [],
                "implementation": implementation or {}
            },
            "related_insights": related_insights or [],
            "session_source": session_source
        }
        
        result = self.insights.insert_one(doc)
        print(f"✅ Insight added: {title} (ID: {result.inserted_id})")
        return str(result.inserted_id)
    
    def search_insights(
        self,
        query: str = None,
        category: str = None,
        tags: List[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search training insights.
        
        Args:
            query: Text search in title/core_idea
            category: Filter by category
            tags: Filter by tags
            limit: Max results
        
        Returns:
            List of matching insights
        """
        filter_query = {}
        
        if query:
            filter_query["$or"] = [
                {"title": {"$regex": query, "$options": "i"}},
                {"insight.core_idea": {"$regex": query, "$options": "i"}}
            ]
        
        if category:
            filter_query["category"] = category
        
        if tags:
            filter_query["tags"] = {"$in": tags}
        
        results = list(self.insights.find(filter_query).sort("date_created", DESCENDING).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for r in results:
            r["_id"] = str(r["_id"])
        
        return results
    
    # ========== CONCEPT EVOLUTION ==========
    
    def track_concept_evolution(
        self,
        concept_name: str,
        stage: int,
        week: int = None,
        month: int = None,
        confidence: float = 0.0,
        understanding_level: str = "",
        description: str = "",
        properties: Dict = None
    ) -> str:
        """
        Track how a concept evolves over time (e.g., "mama" from sound to abstract).
        
        Args:
            concept_name: Name of concept (mama, pisică, etc.)
            stage: Evolution stage number
            week: Training week
            month: Training month
            confidence: Confidence level (0.0 - 1.0)
            understanding_level: sound_recognition, entity_recognition, etc.
            description: Human-readable description
            properties: Dict of properties at this stage
        
        Returns:
            Updated or created document ID
        """
        evolution_entry = {
            "stage": stage,
            "week": week,
            "month": month,
            "confidence": confidence,
            "understanding": {
                "level": understanding_level,
                "description": description,
                "properties": properties or {}
            },
            "timestamp": datetime.utcnow()
        }
        
        # Upsert: update if exists, create if not
        result = self.concepts.update_one(
            {"concept_name": concept_name},
            {
                "$set": {
                    "concept_name": concept_name,
                    "last_updated": datetime.utcnow()
                },
                "$push": {
                    "evolution_timeline": evolution_entry
                }
            },
            upsert=True
        )
        
        if result.upserted_id:
            print(f"✅ New concept tracked: {concept_name} (Stage {stage})")
            return str(result.upserted_id)
        else:
            print(f"✅ Concept updated: {concept_name} (Stage {stage})")
            return concept_name
    
    def get_concept_evolution(self, concept_name: str) -> Optional[Dict]:
        """Get full evolution timeline for a concept."""
        result = self.concepts.find_one({"concept_name": concept_name})
        if result:
            result["_id"] = str(result["_id"])
        return result
    
    # ========== LEARNING SESSIONS ==========
    
    def log_training_session(
        self,
        session_id: str,
        week: int,
        day: int,
        curriculum: Dict,
        performance: Dict,
        new_concepts: List[str] = None,
        curiosity_questions: List[str] = None,
        model_state: Dict = None
    ) -> str:
        """
        Log a training session with BabyNova.
        
        Args:
            session_id: Unique session identifier
            week: Training week number
            day: Day within week
            curriculum: Topics covered
            performance: Accuracy metrics
            new_concepts: New concepts learned
            curiosity_questions: Questions BabyNova asked
            model_state: Model parameters at this point
        
        Returns:
            Inserted document ID
        """
        doc = {
            "session_id": session_id,
            "date": datetime.utcnow(),
            "week": week,
            "day": day,
            "curriculum": curriculum,
            "performance": performance,
            "new_concepts_learned": new_concepts or [],
            "curiosity_questions_asked": curiosity_questions or [],
            "model_state": model_state or {}
        }
        
        result = self.sessions.insert_one(doc)
        
        accuracy = performance.get("accuracy", 0)
        print(f"📊 Session logged: {session_id} (Week {week}, Accuracy: {accuracy:.2%})")
        
        return str(result.inserted_id)
    
    def get_training_progress(self, week: int = None) -> List[Dict]:
        """Get training progress, optionally filtered by week."""
        query = {"week": week} if week else {}
        results = list(self.sessions.find(query).sort("date", DESCENDING))
        
        for r in results:
            r["_id"] = str(r["_id"])
        
        return results
    
    # ========== ARCHITECTURAL DECISIONS ==========
    
    def record_decision(
        self,
        title: str,
        choice: str,
        rationale: Dict,
        alternatives: List[str] = None,
        implementation_status: str = "planned",
        validation_metrics: Dict = None,
        session_source: str = None,
        decided_by: List[str] = None
    ) -> str:
        """
        Record an architectural decision.
        
        Args:
            title: Decision title
            choice: What was chosen
            rationale: Why this choice
            alternatives: Other options considered
            implementation_status: planned/in_progress/done
            validation_metrics: How to measure success
            session_source: Session where decision was made
            decided_by: Who made decision
        
        Returns:
            Inserted document ID
        """
        doc = {
            "title": title,
            "date_decided": datetime.utcnow(),
            "decided_by": decided_by or ["Sora", "Cezar"],
            "decision": {
                "choice": choice,
                "alternatives_considered": alternatives or [],
                "rationale": rationale
            },
            "implementation_status": implementation_status,
            "validation_metrics": validation_metrics or {},
            "session_source": session_source
        }
        
        result = self.decisions.insert_one(doc)
        print(f"📋 Decision recorded: {title}")
        return str(result.inserted_id)
    
    def get_decisions(self, status: str = None) -> List[Dict]:
        """Get architectural decisions, optionally filtered by status."""
        query = {"implementation_status": status} if status else {}
        results = list(self.decisions.find(query).sort("date_decided", DESCENDING))
        
        for r in results:
            r["_id"] = str(r["_id"])
        
        return results
    
    # ========== UTILITY ==========
    
    def get_stats(self) -> Dict:
        """Get statistics about Neocortex content."""
        return {
            "insights_count": self.insights.count_documents({}),
            "concepts_tracked": self.concepts.count_documents({}),
            "sessions_logged": self.sessions.count_documents({}),
            "decisions_made": self.decisions.count_documents({}),
            "database_size_mb": self.db.command("dbStats")["dataSize"] / (1024 * 1024)
        }
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        print("🧬 Neocortex connection closed")


# ========== CLI for testing ==========

if __name__ == "__main__":
    print("🧪 Testing NOVA Neocortex...")
    
    # Initialize
    neocortex = NovaNeocortex()
    
    # Test: Add insight
    insight_id = neocortex.add_insight(
        title="Landmark-Based Vision Learning",
        category="vision_learning",
        core_idea="3-5 key geometric landmarks provide better generalization than 1,000,000 images",
        tags=["vision", "pattern_recognition", "efficiency", "landmarks"],
        examples=[
            {
                "concept": "pisică",
                "landmarks": ["4_legs", "pointed_ears", "whiskers", "fur", "tail"],
                "training_images_needed": 10
            }
        ],
        evidence=[
            "Copiii învață pisică din 5-10 exemple, nu din milioane",
            "Pattern recognition > brute force memorization"
        ],
        session_source="20260108_113000",
        contributors=["Sora", "Cezar"]
    )
    
    # Test: Track concept evolution
    neocortex.track_concept_evolution(
        concept_name="mama",
        stage=1,
        week=1,
        confidence=0.2,
        understanding_level="sound_recognition",
        description="mama = un sunet pe care îl aud des",
        properties={"is_sound": True, "frequency": "high"}
    )
    
    # Test: Log training session
    neocortex.log_training_session(
        session_id="training_20260108_001",
        week=1,
        day=1,
        curriculum={"topics": ["mama", "tata", "pisică"], "template": "Ce spune {animal}?"},
        performance={"total_questions": 100, "correct_answers": 45, "accuracy": 0.45}
    )
    
    # Test: Record decision
    neocortex.record_decision(
        title="Use Markov Chains for Week 1-2",
        choice="Markov chains",
        rationale={
            "reasons": ["Ultra-light: ~10KB memory", "Fast training: 1 sec/session"],
            "tradeoffs": {
                "pros": ["Minimal compute", "Fast iteration"],
                "cons": ["No long-term dependencies", "Must upgrade later"]
            }
        },
        alternatives=["Transformer from day 1", "RNN", "Simple rules"],
        session_source="20260107_153515"
    )
    
    # Stats
    print("\n📊 Neocortex Stats:")
    stats = neocortex.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Close
    neocortex.close()
    
    print("\n✅ NOVA Neocortex ready for knowledge storage!")
