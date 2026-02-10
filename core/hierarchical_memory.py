"""
Advanced Memory System with Hierarchical Storage

Features:
- Episodic memory (specific events/conversations)
- Semantic memory (general knowledge/facts)
- Procedural memory (learned behaviors/patterns)
- Importance-based retention
- LLM-powered summarization
- Temporal decay with importance weighting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict


@dataclass
class MemoryEntry:
    """Base memory entry"""
    content: str
    timestamp: datetime
    importance: float  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)
    
    def decay_importance(self, current_time: datetime, half_life_hours: float = 24.0):
        """Apply temporal decay to importance"""
        hours_elapsed = (current_time - self.timestamp).total_seconds() / 3600
        decay_factor = 0.5 ** (hours_elapsed / half_life_hours)
        # Importance decays but access count provides resistance
        access_boost = min(0.3, self.access_count * 0.05)
        self.importance = max(0.1, self.importance * decay_factor + access_boost)


@dataclass
class EpisodicMemory(MemoryEntry):
    """Memory of specific events/conversations"""
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    emotion: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class SemanticMemory(MemoryEntry):
    """General knowledge/facts"""
    category: str = "general"
    confidence: float = 1.0
    source: Optional[str] = None


@dataclass
class ProceduralMemory(MemoryEntry):
    """Learned behaviors/patterns"""
    trigger: str = ""
    action: str = ""
    success_rate: float = 0.0
    usage_count: int = 0


class HierarchicalMemorySystem:
    """
    Advanced memory system with multiple memory types
    """
    
    def __init__(self, 
                 max_episodic: int = 100,
                 max_semantic: int = 500,
                 max_procedural: int = 50,
                 importance_threshold: float = 0.3):
        self.episodic_memories: List[EpisodicMemory] = []
        self.semantic_memories: List[SemanticMemory] = []
        self.procedural_memories: List[ProceduralMemory] = []
        
        self.max_episodic = max_episodic
        self.max_semantic = max_semantic
        self.max_procedural = max_procedural
        self.importance_threshold = importance_threshold
        
        # Entity tracking
        self.entities: Dict[str, Dict] = {}  # entity_name -> {type, mentions, relationships}
        
    def add_episodic(self, content: str, importance: float = 0.5, 
                     participants: List[str] = None, **kwargs) -> EpisodicMemory:
        """Add episodic memory"""
        memory = EpisodicMemory(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            participants=participants or [],
            **kwargs
        )
        
        self.episodic_memories.append(memory)
        
        # Extract and track entities
        for participant in memory.participants:
            self._track_entity(participant, "person")
        
        # Consolidate if over limit
        if len(self.episodic_memories) > self.max_episodic:
            self._consolidate_episodic()
        
        return memory
    
    def add_semantic(self, content: str, importance: float = 0.7,
                     category: str = "general", **kwargs) -> SemanticMemory:
        """Add semantic memory"""
        memory = SemanticMemory(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            category=category,
            **kwargs
        )
        
        self.semantic_memories.append(memory)
        
        # Consolidate if over limit
        if len(self.semantic_memories) > self.max_semantic:
            self._consolidate_semantic()
        
        return memory
    
    def add_procedural(self, trigger: str, action: str, 
                       importance: float = 0.6) -> ProceduralMemory:
        """Add procedural memory"""
        # Check if pattern already exists
        for mem in self.procedural_memories:
            if mem.trigger == trigger and mem.action == action:
                mem.usage_count += 1
                mem.importance = min(1.0, mem.importance + 0.05)
                return mem
        
        memory = ProceduralMemory(
            content=f"{trigger} -> {action}",
            timestamp=datetime.now(),
            importance=importance,
            trigger=trigger,
            action=action
        )
        
        self.procedural_memories.append(memory)
        
        if len(self.procedural_memories) > self.max_procedural:
            self._consolidate_procedural()
        
        return memory
    
    def _track_entity(self, entity_name: str, entity_type: str = "unknown"):
        """Track entity mentions"""
        if entity_name not in self.entities:
            self.entities[entity_name] = {
                "type": entity_type,
                "mentions": 0,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "relationships": []
            }
        
        self.entities[entity_name]["mentions"] += 1
        self.entities[entity_name]["last_seen"] = datetime.now()
    
    def _consolidate_episodic(self):
        """Consolidate episodic memories by removing low-importance ones"""
        current_time = datetime.now()
        
        # Apply decay
        for mem in self.episodic_memories:
            mem.decay_importance(current_time)
        
        # Sort by importance
        self.episodic_memories.sort(key=lambda m: m.importance, reverse=True)
        
        # Keep top memories
        removed = self.episodic_memories[self.max_episodic:]
        self.episodic_memories = self.episodic_memories[:self.max_episodic]
        
        print(f"Consolidated episodic: removed {len(removed)} low-importance memories")
    
    def _consolidate_semantic(self):
        """Consolidate semantic memories"""
        current_time = datetime.now()
        
        for mem in self.semantic_memories:
            mem.decay_importance(current_time, half_life_hours=168)  # 1 week half-life
        
        self.semantic_memories.sort(key=lambda m: m.importance, reverse=True)
        self.semantic_memories = self.semantic_memories[:self.max_semantic]
    
    def _consolidate_procedural(self):
        """Consolidate procedural memories"""
        # Sort by usage and success rate
        self.procedural_memories.sort(
            key=lambda m: (m.usage_count * m.success_rate, m.importance),
            reverse=True
        )
        self.procedural_memories = self.procedural_memories[:self.max_procedural]
    
    def retrieve_episodic(self, query: str = None, top_k: int = 5, 
                          min_importance: float = None) -> List[EpisodicMemory]:
        """Retrieve episodic memories"""
        if min_importance is None:
            min_importance = self.importance_threshold
        
        # Filter by importance
        candidates = [m for m in self.episodic_memories if m.importance >= min_importance]
        
        # Sort by recency and importance
        candidates.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        return candidates[:top_k]
    
    def retrieve_semantic(self, category: str = None, top_k: int = 10) -> List[SemanticMemory]:
        """Retrieve semantic memories"""
        if category:
            candidates = [m for m in self.semantic_memories if m.category == category]
        else:
            candidates = self.semantic_memories
        
        candidates.sort(key=lambda m: m.importance, reverse=True)
        return candidates[:top_k]
    
    def retrieve_procedural(self, trigger: str = None) -> List[ProceduralMemory]:
        """Retrieve procedural memories"""
        if trigger:
            # Find matching triggers
            matches = [m for m in self.procedural_memories if trigger.lower() in m.trigger.lower()]
            matches.sort(key=lambda m: m.success_rate, reverse=True)
            return matches
        
        return sorted(self.procedural_memories, key=lambda m: m.usage_count, reverse=True)
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict]:
        """Get information about a tracked entity"""
        return self.entities.get(entity_name)
    
    def get_memory_summary(self) -> Dict:
        """Get summary of memory system state"""
        return {
            "episodic_count": len(self.episodic_memories),
            "semantic_count": len(self.semantic_memories),
            "procedural_count": len(self.procedural_memories),
            "entities_tracked": len(self.entities),
            "avg_episodic_importance": np.mean([m.importance for m in self.episodic_memories]) if self.episodic_memories else 0.0,
            "avg_semantic_importance": np.mean([m.importance for m in self.semantic_memories]) if self.semantic_memories else 0.0,
        }
    
    def save_state(self, filepath: str):
        """Save memory state to file"""
        state = {
            "episodic": [self._memory_to_dict(m) for m in self.episodic_memories],
            "semantic": [self._memory_to_dict(m) for m in self.semantic_memories],
            "procedural": [self._memory_to_dict(m) for m in self.procedural_memories],
            "entities": self.entities
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str):
        """Load memory state from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.episodic_memories = [self._dict_to_episodic(d) for d in state.get("episodic", [])]
        self.semantic_memories = [self._dict_to_semantic(d) for d in state.get("semantic", [])]
        self.procedural_memories = [self._dict_to_procedural(d) for d in state.get("procedural", [])]
        self.entities = state.get("entities", {})
    
    def _memory_to_dict(self, memory: MemoryEntry) -> Dict:
        """Convert memory to dictionary"""
        d = {
            "content": memory.content,
            "timestamp": memory.timestamp.isoformat(),
            "importance": memory.importance,
            "access_count": memory.access_count,
            "metadata": memory.metadata
        }
        
        if isinstance(memory, EpisodicMemory):
            d["type"] = "episodic"
            d["participants"] = memory.participants
            d["location"] = memory.location
            d["emotion"] = memory.emotion
            d["summary"] = memory.summary
        elif isinstance(memory, SemanticMemory):
            d["type"] = "semantic"
            d["category"] = memory.category
            d["confidence"] = memory.confidence
        elif isinstance(memory, ProceduralMemory):
            d["type"] = "procedural"
            d["trigger"] = memory.trigger
            d["action"] = memory.action
            d["success_rate"] = memory.success_rate
            d["usage_count"] = memory.usage_count
        
        return d
    
    def _dict_to_episodic(self, d: Dict) -> EpisodicMemory:
        """Convert dictionary to episodic memory"""
        return EpisodicMemory(
            content=d["content"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            importance=d["importance"],
            access_count=d.get("access_count", 0),
            participants=d.get("participants", []),
            location=d.get("location"),
            emotion=d.get("emotion"),
            summary=d.get("summary"),
            metadata=d.get("metadata", {})
        )
    
    def _dict_to_semantic(self, d: Dict) -> SemanticMemory:
        """Convert dictionary to semantic memory"""
        return SemanticMemory(
            content=d["content"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            importance=d["importance"],
            access_count=d.get("access_count", 0),
            category=d.get("category", "general"),
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {})
        )
    
    def _dict_to_procedural(self, d: Dict) -> ProceduralMemory:
        """Convert dictionary to procedural memory"""
        return ProceduralMemory(
            content=d["content"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            importance=d["importance"],
            access_count=d.get("access_count", 0),
            trigger=d.get("trigger", ""),
            action=d.get("action", ""),
            success_rate=d.get("success_rate", 0.0),
            usage_count=d.get("usage_count", 0),
            metadata=d.get("metadata", {})
        )
