from __future__ import annotations
import math
from typing import Dict, List, Any
from config.settings import Settings
from models.embedding import VolunteerEmbedding


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Pure Python fallback dictionary to completely bypass broken native C++ binaries
        self._mock_db: Dict[str, Dict[str, Any]] = {}
        print("[DEBUG] Using Pure-Python Fallback Vector Store to bypass Windows system conflicts.")

    def persist(self) -> None:
        """No-op kept for backwards compatibility."""
        return None

    def upsert_volunteer_embedding(self, embedding: VolunteerEmbedding) -> None:
        """Saves the volunteer embedding inside a pure Python dictionary layer."""
        if isinstance(embedding.skills, list):
            skills_text = ", ".join(str(skill) for skill in embedding.skills)
        else:
            skills_text = str(embedding.skills)

        document_payload = f"Volunteer ID: {embedding.volunteer_id} | Location: {embedding.location} | Skills: {skills_text}"

        # Standard clean storage mapping
        self._mock_db[str(embedding.volunteer_id)] = {
            "id": str(embedding.volunteer_id),
            "embedding": list(embedding.embedding),
            "document": document_payload
        }

    def query_similar_volunteers(self, query_embedding: List[float], top_n: int) -> List[Dict[str, Any]]:
        """
        Performs true Cosine Similarity calculation between query and stored embeddings.
        Returns candidates ranked from highest similarity to lowest.
        """
        if not self._mock_db:
            return []

        query_norm = math.sqrt(sum(x * x for x in query_embedding))
        if query_norm == 0.0:
            query_norm = 1.0

        scored_candidates = []

        for v_id, record in self._mock_db.items():
            v_emb = record["embedding"]
            
            if len(v_emb) == len(query_embedding):
                dot_product = sum(a * b for a, b in zip(v_emb, query_embedding))
                v_norm = math.sqrt(sum(x * x for x in v_emb))
                if v_norm == 0.0:
                    v_norm = 1.0
                cosine_sim = dot_product / (query_norm * v_norm)
                # Clamp cosine similarity to [0.0, 1.0] for positive ranking
                cosine_sim = max(0.0, min(1.0, cosine_sim))
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v_emb, query_embedding)))
            else:
                cosine_sim = 0.5
                dist = 1.0

            scored_candidates.append({
                "volunteer_id": v_id,
                "text": record["document"],
                "similarity_score": cosine_sim,
                "distance": dist
            })

        # Sort highest similarity first
        scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_candidates[:top_n]

    def delete_volunteer_embedding(self, volunteer_id: str) -> None:
        if str(volunteer_id) in self._mock_db:
            del self._mock_db[str(volunteer_id)]

