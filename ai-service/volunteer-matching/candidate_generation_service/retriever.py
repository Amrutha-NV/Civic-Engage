from __future__ import annotations

from typing import List

from models.candidate import Candidate
from models.embedding import CampaignEmbedding
from embedding_service.vector_store import VectorStore


class CandidateRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, campaign_embedding: CampaignEmbedding, top_n: int) -> List[Candidate]:
        raw_results = self.vector_store.query_similar_volunteers(campaign_embedding.embedding, top_n)
        candidates: List[Candidate] = []
        for raw in raw_results:
            sim_score = raw.get("similarity_score")
            if sim_score is None:
                # Fallback calculation if distance is provided
                dist = raw.get("distance", 1.0)
                sim_score = max(0.0, 1.0 - (dist / 2.0))

            candidates.append(
                Candidate(
                    volunteer_id=raw["volunteer_id"],
                    similarity_score=float(sim_score),
                    metadata={"document": raw.get("text", "")},
                )
            )
        return sorted(candidates, key=lambda candidate: candidate.similarity_score, reverse=True)

