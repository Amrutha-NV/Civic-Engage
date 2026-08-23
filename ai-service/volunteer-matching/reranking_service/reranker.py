from typing import List, Optional, Dict

from models.candidate import Candidate
from models.recommendation import Recommendation


class Reranker:
    def rank(
        self,
        candidates: List[Candidate],
        top_k: int,
        reasons_map: Optional[Dict[str, str]] = None,
    ) -> List[Recommendation]:
        ranked = sorted(candidates, key=lambda item: item.final_score, reverse=True)
        recommendations: List[Recommendation] = []
        reasons_map = reasons_map or {}

        for candidate in ranked[:top_k]:
            v_id = str(candidate.volunteer_id)
            reason = reasons_map.get(v_id) or self._build_reason(candidate)

            recommendations.append(
                Recommendation(
                    volunteer_id=candidate.volunteer_id,
                    similarity_score=candidate.similarity_score,
                    business_score=candidate.business_score,
                    final_score=candidate.final_score,
                    reason=reason,
                    metadata=candidate.metadata,
                )
            )
        return recommendations

    @staticmethod
    def _build_reason(candidate: Candidate) -> str:
        metadata = candidate.metadata or {}
        matched_skills = metadata.get("matched_skills", [])
        if matched_skills:
            return f"Strong match with verified skills in {', '.join(matched_skills[:2])}."
        return f"High composite match with similarity {candidate.similarity_score:.2f} and quality score {candidate.business_score:.2f}."

