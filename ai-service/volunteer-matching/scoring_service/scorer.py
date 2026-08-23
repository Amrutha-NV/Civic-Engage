from __future__ import annotations

from typing import Dict, List

from models.candidate import Candidate
from models.campaign import Campaign
from models.volunteer import Volunteer


import math

class Scorer:
    def __init__(self, weight_config: Dict[str, float]):
        self.weights = weight_config

    def score_candidates(
        self,
        campaign: Campaign,
        candidates: List[Candidate],
        volunteers: List[Volunteer],
    ) -> List[Candidate]:
        volunteer_map = {v.id: v for v in volunteers}
        scored_candidates: List[Candidate] = []
        for candidate in candidates:
            volunteer = volunteer_map.get(candidate.volunteer_id)
            if volunteer is None:
                continue
            business_score, breakdown = self._compute_business_score_and_breakdown(candidate, volunteer, campaign)
            final_score = self._compute_final_score(candidate.similarity_score, business_score)
            
            candidate.business_score = round(business_score, 4)
            candidate.final_score = round(final_score, 4)
            
            if candidate.metadata is None:
                candidate.metadata = {}
            candidate.metadata["breakdown"] = breakdown

            scored_candidates.append(candidate)
        return scored_candidates

    def _compute_business_score_and_breakdown(
        self, candidate: Candidate, volunteer: Volunteer, campaign: Campaign
    ) -> tuple[float, Dict[str, int]]:
        metadata = candidate.metadata or {}

        # 1. Skill overlap score (0.0 to 1.0)
        skill_overlap = metadata.get("skill_overlap_ratio", 0.75)
        # Blend semantic similarity with explicit skill overlap for skills breakdown
        skills_pct = int(min(99, max(60, (0.5 * candidate.similarity_score + 0.5 * skill_overlap) * 100)))

        # 2. Location proximity decay (0.0 to 1.0)
        distance_km = metadata.get("distance_km", 5.0)
        proximity_factor = math.exp(-distance_km / 35.0)

        # 3. Normalized Impact Score (0.0 to 1.0)
        raw_impact = volunteer.impact_score or volunteer.social_impact_score or 85.0
        impact_norm = (raw_impact / 100.0) if raw_impact > 1.0 else raw_impact

        # 4. Availability score (0.0 to 1.0)
        is_avail = metadata.get("is_available", True)
        avail_score = 1.0 if is_avail else 0.5
        avail_pct = 95 if is_avail else 75

        # 5. Attendance & Verification
        attendance = volunteer.attendance_rate if volunteer.attendance_rate <= 1.0 else (volunteer.attendance_rate / 100.0)
        verified_score = 1.0 if volunteer.verified else 0.7

        # 6. Interests match factor
        interests = [i.lower() for i in (volunteer.interests or [])]
        camp_cat = (campaign.category or "").lower()
        interests_match = any(camp_cat in i or i in camp_cat for i in interests) if interests else True
        interests_pct = 92 if interests_match else 78

        # Weighted business score computation
        w_skill = self.weights.get("skill_overlap", 0.30)
        w_prox = self.weights.get("proximity", 0.15)
        w_avail = self.weights.get("availability", 0.15)
        w_impact = self.weights.get("social_impact", 0.15)
        w_attend = self.weights.get("attendance", 0.15)
        w_verif = self.weights.get("verification", 0.10)

        business_score = (
            w_skill * skill_overlap +
            w_prox * proximity_factor +
            w_avail * avail_score +
            w_impact * impact_norm +
            w_attend * attendance +
            w_verif * verified_score
        )

        breakdown = {
            "skills": skills_pct,
            "interests": interests_pct,
            "availability": avail_pct,
            "proximity": int(proximity_factor * 100),
        }

        return min(1.0, max(0.1, business_score)), breakdown

    def _compute_final_score(self, similarity_score: float, business_score: float) -> float:
        similarity_weight = self.weights.get("similarity", 0.40)
        business_weight = self.weights.get("business", 0.60)
        return similarity_weight * similarity_score + business_weight * business_score

