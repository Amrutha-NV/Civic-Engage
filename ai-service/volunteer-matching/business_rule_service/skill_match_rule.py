from __future__ import annotations

from typing import List, Set
from models.candidate import Candidate
from models.campaign import Campaign
from models.volunteer import Volunteer
from business_rule_service.rule_engine import BusinessRule


class SkillMatchRule(BusinessRule):
    """
    Evaluates skill compatibility between campaign required skills and volunteer skills.
    Attaches matched_skills, matched_skills_count, and skill_overlap_ratio to candidate metadata.
    """
    def __init__(self, min_overlap_ratio: float = 0.0):
        self.min_overlap_ratio = min_overlap_ratio

    def apply(self, campaign: Campaign, candidates: List[Candidate], volunteers: List[Volunteer]) -> List[Candidate]:
        volunteer_map = {v.id: v for v in volunteers}
        req_skills = campaign.required_skills or []
        req_skills_lower = [s.strip().lower() for s in req_skills if s]

        filtered_candidates: List[Candidate] = []

        for candidate in candidates:
            vol = volunteer_map.get(candidate.volunteer_id)
            if vol is None:
                continue

            vol_skills = vol.skills or []
            vol_skills_lower = [s.strip().lower() for s in vol_skills if s]

            matched_skills: List[str] = []
            if req_skills_lower:
                for req in req_skills_lower:
                    for vs in vol.skills:
                        vs_lower = vs.strip().lower()
                        if req in vs_lower or vs_lower in req:
                            if vs not in matched_skills:
                                matched_skills.append(vs)
                            break

                overlap_ratio = len(matched_skills) / len(req_skills_lower)
            else:
                # If campaign doesn't specify skills, all volunteer skills are valid
                matched_skills = list(vol.skills[:3])
                overlap_ratio = 1.0

            if candidate.metadata is None:
                candidate.metadata = {}

            candidate.metadata["matched_skills"] = matched_skills
            candidate.metadata["skill_overlap_ratio"] = round(overlap_ratio, 3)
            candidate.metadata["matched_skills_count"] = len(matched_skills)

            if overlap_ratio >= self.min_overlap_ratio or len(candidates) <= 3:
                filtered_candidates.append(candidate)

        # Fallback: if hard filtering eliminated everyone, preserve original candidates
        return filtered_candidates if filtered_candidates else candidates
