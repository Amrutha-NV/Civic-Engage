from __future__ import annotations

from typing import List

from models.candidate import Candidate
from models.campaign import Campaign
from models.volunteer import Volunteer
from business_rule_service.rule_engine import BusinessRule


class VerificationRule(BusinessRule):
    def apply(self, campaign: Campaign, candidates: List[Candidate], volunteers: List[Volunteer]) -> List[Candidate]:
        volunteer_map = {v.id: v for v in volunteers}
        return [
            candidate
            for candidate in candidates
            if volunteer_map.get(candidate.volunteer_id) is not None
            and volunteer_map[candidate.volunteer_id].verified
        ]
