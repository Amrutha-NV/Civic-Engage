from __future__ import annotations

from typing import List

from models.candidate import Candidate
from models.campaign import Campaign
from models.volunteer import Volunteer


class RuleEngine:
    def __init__(self, rules: List["BusinessRule"]):
        self.rules = rules

    def apply(self, campaign: Campaign, candidates: List[Candidate], volunteers: List[Volunteer]) -> List[Candidate]:
        filtered_candidates = candidates
        for rule in self.rules:
            filtered_candidates = rule.apply(campaign, filtered_candidates, volunteers)
        return filtered_candidates


class BusinessRule:
    def apply(self, campaign: Campaign, candidates: List[Candidate], volunteers: List[Volunteer]) -> List[Candidate]:
        raise NotImplementedError
