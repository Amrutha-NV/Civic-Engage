from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from models.campaign import Campaign
from models.candidate import Candidate
from models.embedding import CampaignEmbedding
from models.volunteer import Volunteer


@dataclass
class PipelineContext:
    campaign: Optional[Campaign] = None
    volunteers: List[Volunteer] = field(default_factory=list)
    campaign_embedding: Optional[CampaignEmbedding] = None
    candidates: List[Candidate] = field(default_factory=list)
    recommendations: List = field(default_factory=list)
