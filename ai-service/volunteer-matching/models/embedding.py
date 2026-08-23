from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class VolunteerEmbedding(BaseModel):
    volunteer_id: str
    embedding: List[float]
    location: str
    skills: List[str]
    metadata: Optional[dict] = None


class CampaignEmbedding(BaseModel):
    campaign_id: str
    embedding: List[float]
    category: str
    metadata: Optional[dict] = None
