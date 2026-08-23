from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Recommendation(BaseModel):
    volunteer_id: str
    similarity_score: float
    business_score: float
    final_score: float
    reason: Optional[str] = None
    metadata: Optional[dict] = None
