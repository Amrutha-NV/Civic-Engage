from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class Candidate(BaseModel):
    volunteer_id: str
    similarity_score: float
    metadata: Dict[str, Optional[object]]
    business_score: float = 0.0
    final_score: float = 0.0
