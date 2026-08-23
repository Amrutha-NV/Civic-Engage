from __future__ import annotations

from typing import List, Optional, Union
from pydantic import BaseModel, Field

class Volunteer(BaseModel):
    id: str = Field(..., alias="_id")
    name: str = "Volunteer"
    email: Optional[str] = ""
    phone: Optional[str] = ""
    avatar: Optional[str] = ""
    location: Optional[str] = "Manila"
    coordinates: Optional[dict] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    skills: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    availability: Union[str, List[str]] = "Available"
    occupation: Optional[str] = "Volunteer"
    attendance_rate: float = 0.95
    social_impact_score: float = 85.0
    impact_score: Optional[float] = 85.0
    verified: bool = True
    active_campaigns: int = 0
    history_ids: List[str] = Field(default_factory=list)
    metadata: Optional[dict] = None

    class Config:
        populate_by_name = True
        extra = "ignore"
