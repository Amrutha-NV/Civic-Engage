from __future__ import annotations

from typing import List, Optional, Union
from pydantic import BaseModel, Field

class Campaign(BaseModel):
    id: str = Field(..., alias="_id")
    title: str
    description: Optional[str] = ""
    category: Optional[str] = "General"
    location: Optional[str] = ""
    coordinates: Optional[dict] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    required_skills: List[str] = Field(default_factory=list, alias="requiredSkills")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    date: Optional[str] = None
    maximum_volunteers: int = Field(default=50, alias="capacity")
    current_volunteers: int = Field(default=0, alias="volunteersJoined")
    status: Optional[str] = "Active"
    metadata: Optional[dict] = None

    class Config:
        populate_by_name = True
        extra = "ignore"
