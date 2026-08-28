from typing import List, Optional
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    id: str
    skills: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    coordinates: Optional[List[float]] = None
    availability: Optional[str] = None
    history: List[str] = Field(default_factory=list)


class CampaignEvent(BaseModel):
    id: str
    title: str
    description: str
    category: Optional[str] = None
    requiredSkills: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    coordinates: Optional[List[float]] = None
    date: Optional[str] = None


class RecommendationResponse(BaseModel):
    id: str
    title: str
    matchScore: float
    breakdown: dict
    distanceKm: Optional[float] = None
    matchedSkills: List[str] = Field(default_factory=list)
    reason: str