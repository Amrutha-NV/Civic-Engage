from __future__ import annotations

import hashlib
import math
from typing import List

from config.settings import Settings
from models.embedding import CampaignEmbedding, VolunteerEmbedding
from models.volunteer import Volunteer
from models.campaign import Campaign
from embedding_service.text_builder import TextBuilder


class EmbeddingGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model_name)
        except Exception:
            self._model = None

    def _encode(self, text: str) -> List[float]:
        self._load_model()
        if self._model is not None:
            return self._model.encode(text, normalize_embeddings=True).tolist()
        return self._dummy_encode(text)

    @staticmethod
    def _dummy_encode(text: str, dimensions: int = 384) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [float(digest[i % len(digest)]) for i in range(dimensions)]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm > 0 else vector

    def generate_volunteer_embedding(self, volunteer: Volunteer) -> VolunteerEmbedding:
        text = TextBuilder.build_volunteer_text(volunteer)
        vector = self._encode(text)
        return VolunteerEmbedding(
            volunteer_id=volunteer.id,
            embedding=vector,
            location=volunteer.location,
            skills=volunteer.skills,
            metadata={
                "verified": volunteer.verified,
                "attendance_rate": volunteer.attendance_rate,
                "social_impact_score": volunteer.social_impact_score,
                "active_campaigns": volunteer.active_campaigns,
            },
        )

    def generate_campaign_embedding(self, campaign: Campaign) -> CampaignEmbedding:
        text = TextBuilder.build_campaign_text(campaign)
        vector = self._encode(text)
        return CampaignEmbedding(
            campaign_id=campaign.id,
            embedding=vector,
            category=campaign.category,
            metadata={
                "location": campaign.location,
                "required_skills": campaign.required_skills,
            },
        )
