from __future__ import annotations

from typing import Optional

import requests
from pydantic import ValidationError

from config.settings import Settings
from models.campaign import Campaign


class CampaignFetcher:
    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch_campaign(self, campaign_id: str) -> Campaign:
        url = f"{self.settings.backend_base_url}{self.settings.campaign_endpoint}{campaign_id}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if hasattr(Campaign, "model_validate"):
                return Campaign.model_validate(data)
            return Campaign.parse_obj(data)
        except (requests.RequestException, ValidationError) as exc:
            raise RuntimeError(f"Failed to fetch campaign from backend: {exc}") from exc

    def fetch_campaign_dummy(self, campaign_id: str) -> Campaign:
        dummy = {
            "_id": campaign_id,
            "title": "Tree Planting Drive",
            "description": "Plant native trees in community parks.",
            "category": "environmental",
            "location": "Springfield",
            "required_skills": ["gardening", "teamwork"],
            "start_date": None,
            "end_date": None,
            "maximum_volunteers": 20,
            "current_volunteers": 5,
            "status": "active",
        }
        if hasattr(Campaign, "model_validate"):
            return Campaign.model_validate(dummy)
        return Campaign.parse_obj(dummy)

