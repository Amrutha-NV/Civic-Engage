from __future__ import annotations

from typing import List

import requests
from pydantic import ValidationError

from config.settings import Settings
from models.volunteer import Volunteer


class VolunteerFetcher:
    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch_volunteers(self) -> List[Volunteer]:
        url = f"{self.settings.backend_base_url}{self.settings.users_endpoint}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if hasattr(Volunteer, "model_validate"):
                return [Volunteer.model_validate(item) for item in data]
            return [Volunteer.parse_obj(item) for item in data]
        except (requests.RequestException, ValidationError) as exc:
            raise RuntimeError(f"Failed to fetch volunteers from backend: {exc}") from exc

    def fetch_volunteers_dummy(self) -> List[Volunteer]:
        dummy = [
            {
                "_id": "vol1",
                "name": "Alice Morrison",
                "email": "alice@example.org",
                "location": "Springfield",
                "skills": ["gardening", "event planning"],
                "availability": ["weekends"],
                "attendance_rate": 0.92,
                "social_impact_score": 0.82,
                "verified": True,
                "active_campaigns": 1,
                "history_ids": ["hist1", "hist2"],
            },
            {
                "_id": "vol2",
                "name": "Brian Kim",
                "email": "brian@example.org",
                "location": "Shelbyville",
                "skills": ["first aid", "medical support"],
                "availability": ["weekdays"],
                "attendance_rate": 0.87,
                "social_impact_score": 0.91,
                "verified": False,
                "active_campaigns": 0,
                "history_ids": ["hist3"],
            },
        ]
        if hasattr(Volunteer, "model_validate"):
            return [Volunteer.model_validate(item) for item in dummy]
        return [Volunteer.parse_obj(item) for item in dummy]

