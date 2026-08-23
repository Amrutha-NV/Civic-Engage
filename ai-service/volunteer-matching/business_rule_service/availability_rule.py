from __future__ import annotations

import datetime
from typing import List, Union

from models.candidate import Candidate
from models.campaign import Campaign
from models.volunteer import Volunteer
from business_rule_service.rule_engine import BusinessRule


class AvailabilityRule(BusinessRule):
    def apply(self, campaign: Campaign, candidates: List[Candidate], volunteers: List[Volunteer]) -> List[Candidate]:
        volunteer_map = {v.id: v for v in volunteers}
        campaign_is_weekend = self._is_weekend_campaign(campaign)

        filtered: List[Candidate] = []
        for candidate in candidates:
            vol = volunteer_map.get(candidate.volunteer_id)
            if vol is None:
                continue

            avail = vol.availability
            available = self._is_available(avail, campaign_is_weekend)
            if candidate.metadata is None:
                candidate.metadata = {}
            candidate.metadata["is_available"] = available

            if available or len(candidates) <= 2:
                filtered.append(candidate)

        return filtered if filtered else candidates


    @staticmethod
    def _is_weekend_campaign(campaign: Campaign) -> bool:
        date_str = campaign.date or campaign.start_date
        if not date_str:
            return False
        try:
            # Parse date YYYY-MM-DD or similar standard ISO format
            clean_date = date_str.split("T")[0]
            dt = datetime.datetime.strptime(clean_date, "%Y-%m-%d")
            # 5 = Saturday, 6 = Sunday
            return dt.weekday() in (5, 6)
        except Exception:
            # Default to false (weekday) if date string parsing fails
            return False

    @staticmethod
    def _is_available(avail: Union[str, List[str]], is_weekend: bool) -> bool:
        if not avail:
            return True

        if isinstance(avail, list):
            avail_str = " ".join(avail).lower()
        else:
            avail_str = str(avail).lower()

        if "available" in avail_str or "flexible" in avail_str:
            return True

        if is_weekend and "weekend" in avail_str:
            return True

        if not is_weekend and "weekday" in avail_str:
            return True

        return False
