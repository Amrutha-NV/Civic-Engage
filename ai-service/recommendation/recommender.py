import math
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRecommender:
    """
    Semantic volunteer-to-campaign recommendation engine.

    Combines:
    - Semantic similarity using all-MiniLM-L6-v2
    - Explicit skill matching
    - Interest/category affinity
    - Geographic proximity
    - Availability matching
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    @staticmethod
    def _normalise_list(values: Any) -> List[str]:
        """Convert a value into a clean lowercase list of strings."""
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        if not isinstance(values, list):
            return []

        return [
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        ]

    @staticmethod
    def _normalise_text(value: Any) -> str:
        """Convert a value to normalized lowercase text."""
        if value is None:
            return ""

        return str(value).strip().lower()

    @staticmethod
    def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Calculate cosine similarity without relying on sklearn."""
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = float(
            np.dot(vector_a, vector_b) / (norm_a * norm_b)
        )

        return float(np.clip(similarity, 0.0, 1.0))

    @staticmethod
    def haversine_distance(
        coordinates_a: Optional[List[float]],
        coordinates_b: Optional[List[float]],
    ) -> Optional[float]:
        """
        Calculate distance between two latitude/longitude points.

        Coordinates must be:
        [latitude, longitude]

        Returns:
            Distance in kilometres, or None if coordinates are unavailable.
        """
        if not coordinates_a or not coordinates_b:
            return None

        if len(coordinates_a) < 2 or len(coordinates_b) < 2:
            return None

        try:
            lat1, lon1 = float(coordinates_a[0]), float(coordinates_a[1])
            lat2, lon2 = float(coordinates_b[0]), float(coordinates_b[1])
        except (TypeError, ValueError):
            return None

        earth_radius_km = 6371.0

        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

        delta_lat = math.radians(float(coordinates_b[0]) - float(coordinates_a[0]))
        delta_lon = math.radians(float(coordinates_b[1]) - float(coordinates_a[1]))

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(delta_lon / 2) ** 2
        )

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return round(earth_radius_km * c, 2)

    @staticmethod
    def proximity_score(distance_km: Optional[float]) -> float:
        """
        Convert geographic distance into a continuous proximity score.

        Nearby campaigns receive a higher score while distant campaigns
        gradually receive a lower score.
        """
        if distance_km is None:
            return 0.5

        # Continuous exponential decay.
        score = math.exp(-distance_km / 15.0)

        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def skill_match(
        user_skills: List[str],
        required_skills: List[str],
    ) -> tuple[float, List[str]]:
        """
        Calculate explicit skill overlap.

        Returns:
            (skill_score, matched_skills)
        """
        user_set = set(user_skills)
        required_set = set(required_skills)

        if not required_set:
            return 0.5, []

        matched = sorted(user_set.intersection(required_set))

        score = len(matched) / len(required_set)

        return float(np.clip(score, 0.0, 1.0)), matched

    @staticmethod
    def interest_category_score(
        user_interests: List[str],
        category: str,
        title: str,
        description: str,
    ) -> float:
        """
        Calculate affinity between volunteer interests and campaign content.
        """
        if not user_interests:
            return 0.5

        campaign_text = " ".join(
            [
                category,
                title,
                description,
            ]
        ).lower()

        matches = 0

        for interest in user_interests:
            if interest in campaign_text:
                matches += 1

        return float(
            np.clip(matches / len(user_interests), 0.0, 1.0)
        )

    @staticmethod
    def availability_score(
        user_availability: str,
        event_date: str,
    ) -> float:
        """
        Calculate a simple availability compatibility score.

        The score remains neutral when campaign date/availability information
        is missing.
        """
        if not user_availability or not event_date:
            return 0.5

        availability = user_availability.lower().strip()
        date_text = event_date.lower().strip()

        # Direct textual overlap.
        if availability in date_text or date_text in availability:
            return 1.0

        # Common weekend indication.
        weekend_terms = ["weekend", "saturday", "sunday"]

        if any(term in availability for term in weekend_terms):
            if any(term in date_text for term in weekend_terms):
                return 1.0

        return 0.5

    @staticmethod
    def build_user_text(user_data: Dict[str, Any]) -> str:
        """Build semantic text representing the volunteer."""
        skills = " ".join(
            SemanticRecommender._normalise_list(
                user_data.get("skills", [])
            )
        )

        interests = " ".join(
            SemanticRecommender._normalise_list(
                user_data.get("interests", [])
            )
        )

        location = SemanticRecommender._normalise_text(
            user_data.get("location", "")
        )

        history = " ".join(
            SemanticRecommender._normalise_list(
                user_data.get("history", [])
            )
        )

        return (
            f"skills {skills}. "
            f"interests {interests}. "
            f"location {location}. "
            f"previous volunteer experience {history}."
        ).strip()

    @staticmethod
    def build_event_text(event: Dict[str, Any]) -> str:
        """Build semantic text representing a campaign."""
        title = SemanticRecommender._normalise_text(
            event.get("title", "")
        )

        description = SemanticRecommender._normalise_text(
            event.get("description", "")
        )

        category = SemanticRecommender._normalise_text(
            event.get("category", "")
        )

        required_skills = " ".join(
            SemanticRecommender._normalise_list(
                event.get("requiredSkills", [])
            )
        )

        location = SemanticRecommender._normalise_text(
            event.get("location", "")
        )

        return (
            f"campaign title {title}. "
            f"description {description}. "
            f"category {category}. "
            f"required skills {required_skills}. "
            f"location {location}."
        ).strip()

    def recommend(
        self,
        user_data: Dict[str, Any],
        events_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate ranked campaign recommendations for a volunteer.
        """
        if not events_data:
            return []

        user_skills = self._normalise_list(
            user_data.get("skills", [])
        )

        user_interests = self._normalise_list(
            user_data.get("interests", [])
        )

        user_availability = self._normalise_text(
            user_data.get("availability", "")
        )

        user_coordinates = user_data.get("coordinates")

        user_text = self.build_user_text(user_data)

        # Create semantic embedding for the volunteer.
        user_embedding = self.model.encode(
            user_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        results = []

        for event in events_data:
            event_text = self.build_event_text(event)

            event_embedding = self.model.encode(
                event_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            semantic_score = self._cosine_similarity(
                user_embedding,
                event_embedding,
            )

            required_skills = self._normalise_list(
                event.get("requiredSkills", [])
            )

            skill_score, matched_skills = self.skill_match(
                user_skills,
                required_skills,
            )

            category_score = self.interest_category_score(
                user_interests,
                self._normalise_text(event.get("category", "")),
                self._normalise_text(event.get("title", "")),
                self._normalise_text(event.get("description", "")),
            )

            distance_km = self.haversine_distance(
                user_coordinates,
                event.get("coordinates"),
            )

            proximity = self.proximity_score(distance_km)

            availability = self.availability_score(
                user_availability,
                self._normalise_text(event.get("date", "")),
            )

            # Multi-factor weighted score.
            #
            # Semantic similarity: 40%
            # Explicit skills:     30%
            # Interests/category:  15%
            # Proximity:           10%
            # Availability:         5%
            final_score = (
                semantic_score * 0.40
                + skill_score * 0.30
                + category_score * 0.15
                + proximity * 0.10
                + availability * 0.05
            )

            match_score = round(
                float(np.clip(final_score * 100, 0.0, 100.0)),
                2,
            )

            breakdown = {
                "skills": round(skill_score * 100, 2),
                "interests": round(category_score * 100, 2),
                "proximity": round(proximity * 100, 2),
                "availability": round(availability * 100, 2),
            }

            event_copy = dict(event)

            event_copy["matchScore"] = match_score
            event_copy["breakdown"] = breakdown
            event_copy["distanceKm"] = distance_km
            event_copy["matchedSkills"] = matched_skills

            # Temporary deterministic reason.
            # Groq-based explanation will be added in explainer.py.
            if matched_skills:
                skill_text = ", ".join(matched_skills)
                reason = (
                    f"Strong match based on your {skill_text} skills "
                    f"and campaign requirements."
                )
            else:
                reason = (
                    "Recommended based on semantic similarity between "
                    "your interests and the campaign."
                )

            event_copy["reason"] = reason

            results.append(event_copy)

        # Highest match first.
        results.sort(
            key=lambda item: (
                item.get("matchScore", 0),
                item.get("title", ""),
            ),
            reverse=True,
        )

        return results


# Backward-compatible function.
# This allows existing code that calls recommend_events()
# to continue working while we migrate the service.
def recommend_events(
    user_data: Dict[str, Any],
    events_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    recommender = SemanticRecommender()
    return recommender.recommend(user_data, events_data)