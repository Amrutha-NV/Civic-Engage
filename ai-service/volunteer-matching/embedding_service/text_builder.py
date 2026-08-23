from __future__ import annotations

from typing import List

from models.campaign import Campaign
from models.volunteer import Volunteer


class TextBuilder:
    @staticmethod
    def build_volunteer_text(volunteer: Volunteer) -> str:
        skills = ", ".join(volunteer.skills) if isinstance(volunteer.skills, list) else str(volunteer.skills or "")
        interests = ", ".join(volunteer.interests) if isinstance(volunteer.interests, list) else str(volunteer.interests or "")
        
        if isinstance(volunteer.availability, list):
            availability = ", ".join(volunteer.availability)
        else:
            availability = str(volunteer.availability or "Available")

        occupation = volunteer.occupation or "Volunteer"
        location = volunteer.location or "Community"
        impact = volunteer.impact_score or volunteer.social_impact_score or 85.0

        return (
            f"Volunteer profile for {volunteer.name} ({occupation}) based in {location}. "
            f"Key skills and competencies: {skills if skills else 'community service, civic participation'}. "
            f"Passions and cause interests: {interests if interests else 'civic engagement'}. "
            f"Schedule availability: {availability}. "
            f"Track record attendance rate: {volunteer.attendance_rate:.2f}, "
            f"social impact score: {impact:.1f}. "
            f"Verified civic participant: {volunteer.verified}."
        )

    @staticmethod
    def build_campaign_text(campaign: Campaign) -> str:
        required_skills = (
            ", ".join(campaign.required_skills)
            if isinstance(campaign.required_skills, list)
            else str(campaign.required_skills or "")
        )
        category = campaign.category or "Civic & Community"
        location = campaign.location or "Local Area"
        description = campaign.description or "Civic community empowerment initiative."

        return (
            f"Civic campaign event: {campaign.title}. Category focus: {category}. "
            f"Location: {location}. "
            f"Overview and goals: {description}. "
            f"Required volunteer skills: {required_skills if required_skills else 'enthusiasm, teamwork, dedication'}. "
            f"Volunteer team capacity: {campaign.maximum_volunteers} volunteers."
        )

