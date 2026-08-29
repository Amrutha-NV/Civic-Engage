import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

# explainer.py is located at:
# Civic-Engage/ai-service/recommendation/explainer.py
#
# parents[2] points to:
# Civic-Engage/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)
load_dotenv()  # local fallback



# ============================================================
# RECOMMENDATION EXPLAINER
# ============================================================

class RecommendationExplainer:
    """
    Generates natural-language explanations for volunteer
    campaign recommendations.

    Groq is used when a valid GROQ_API_KEY is available.

    If Groq is unavailable or an API request fails, the system
    automatically falls back to a deterministic explanation so
    that recommendation generation never fails.
    """

    def __init__(self):
        # Load Groq API key from .env
        self.api_key = os.getenv("GROQ_API_KEY")

        # Create Groq client only when an API key is available
        self.client = (
            Groq(api_key=self.api_key)
            if self.api_key
            else None
        )

        # Use the model configured in .env.
        # Falls back to llama-3.3-70b-versatile if not specified.
        self.model = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile"
        )

    # ========================================================
    # DETERMINISTIC FALLBACK
    # ========================================================

    @staticmethod
    def _fallback_reason(
        recommendation: Dict[str, Any],
    ) -> str:
        """
        Generates a deterministic explanation when Groq is
        unavailable or the LLM request fails.
        """
        matched_skills = recommendation.get(
            "matchedSkills",
            []
        )

        breakdown = recommendation.get(
            "breakdown",
            {}
        )

        skill_score = breakdown.get(
            "skills",
            0
        )

        interest_score = breakdown.get(
            "interests",
            0
        )

        proximity_score = breakdown.get(
            "proximity",
            0
        )

        reasons: List[str] = []

        # ----------------------------------------------------
        # Skill matching
        # ----------------------------------------------------

        if matched_skills:
            skills = ", ".join(matched_skills)

            reasons.append(
                f"your skills ({skills}) match the "
                f"campaign requirements"
            )

        # ----------------------------------------------------
        # Interest matching
        # ----------------------------------------------------

        if interest_score >= 60:
            reasons.append(
                "the campaign aligns well with your interests"
            )

        # ----------------------------------------------------
        # Location / proximity
        # ----------------------------------------------------

        if proximity_score >= 60:

            distance = recommendation.get(
                "distanceKm"
            )

            if distance is not None:
                reasons.append(
                    f"the campaign is relatively close to you "
                    f"({distance} km away)"
                )
            else:
                reasons.append(
                    "the campaign has good location compatibility"
                )

        # ----------------------------------------------------
        # Overall skill compatibility
        # ----------------------------------------------------

        if skill_score >= 70 and not matched_skills:
            reasons.append(
                "your overall profile is well aligned with "
                "the required skills"
            )

        # ----------------------------------------------------
        # Generic fallback
        # ----------------------------------------------------

        if not reasons:
            reasons.append(
                "the campaign has a good overall match "
                "with your profile"
            )

        return (
            "Recommended because "
            + ", ".join(reasons)
            + "."
        )

    # ========================================================
    # GROQ EXPLANATION
    # ========================================================

    def explain(
        self,
        recommendation: Dict[str, Any],
        user_profile: Dict[str, Any],
    ) -> str:
        """
        Generate a natural-language explanation for a
        recommendation.

        Groq is used when configured.

        If Groq is unavailable or fails, a deterministic
        fallback explanation is returned.
        """
        title = recommendation.get("title", "this campaign")

        # ----------------------------------------------------
        # If no Groq API key is available
        # ----------------------------------------------------

        if not self.client:
            return self._fallback_reason(
                recommendation
            )

        # ----------------------------------------------------
        # Extract recommendation information
        # ----------------------------------------------------

        description = recommendation.get(
            "description",
            ""
        )

        matched_skills = recommendation.get(
            "matchedSkills",
            []
        )

        breakdown = recommendation.get(
            "breakdown",
            {}
        )

        match_score = recommendation.get(
            "matchScore",
            0
        )

        distance = recommendation.get(
            "distanceKm"
        )

        # ----------------------------------------------------
        # Construct LLM prompt
        # ----------------------------------------------------

        prompt = f"""
You are explaining why a volunteer campaign is recommended
for a particular volunteer.

VOLUNTEER PROFILE

Skills:
{user_profile.get("skills", [])}

Interests:
{user_profile.get("interests", [])}

Location:
{user_profile.get("location", "")}

Availability:
{user_profile.get("availability", "")}


CAMPAIGN

Title:
{title}

Description:
{description}


RECOMMENDATION DETAILS

Overall match score:
{match_score}

Matched skills:
{matched_skills}

Score breakdown:
{breakdown}

Distance:
{distance} km


TASK

Write ONE concise explanation of why this campaign is
recommended for this volunteer.

Mention the strongest matching factors.

Use only the information provided above.

Do not invent information.

Do not mention:
- AI
- language models
- prompts
- internal scoring
- token limits
- technical implementation

Keep the explanation under 40 words.
"""

        # ====================================================
        # CALL GROQ
        # ====================================================

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You explain volunteer recommendations "
                            "clearly, concisely, and honestly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=250,
            )

            # ------------------------------------------------
            # Extract generated explanation
            # ------------------------------------------------

            explanation = response.choices[0].message.content

            if explanation:
                explanation = explanation.strip()

                if explanation:
                    return explanation

        # ----------------------------------------------------
        # If Groq fails, do not break recommendation service
        # ----------------------------------------------------

        except Exception:
            pass

        # ----------------------------------------------------
        # Deterministic fallback
        # ----------------------------------------------------

        return self._fallback_reason(
            recommendation
        )

