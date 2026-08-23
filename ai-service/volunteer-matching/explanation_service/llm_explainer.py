from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Any, Optional
import requests

from config.settings import Settings
from models.campaign import Campaign
from models.volunteer import Volunteer
from models.candidate import Candidate


class LLMExplanationService:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Load API keys from settings or environment variables
        self.groq_api_key = (
            settings.groq_api_key
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("GROQ_KEY")
        )
        self.openai_api_key = (
            settings.openai_api_key
            or os.environ.get("OPENAI_API_KEY")
        )
        self.gemini_api_key = (
            settings.gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
        )

    def generate_explanations(
        self,
        campaign: Campaign,
        candidates: List[Candidate],
        volunteers: List[Volunteer],
    ) -> Dict[str, str]:
        """
        Generates personalized 1-2 sentence recommendation justifications for candidates.
        Attempts LLM generation (Groq / OpenAI / Gemini) with high-speed deterministic fallback.
        """
        volunteer_map = {v.id: v for v in volunteers}
        candidate_data = []

        for cand in candidates:
            vol = volunteer_map.get(cand.volunteer_id)
            if not vol:
                continue

            metadata = cand.metadata or {}
            matched_skills = metadata.get("matched_skills", [])
            dist_km = metadata.get("distance_km", 5.0)
            avail_str = "Available" if metadata.get("is_available", True) else "Flexible"

            candidate_data.append({
                "id": str(cand.volunteer_id),
                "name": vol.name,
                "skills": vol.skills or [],
                "matched_skills": matched_skills,
                "interests": vol.interests or [],
                "distance_km": dist_km,
                "availability": avail_str,
                "attendance_rate": vol.attendance_rate,
                "impact_score": vol.impact_score or vol.social_impact_score or 85.0,
                "verified": vol.verified,
                "similarity_score": round(cand.similarity_score, 2),
                "final_score": round(cand.final_score, 2),
            })

        if not candidate_data:
            return {}

        # 1. Try Groq API if key is available
        if self.groq_api_key:
            try:
                llm_reasons = self._call_groq_api(campaign, candidate_data)
                if llm_reasons:
                    return self._fill_missing_with_fallback(llm_reasons, campaign, candidate_data)
            except Exception as e:
                print(f"[LLM Explanation] Groq API call failed: {e}. Using deterministic fallback.")

        # 2. Try OpenAI API if key is available
        if self.openai_api_key:
            try:
                llm_reasons = self._call_openai_api(campaign, candidate_data)
                if llm_reasons:
                    return self._fill_missing_with_fallback(llm_reasons, campaign, candidate_data)
            except Exception as e:
                print(f"[LLM Explanation] OpenAI API call failed: {e}. Using deterministic fallback.")

        # 3. Deterministic Heuristic Fallback
        return self._generate_heuristic_explanations(campaign, candidate_data)

    def _sanitize_reason(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Normalize non-standard unicode characters
        text = text.replace("\u202f", " ").replace("\xa0", " ").replace("\u200b", "")
        text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        return text.strip()

    def _call_groq_api(self, campaign: Campaign, candidates: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Calls Groq Chat Completions API using llama-3.3-70b-versatile or configured model.
        """
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are an expert AI matching assistant for the CivicEngage NGO volunteer platform. "
            "Your task is to write a concise, compelling 1-2 sentence explanation for each recommended volunteer "
            "explaining specifically why they are an outstanding match for the civic campaign. "
            "Highlight their matched skills, proximity, availability, or civic track record. "
            "You MUST respond ONLY with a valid JSON object where keys are volunteer IDs and values are explanation strings."
        )

        user_content = {
            "campaign": {
                "title": campaign.title,
                "category": campaign.category,
                "description": campaign.description,
                "required_skills": campaign.required_skills,
                "location": campaign.location,
            },
            "candidates": candidates,
        }

        payload = {
            "model": self.settings.groq_model or "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Generate matching explanations for these candidates:\n{json.dumps(user_content, indent=2)}\nReturn JSON format: {{ \"<volunteer_id>\": \"<1-2 sentence reason>\" }}",
                },
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=4.5)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)

        # Handle nested keys if LLM wraps in an object e.g. {"recommendations": {...}} or {"volunteers": {...}}
        raw_dict = {}
        if isinstance(parsed, dict):
            for nested_key in ["recommendations", "volunteers", "explanations", "results"]:
                if nested_key in parsed and isinstance(parsed[nested_key], dict):
                    raw_dict = parsed[nested_key]
                    break
            if not raw_dict:
                raw_dict = parsed

        return {str(k): self._sanitize_reason(str(v)) for k, v in raw_dict.items()}


    def _call_openai_api(self, campaign: Campaign, candidates: List[Dict[str, Any]]) -> Dict[str, str]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are an expert matching assistant for CivicEngage. "
            "Write a concise, compelling 1-2 sentence reason for each recommended volunteer. "
            "Respond ONLY with a JSON object mapping volunteer IDs to reason strings."
        )

        user_content = {
            "campaign": {
                "title": campaign.title,
                "category": campaign.category,
                "required_skills": campaign.required_skills,
                "location": campaign.location,
            },
            "candidates": candidates,
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_content)},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=4.5)
        response.raise_for_status()
        data = response.json()
        parsed = json.loads(data["choices"][0]["message"]["content"].strip())
        return {str(k): str(v) for k, v in parsed.items()}

    def _generate_heuristic_explanations(
        self, campaign: Campaign, candidates: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        High-speed deterministic explanation generator used when LLM is unavailable or offline.
        Produces natural, tailored justifications based on skills, proximity, and impact.
        """
        explanations = {}
        for c in candidates:
            v_id = c["id"]
            name = c["name"]
            matched_skills = c["matched_skills"]
            dist_km = c["distance_km"]
            impact = int(c["impact_score"])
            verified = c["verified"]

            skills_phrase = (
                f"verified expertise in {', '.join(matched_skills[:2])}"
                if matched_skills
                else f"demonstrated dedication in {campaign.category or 'civic service'}"
            )

            dist_phrase = f"located {dist_km:.1f} km away" if dist_km < 30 else "readily available"

            verif_phrase = "verified track record" if verified else "active community standing"

            reason = (
                f"Strong match for {campaign.title} with {skills_phrase}, {dist_phrase}, "
                f"and an exceptional {impact}% {verif_phrase}."
            )
            explanations[v_id] = reason

        return explanations

    def _fill_missing_with_fallback(
        self,
        llm_reasons: Dict[str, str],
        campaign: Campaign,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        fallback_reasons = self._generate_heuristic_explanations(campaign, candidates)
        merged = {}
        for c in candidates:
            v_id = c["id"]
            if v_id in llm_reasons and len(llm_reasons[v_id].strip()) > 10:
                merged[v_id] = llm_reasons[v_id].strip()
            else:
                merged[v_id] = fallback_reasons.get(v_id, "Highly recommended civic volunteer.")
        return merged
