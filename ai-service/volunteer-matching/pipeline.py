from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable

from config.settings import Settings
from embedding_service.embedding_generator import EmbeddingGenerator
from embedding_service.vector_store import VectorStore
from fetching_service.campaign_fetcher import CampaignFetcher
from fetching_service.volunteer_fetcher import VolunteerFetcher
from candidate_generation_service.retriever import CandidateRetriever
from business_rule_service.rule_engine import RuleEngine
from business_rule_service.skill_match_rule import SkillMatchRule
from business_rule_service.availability_rule import AvailabilityRule
from business_rule_service.max_campaign_rule import MaxCampaignRule
from business_rule_service.location_rule import LocationRule
from scoring_service.scorer import Scorer
from scoring_service.weight_config import get_weights_for_category
from reranking_service.reranker import Reranker
from explanation_service.llm_explainer import LLMExplanationService
from utils.pipeline_context import PipelineContext
from models.campaign import Campaign
from models.volunteer import Volunteer


class RecommendationPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.fetch_campaigner = CampaignFetcher(settings)
        self.fetch_volunteer = VolunteerFetcher(settings)
        self.embedding_generator = EmbeddingGenerator(settings)
        self.vector_store = VectorStore(settings)
        self.candidate_retriever = CandidateRetriever(self.vector_store)
        self.rule_engine = RuleEngine([
            SkillMatchRule(min_overlap_ratio=0.0),
            AvailabilityRule(),
            LocationRule(max_distance_km=settings.max_distance_km),
            MaxCampaignRule(max_active_campaigns=3),
        ])
        self.reranker = Reranker()
        self.explainer = LLMExplanationService(settings)
        self.scorer: Optional[Scorer] = None

    def recommend(
        self, 
        campaign_id: str, 
        use_dummy_data: bool = False,
        stage_callback: Optional[Callable[[str, Any], None]] = None
    ) -> list[dict]:
        """
        Executes the full end-to-end recommendation workflow fetching campaign & volunteers by ID.
        """
        def trigger_stage_trace(stage_name: str, payload: Any) -> None:
            if stage_callback:
                stage_callback(stage_name, payload)

        context = PipelineContext()

        # === STAGE 1: Fetch Stage ===
        if use_dummy_data:
            context.campaign = self.fetch_campaigner.fetch_campaign_dummy(campaign_id)
            context.volunteers = self.fetch_volunteer.fetch_volunteers_dummy()
        else:
            context.campaign = self.fetch_campaigner.fetch_campaign(campaign_id)
            context.volunteers = self.fetch_volunteer.fetch_volunteers()
            
        trigger_stage_trace("1_fetch", {
            "campaign": context.campaign,
            "volunteers": context.volunteers
        })

        return self.recommend_from_payload(context.campaign, context.volunteers, stage_callback)

    def recommend_from_payload(
        self,
        campaign: Campaign,
        volunteers: List[Volunteer],
        stage_callback: Optional[Callable[[str, Any], None]] = None
    ) -> list[dict]:
        """
        Executes the recommendation workflow directly on passed Campaign & Volunteer Pydantic models.
        """
        def trigger_stage_trace(stage_name: str, payload: Any) -> None:
            if stage_callback:
                stage_callback(stage_name, payload)

        if not volunteers:
            return []

        # === STAGE 2: Embedding Stage ===
        campaign_embedding = self.embedding_generator.generate_campaign_embedding(campaign)
        
        volunteer_embeddings = []
        for volunteer in volunteers:
            volunteer_embedding = self.embedding_generator.generate_volunteer_embedding(volunteer)
            self.vector_store.upsert_volunteer_embedding(volunteer_embedding)
            volunteer_embeddings.append(volunteer_embedding)
            
        trigger_stage_trace("2_embedding", {
            "campaign_embedding": campaign_embedding,
            "volunteer_embeddings": volunteer_embeddings
        })

        # === STAGE 3: Candidate Generation Stage (Cosine Similarity) ===
        top_n = max(len(volunteers), self.settings.top_n_candidates)
        candidates = self.candidate_retriever.retrieve(
            campaign_embedding, 
            top_n
        )
        trigger_stage_trace("3_candidate_generation", candidates)

        # === STAGE 4: Business Rule Stage (Skill match, Geofencing, Availability) ===
        filtered_candidates = self.rule_engine.apply(
            campaign, 
            candidates, 
            volunteers
        )
        trigger_stage_trace("4_business_rules", filtered_candidates)

        # === STAGE 5: Scoring Stage ===
        current_scorer = self.scorer
        if current_scorer is None:
            weights = get_weights_for_category(campaign.category or "environmental")
            current_scorer = Scorer(weights)
            
        scored_candidates = current_scorer.score_candidates(
            campaign, 
            filtered_candidates, 
            volunteers
        )
        trigger_stage_trace("5_scoring", scored_candidates)

        # === STAGE 6: Re-ranking & Top-K Selection ===
        top_k = min(len(scored_candidates), self.settings.top_k_recommendations)
        if top_k == 0 and scored_candidates:
            top_k = len(scored_candidates)

        top_candidates = sorted(scored_candidates, key=lambda c: c.final_score, reverse=True)[:top_k]

        # === STAGE 7: LLM Explanation Stage ===
        reasons_map = self.explainer.generate_explanations(
            campaign,
            top_candidates,
            volunteers
        )
        trigger_stage_trace("7_llm_explanations", reasons_map)

        recommendations = self.reranker.rank(
            top_candidates, 
            top_k if top_k > 0 else 10,
            reasons_map=reasons_map
        )
        trigger_stage_trace("8_reranking", recommendations)

        # Build output format matching backend and frontend contracts
        volunteer_map = {v.id: v for v in volunteers}
        final_output = []

        for rec in recommendations:
            vol = volunteer_map.get(rec.volunteer_id)
            if not vol:
                continue

            metadata = rec.metadata or {}
            matched_skills = metadata.get("matched_skills")
            if matched_skills is None:
                req_skills_set = set(s.lower() for s in (campaign.required_skills or []))
                v_skills_set = set(s.lower() for s in (vol.skills or []))
                matched_skills = list(v_skills_set.intersection(req_skills_set))

            # Calibrate match percentage into smooth [65, 99]% range
            if rec.final_score <= 1.0:
                raw_pct = rec.final_score * 100
            else:
                raw_pct = rec.final_score
            match_pct = int(min(99, max(68, round(raw_pct))))

            dist_km = float(metadata.get("distance_km", 5.0))
            proximity_pct = int(min(99, max(65, round(100.0 / (1.0 + (dist_km / 15.0))))))

            raw_breakdown = metadata.get("breakdown") or {}
            breakdown = {
                "skills": raw_breakdown.get("skills", int(min(98, max(70, round(rec.similarity_score * 100))))),
                "interests": raw_breakdown.get("interests", int(min(95, max(75, round(rec.business_score * 100))))),
                "availability": raw_breakdown.get("availability", 95 if vol.availability else 80),
                "proximity": raw_breakdown.get("proximity", proximity_pct),
            }

            final_output.append({
                "id": vol.id,
                "userId": vol.id,
                "name": vol.name,
                "email": vol.email or "",
                "phone": vol.phone or "",
                "avatar": vol.avatar or "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
                "skills": vol.skills or [],
                "interests": vol.interests or [],
                "availability": vol.availability or "Available",
                "location": vol.location or "Manila",
                "coordinates": vol.coordinates,
                "impactScore": vol.impact_score or vol.social_impact_score or 85,
                "matchPercent": match_pct,
                "similarityScore": round(rec.similarity_score, 4),
                "businessScore": round(rec.business_score, 4),
                "finalScore": round(rec.final_score, 4),
                "breakdown": breakdown,
                "whyRecommended": rec.reason,
                "matchedSkills": matched_skills,
                "distanceKm": round(dist_km, 1),
                "activeCampaigns": vol.active_campaigns,
                "historyCount": len(vol.history_ids or [])
            })


        final_output.sort(key=lambda x: x["matchPercent"], reverse=True)
        trigger_stage_trace("9_final_output", final_output)

        return final_output

