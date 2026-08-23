from __future__ import annotations
from typing import Dict, Any, List, Optional
import sys
import traceback

from config.settings import Settings
from embedding_service.embedding_generator import EmbeddingGenerator
from embedding_service.vector_store import VectorStore
from fetching_service.campaign_fetcher import CampaignFetcher
from fetching_service.volunteer_fetcher import VolunteerFetcher
from candidate_generation_service.retriever import CandidateRetriever
from business_rule_service.rule_engine import RuleEngine
from business_rule_service.availability_rule import AvailabilityRule
from business_rule_service.verification_rule import VerificationRule
from business_rule_service.max_campaign_rule import MaxCampaignRule
from scoring_service.scorer import Scorer
from scoring_service.weight_config import get_weights_for_category
from reranking_service.reranker import Reranker
from utils.pipeline_context import PipelineContext
from pipeline import RecommendationPipeline


class RecommendationPipelineInspector:
    def __init__(self, settings: Settings, pipeline: RecommendationPipeline):
        self.settings = settings
        self.pipeline = pipeline
        self.stage_outputs: Dict[str, Any] = {}

    def run_fetch_stage(self) -> Dict[str, Any]:
        """Stage 1: Fetch campaign and volunteer data."""
        print("[TRACE] Inside run_fetch_stage: Requesting campaign dummy data...")
        campaign = self.pipeline.fetch_campaigner.fetch_campaign_dummy("camp1")
        print(f"[TRACE] Campaign retrieved successfully (ID: {getattr(campaign, 'id', 'unknown')})")
        
        print("[TRACE] Requesting volunteer dummy data list...")
        volunteers = self.pipeline.fetch_volunteer.fetch_volunteers_dummy()
        print(f"[TRACE] Volunteer list retrieved successfully. Count: {len(volunteers)}")
        
        self.stage_outputs["fetch"] = {
            "campaign": campaign,
            "volunteers": volunteers
        }
        return self.stage_outputs["fetch"]

    def run_embedding_stage(self) -> Dict[str, Any]:
        """Stage 2: Generate and store vector embeddings."""
        campaign = self.stage_outputs["fetch"]["campaign"]
        volunteers = self.stage_outputs["fetch"]["volunteers"]
        
        print("[TRACE] Inside run_embedding_stage: Generating campaign vector embedding...")
        campaign_embedding = self.pipeline.embedding_generator.generate_campaign_embedding(campaign)
        emb_len = len(getattr(campaign_embedding, "embedding", []))
        print(f"[TRACE] Campaign embedding completed. Dimensionality length: {emb_len}")
        
        volunteer_embeddings = []
        for i, volunteer in enumerate(volunteers, 1):
            vol_id = getattr(volunteer, "id", f"index_{i}")
            print(f"[TRACE] Iteration {i}: Generating vector embedding for volunteer ID: {vol_id}...")
            vol_emb = self.pipeline.embedding_generator.generate_volunteer_embedding(volunteer)
            
            print(f"[TRACE] Iteration {i}: Upserting volunteer vector embedding into vector_store...")
            self.pipeline.vector_store.upsert_volunteer_embedding(vol_emb)
            print(f"[TRACE] Iteration {i}: Upsert confirmation received for volunteer ID: {vol_id}")
            volunteer_embeddings.append(vol_emb)
            
        self.stage_outputs["embedding"] = {
            "campaign_embedding": campaign_embedding,
            "volunteer_embeddings": volunteer_embeddings
        }
        return self.stage_outputs["embedding"]

    def run_candidate_generation_stage(self) -> List[Any]:
        """Stage 3: Retrieve top-N candidate matches."""
        campaign_embedding = self.stage_outputs["embedding"]["campaign_embedding"]
        
        print(f"[TRACE] Inside run_candidate_generation_stage: querying retriever for top_n: {self.settings.top_n_candidates}...")
        candidates = self.pipeline.candidate_retriever.retrieve(
            campaign_embedding, 
            self.settings.top_n_candidates
        )
        print(f"[TRACE] Retriever response received. Found raw matches count: {len(candidates)}")
        
        self.stage_outputs["candidate_generation"] = candidates
        return candidates

    def run_business_rule_stage(self) -> List[Any]:
        """Stage 4: Apply hard filtering constraints."""
        campaign = self.stage_outputs["fetch"]["campaign"]
        candidates = self.stage_outputs["candidate_generation"]
        volunteers = self.stage_outputs["fetch"]["volunteers"]
        
        print(f"[TRACE] Inside run_business_rule_stage: passing {len(candidates)} candidates into rule_engine.apply()...")
        filtered_candidates = self.pipeline.rule_engine.apply(campaign, candidates, volunteers)
        print(f"[TRACE] Rule engine filtering pipeline completed. Output count: {len(filtered_candidates)}")
        
        self.stage_outputs["business_rules"] = filtered_candidates
        return filtered_candidates

    def run_scoring_stage(self) -> List[Any]:
        """Stage 5: Score candidates using dynamic weights."""
        campaign = self.stage_outputs["fetch"]["campaign"]
        filtered_candidates = self.stage_outputs["business_rules"]
        volunteers = self.stage_outputs["fetch"]["volunteers"]
        
        print("[TRACE] Inside run_scoring_stage: resolving pipeline Scorer instance...")
        scorer = getattr(self.pipeline, "scorer", None)
        if scorer is None:
            cat = getattr(campaign, "category", "default")
            print(f"[TRACE] pipeline.scorer is undefined. Building new Scorer using category weights mapping for: '{cat}'")
            scorer = Scorer(get_weights_for_category(cat))
            
        print(f"[TRACE] Triggering candidate scoring calculations for {len(filtered_candidates)} items...")
        scored_candidates = scorer.score_candidates(campaign, filtered_candidates, volunteers)
        print(f"[TRACE] Scoring math completed successfully.")
        
        self.stage_outputs["scoring"] = scored_candidates
        return scored_candidates

    def run_llm_explanation_stage(self) -> Dict[str, str]:
        """Stage 7: Generate personalized reasons using Groq / LLM."""
        campaign = self.stage_outputs["fetch"]["campaign"]
        scored_candidates = self.stage_outputs["scoring"]
        volunteers = self.stage_outputs["fetch"]["volunteers"]

        print("[TRACE] Inside run_llm_explanation_stage: generating justifications via LLM explainer...")
        reasons_map = self.pipeline.explainer.generate_explanations(
            campaign, scored_candidates, volunteers
        )
        print(f"[TRACE] Explanation generator completed. Generated reasons for {len(reasons_map)} candidates.")
        self.stage_outputs["explanations"] = reasons_map
        return reasons_map

    def run_reranking_stage(self) -> List[Any]:
        """Stage 8: Re-rank with attached LLM justifications and cut off at top-K recommendations."""
        scored_candidates = self.stage_outputs["scoring"]
        reasons_map = self.stage_outputs.get("explanations", {})
        
        print(f"[TRACE] Inside run_reranking_stage: sorting and filtering to top_k: {self.settings.top_k_recommendations}...")
        recommendations = self.pipeline.reranker.rank(
            scored_candidates, 
            self.settings.top_k_recommendations,
            reasons_map=reasons_map
        )
        print(f"[TRACE] Reranker completed successfully. Final recommendation count: {len(recommendations)}")
        
        self.stage_outputs["reranking"] = recommendations
        return recommendations


def main() -> None:
    settings = Settings()
    pipeline = RecommendationPipeline(settings)
    inspector = RecommendationPipelineInspector(settings, pipeline)
    
    try:
        # --- 1. Fetch Stage ---
        print("\n=== Stage 1: Executing Fetch Stage ===")
        fetch_data = inspector.run_fetch_stage()
        print("Campaign Data:", fetch_data["campaign"].model_dump())
        print(f"Volunteers Count: {len(fetch_data['volunteers'])}")
        
        # --- 2. Embedding Stage ---
        print("\n=== Stage 2: Executing Embedding Stage ===")
        emb_data = inspector.run_embedding_stage()
        camp_emb = emb_data["campaign_embedding"]
        print(f"Result -> Campaign Embedding length: {len(camp_emb.embedding)}")
        print(f"Result -> Upserted {len(emb_data['volunteer_embeddings'])} volunteer embeddings.")
        
        # --- 3. Candidate Generation Stage ---
        print("\n=== Stage 3: Executing Candidate Generation Stage (Cosine Similarity) ===")
        candidates = inspector.run_candidate_generation_stage()
        print(f"Result -> Retrieved {len(candidates)} candidates:")
        for candidate in candidates:
            print(f"  Candidate ID: {candidate.volunteer_id} | Cosine Sim: {candidate.similarity_score:.4f}")
            
        # --- 4. Business Rule Stage ---
        print("\n=== Stage 4: Executing Business Rule Stage ===")
        filtered = inspector.run_business_rule_stage()
        print(f"Result -> Filtered to {len(filtered)} candidates:")
        for candidate in filtered:
            meta = candidate.metadata or {}
            print(f"  ID: {candidate.volunteer_id} | Matched Skills: {meta.get('matched_skills')} | Distance: {meta.get('distance_km')}km")
            
        # --- 5. Scoring Stage ---
        print("\n=== Stage 5: Executing Scoring Stage ===")
        scored = inspector.run_scoring_stage()
        print(f"Result -> Scored Candidates:")
        for candidate in scored:
            print(f"  ID: {candidate.volunteer_id} | Sim: {candidate.similarity_score:.3f} | Business: {candidate.business_score:.3f} | Final: {candidate.final_score:.3f}")
            
        # --- 6. LLM Explanation Stage ---
        print("\n=== Stage 6: Executing LLM Explanation Stage (Groq / Heuristic) ===")
        explanations = inspector.run_llm_explanation_stage()
        for vid, reason in explanations.items():
            print(f"  Volunteer {vid} Reason -> \"{reason}\"")

        # --- 7. Re-ranking Stage ---
        print("\n=== Stage 7: Executing Re-ranking Stage ===")
        final_recs = inspector.run_reranking_stage()
        for rec in final_recs:
            print(f"  Top Rec -> ID: {rec.volunteer_id} | Score: {rec.final_score:.3f} | Reason: {rec.reason}")

        # --- 8. Full recommend_from_payload execution ---
        print("\n=== End-to-End Payload Output for Backend/Frontend ===")
        final_output = pipeline.recommend_from_payload(fetch_data["campaign"], fetch_data["volunteers"])
        print(f"Generated {len(final_output)} final volunteer cards:")
        for card in final_output:
            print(f"  * {card['name']} -> {card['matchPercent']}% Match | {card['whyRecommended']} | Breakdown: {card['breakdown']}")

    except Exception as e:
        print(f"\n[CRASH] The pipeline runner caught a critical failure during execution!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("\n=== Stack Trace ===")
        traceback.print_exc()



if __name__ == "__main__":
    main()
