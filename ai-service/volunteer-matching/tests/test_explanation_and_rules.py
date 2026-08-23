import unittest
from config.settings import Settings
from models.campaign import Campaign
from models.volunteer import Volunteer
from models.candidate import Candidate
from business_rule_service.skill_match_rule import SkillMatchRule
from explanation_service.llm_explainer import LLMExplanationService


class ExplanationAndRuleTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.campaign = Campaign(
            _id="camp_101",
            title="Coastal Marine Cleanup",
            category="environmental",
            description="Clean plastic waste from the coastal shorelines.",
            location="Manila Bay",
            requiredSkills=["waste management", "first aid", "teamwork"],
        )
        self.volunteers = [
            Volunteer(
                _id="vol_1",
                name="Elena Cruz",
                skills=["waste management", "first aid", "scuba diving"],
                interests=["Marine Conservation", "Environment"],
                location="Manila",
                verified=True,
                impactScore=94,
            ),
            Volunteer(
                _id="vol_2",
                name="Marco Santos",
                skills=["public speaking", "event planning"],
                interests=["Community Development"],
                location="Quezon City",
                verified=False,
                impactScore=80,
            ),
        ]
        self.candidates = [
            Candidate(volunteer_id="vol_1", similarity_score=0.88, metadata={"distance_km": 3.2}),
            Candidate(volunteer_id="vol_2", similarity_score=0.62, metadata={"distance_km": 15.0}),
        ]

    def test_skill_match_rule(self):
        rule = SkillMatchRule()
        filtered = rule.apply(self.campaign, self.candidates, self.volunteers)
        
        self.assertEqual(len(filtered), 2)
        vol1_meta = filtered[0].metadata
        self.assertIn("waste management", vol1_meta["matched_skills"])
        self.assertIn("first aid", vol1_meta["matched_skills"])
        self.assertAlmostEqual(vol1_meta["skill_overlap_ratio"], 2 / 3, places=2)

    def test_llm_explanation_fallback_generation(self):
        explainer = LLMExplanationService(self.settings)
        # Apply skill match metadata first
        rule = SkillMatchRule()
        augmented_candidates = rule.apply(self.campaign, self.candidates, self.volunteers)
        
        reasons = explainer.generate_explanations(self.campaign, augmented_candidates, self.volunteers)
        
        self.assertEqual(len(reasons), 2)
        self.assertIn("vol_1", reasons)
        self.assertIn("vol_2", reasons)
        self.assertIsInstance(reasons["vol_1"], str)
        self.assertGreater(len(reasons["vol_1"]), 20)
        self.assertGreater(len(reasons["vol_2"]), 20)



if __name__ == "__main__":
    unittest.main()
