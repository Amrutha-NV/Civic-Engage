import unittest

from config.settings import Settings
from pipeline import RecommendationPipeline


class PipelineTests(unittest.TestCase):
    def test_recommendation_pipeline_with_dummy_data(self):
        settings = Settings()
        pipeline = RecommendationPipeline(settings)
        recommendations = pipeline.recommend("camp1", use_dummy_data=True)

        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), settings.top_k_recommendations)
        self.assertGreater(len(recommendations), 0)
        for recommendation in recommendations:
            self.assertIn("id", recommendation)
            self.assertIn("name", recommendation)
            self.assertIn("matchPercent", recommendation)
            self.assertIn("whyRecommended", recommendation)
            self.assertIn("breakdown", recommendation)
            self.assertIn("skills", recommendation["breakdown"])
            self.assertIn("matchedSkills", recommendation)
            self.assertGreaterEqual(recommendation["matchPercent"], 65)
            self.assertLessEqual(recommendation["matchPercent"], 99)


if __name__ == "__main__":
    unittest.main()

