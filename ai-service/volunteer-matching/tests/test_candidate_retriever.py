import unittest

from candidate_generation_service.retriever import CandidateRetriever


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def query_similar_volunteers(self, query_embedding, top_n):
        del query_embedding  # ignore content for stub
        return self.results[:top_n]


class CandidateRetrieverTests(unittest.TestCase):
    def test_retrieve_returns_sorted_candidates_by_similarity(self):
        results = [
            {"volunteer_id": "vol1", "text": "Volunteer 1", "similarity_score": 0.85, "distance": 0.1},
            {"volunteer_id": "vol2", "text": "Volunteer 2", "similarity_score": 0.65, "distance": 0.4},
            {"volunteer_id": "vol3", "text": "Volunteer 3", "similarity_score": 0.95, "distance": 0.05},
        ]
        vector_store = FakeVectorStore(results)
        retriever = CandidateRetriever(vector_store)

        campaign_embedding = type("CampaignEmbedding", (), {"embedding": [0.0]})()
        candidates = retriever.retrieve(campaign_embedding, top_n=3)

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0].volunteer_id, "vol3")
        self.assertEqual(candidates[1].volunteer_id, "vol1")
        self.assertEqual(candidates[2].volunteer_id, "vol2")
        self.assertAlmostEqual(candidates[0].similarity_score, 0.95)
        self.assertAlmostEqual(candidates[1].similarity_score, 0.85)
        self.assertAlmostEqual(candidates[2].similarity_score, 0.65)


if __name__ == "__main__":
    unittest.main()

