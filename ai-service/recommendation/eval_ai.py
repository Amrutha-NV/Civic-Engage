"""
CivicEngage AI Recommendation & Volunteer Matching Evaluation System
Computes Precision@K, Recall@K, F1-Score, and Mean Reciprocal Rank (MRR)
on ground-truth dataset.
"""

import numpy as np

# Ground-truth test dataset: (User Skills & Interests -> Relevant Event IDs)
GROUND_TRUTH = [
    {
        "user_id": "u1",
        "user_profile": {
            "skills": ["Environmental Science", "Event Planning"],
            "interests": ["Environmental Protection"],
            "location": "Marikina"
        },
        "relevant_event_ids": ["e1", "e4"] # Watershed & Coastal Cleanup
    },
    {
        "user_id": "u2",
        "user_profile": {
            "skills": ["Teaching", "Mathematics"],
            "interests": ["Education"],
            "location": "Quezon City"
        },
        "relevant_event_ids": ["e2"] # Digital Literacy
    },
    {
        "user_id": "u3",
        "user_profile": {
            "skills": ["Healthcare", "First Aid"],
            "interests": ["Healthcare & Wellness"],
            "location": "Manila"
        },
        "relevant_event_ids": ["e3"] # First-Aid Health Fair
    }
]

CANDIDATE_EVENTS = [
    {"id": "e1", "title": "Marikina Watershed Reforestation", "category": "Environmental Protection", "requiredSkills": ["Environmental Science"]},
    {"id": "e2", "title": "Digital Literacy Workshop for Kids", "category": "Education & Literacy", "requiredSkills": ["Teaching"]},
    {"id": "e3", "title": "Community First-Aid Health Fair", "category": "Healthcare & Wellness", "requiredSkills": ["Healthcare", "First Aid"]},
    {"id": "e4", "title": "Manila Bay Coastal Clean-up", "category": "Environmental Protection", "requiredSkills": ["Environmental Science"]}
]

def mock_predict_recommendations(user_profile, events, top_k=2):
    # Match based on skill overlap
    user_skills = set(s.lower() for s in user_profile["skills"])
    scores = []
    for evt in events:
        req = set(s.lower() for s in evt["requiredSkills"])
        overlap = len(user_skills.intersection(req))
        scores.append((evt["id"], overlap))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [e_id for e_id, _ in scores[:top_k]]

def evaluate_recommender(k=2):
    precisions = []
    recalls = []
    f1_scores = []
    reciprocal_ranks = []

    print(f"=== Running AI Recommender Evaluation (Top-K={k}) ===")
    for item in GROUND_TRUTH:
        user_id = item["user_id"]
        relevant = set(item["relevant_event_ids"])
        predicted = mock_predict_recommendations(item["user_profile"], CANDIDATE_EVENTS, top_k=k)
        predicted_set = set(predicted)

        hits = len(relevant.intersection(predicted_set))
        precision = hits / k if k > 0 else 0
        recall = hits / len(relevant) if len(relevant) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Compute MRR
        rr = 0.0
        for rank, p_id in enumerate(predicted, 1):
            if p_id in relevant:
                rr = 1.0 / rank
                break

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        reciprocal_ranks.append(rr)

        print(f"User {user_id}: Precision@{k}={precision:.2f}, Recall@{k}={recall:.2f}, F1={f1:.2f}, RR={rr:.2f}")

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    mrr = np.mean(reciprocal_ranks)

    print("\n--- OVERALL METRICS ---")
    print(f"Mean Precision@{k}: {mean_precision * 100:.1f}%")
    print(f"Mean Recall@{k}:    {mean_recall * 100:.1f}%")
    print(f"Mean F1-Score:     {mean_f1 * 100:.1f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print("===============================")

if __name__ == "__main__":
    evaluate_recommender(k=2)
