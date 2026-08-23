import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_events(user_data, events_data):
    """
    Computes recommendation scores between user profile (skills, interests, location)
    and candidate events (title, category, requiredSkills, location) using TF-IDF text similarity
    and skill overlap matching.
    """
    if not events_data:
        return []

    # Construct user text query
    user_skills = " ".join(user_data.get("skills", []))
    user_interests = " ".join(user_data.get("interests", []))
    user_loc = user_data.get("location", "")
    user_text = f"{user_skills} {user_interests} {user_loc}".strip()

    # Construct event texts
    event_texts = []
    for evt in events_data:
        req_skills = " ".join(evt.get("requiredSkills", []))
        cat = evt.get("category", "")
        title = evt.get("title", "")
        loc = evt.get("location", "")
        text = f"{title} {cat} {req_skills} {loc}".strip()
        event_texts.append(text)

    # Vectorize using TF-IDF
    corpus = [user_text] + event_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    user_vec = tfidf_matrix[0:1]
    events_matrix = tfidf_matrix[1:]

    similarities = cosine_similarity(user_vec, events_matrix).flatten()

    results = []
    for i, event in enumerate(events_data):
        raw_sim = float(similarities[i])
        
        # Skill overlap boost
        user_skills_set = set(s.lower() for s in user_data.get("skills", []))
        req_skills_set = set(s.lower() for s in event.get("requiredSkills", []))
        matched = user_skills_set.intersection(req_skills_set)

        skill_boost = len(matched) * 0.15
        final_score = int(min(98, max(65, (raw_sim + skill_boost) * 100)))

        reason = f"Matches your skills in {', '.join(matched) if matched else 'community service'} and event focus."
        
        event_copy = dict(event)
        event_copy["matchScore"] = final_score
        event_copy["reason"] = reason
        results.append(event_copy)

    # Sort descending by match score
    results.sort(key=lambda x: x["matchScore"], reverse=True)
    return results