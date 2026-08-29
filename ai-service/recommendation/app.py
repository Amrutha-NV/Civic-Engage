import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 stdout & stderr on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables from CivicEngage-final/.env (root) first
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # local fallback

from flask import Flask, request, jsonify


from recommender import SemanticRecommender
from explainer import RecommendationExplainer


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# AI COMPONENTS
# ============================================================

# Semantic recommendation engine
recommender = SemanticRecommender()

# Natural-language recommendation explanation
explainer = RecommendationExplainer()


# ============================================================
# HOME / STATUS ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "User Event AI Recommender Service"
    })


# ============================================================
# RECOMMENDATION ENDPOINT
# ============================================================

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # ----------------------------------------------------
        # Read JSON request
        # ----------------------------------------------------
        data = request.get_json(force=True) or {}
        user_info = data.get("user", {})
        events_list = data.get("events", [])

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------
        if not user_info:
            return jsonify({
                "success": False,
                "error": "User profile is required"
            }), 400

        if not isinstance(events_list, list):
            return jsonify({
                "success": False,
                "error": "Events must be provided as a list"
            }), 400

        if not events_list:
            return jsonify({
                "success": True,
                "recommendations": []
            })

        # ----------------------------------------------------
        # Generate semantic recommendations
        # ----------------------------------------------------
        recommendations = recommender.recommend(
            user_info,
            events_list
        )

        # ----------------------------------------------------
        # Generate explanations
        # ----------------------------------------------------
        for rec in recommendations:
            try:
                explanation = explainer.explain(
                    rec,
                    user_info
                )
                rec["explanation"] = explanation
            except Exception:
                rec["explanation"] = rec.get(
                    "reason",
                    "This campaign matches your profile."
                )

        # ----------------------------------------------------
        # Format compact recommendations: ID + Score + Explanation
        # ----------------------------------------------------
        compact_recommendations = [
            {
                "id": str(rec.get("id") or rec.get("_id") or ""),
                "matchScore": rec.get("matchScore", 0.0),
                "matchedSkills": rec.get("matchedSkills", []),
                "explanation": rec.get("explanation", rec.get("reason", "")),
                "reason": rec.get("reason", "This campaign matches your profile."),
                "breakdown": rec.get("breakdown", {}),
                "distanceKm": rec.get("distanceKm"),
            }
            for rec in recommendations
        ]

        return jsonify({
            "success": True,
            "recommendations": compact_recommendations
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}"
        }), 500




# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print(
        "🚀 Starting User Event Recommender "
        "AI service on port 5001..."
    )

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )
