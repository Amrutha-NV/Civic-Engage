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

        for recommendation in recommendations:

            try:
                explanation = explainer.explain(
                    recommendation,
                    user_info
                )

                recommendation["explanation"] = explanation

            except Exception:
                # If explanation fails, keep recommendation
                # available with its existing reason.
                recommendation["explanation"] = (
                    recommendation.get(
                        "reason",
                        "This campaign matches your profile."
                    )
                )

        # ----------------------------------------------------
        # Return final response
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "recommendations": recommendations
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
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
        debug=True
    )