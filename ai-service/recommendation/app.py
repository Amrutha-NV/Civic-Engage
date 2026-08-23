from flask import Flask, request, jsonify
from recommender import recommend_events

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status": "online", "service": "User Event AI Recommender Service"})

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json(force=True) or {}
        user_info = data.get("user", {})
        events_list = data.get("events", [])
        
        recommendations = recommend_events(user_info, events_list)
        return jsonify({"success": True, "recommendations": recommendations})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting User Event Recommender AI service on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=True)