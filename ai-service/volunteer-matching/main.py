from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

from config.settings import Settings
from models.volunteer import Volunteer
from models.campaign import Campaign
from pipeline import RecommendationPipeline

app = FastAPI(title="CivicEngage AI Volunteer Matching Service")
settings = Settings()
service = RecommendationPipeline(settings)


class VolunteerMatchRequest(BaseModel):
    event: Dict[str, Any]
    volunteers: List[Dict[str, Any]]


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "CivicEngage AI Volunteer Matching Service",
        "version": "1.0.0"
    }


@app.post("/recommend-volunteers")
def recommend_volunteers(request_data: VolunteerMatchRequest):
    """
    Accepts event details and candidate volunteers, delegates to RecommendationPipeline
    for multi-stage semantic retrieval, location geofencing, business rule filtering, 
    multi-criteria scoring, and reranking.
    """
    try:
        event_dict = request_data.event
        volunteers_list = request_data.volunteers

        # Ensure id field mapping for Pydantic validation
        if "id" not in event_dict and "_id" in event_dict:
            event_dict["id"] = event_dict["_id"]
        elif "id" not in event_dict:
            event_dict["id"] = "e_unknown"

        campaign_obj = Campaign(**event_dict)

        volunteer_objs: List[Volunteer] = []
        for v in volunteers_list:
            if "id" not in v and "_id" in v:
                v["id"] = v["_id"]
            elif "id" not in v:
                v["id"] = "v_unknown"
            volunteer_objs.append(Volunteer(**v))

        # Execute full recommendation pipeline
        ranked_volunteers = service.recommend_from_payload(campaign_obj, volunteer_objs)

        return {
            "success": True,
            "count": len(ranked_volunteers),
            "volunteers": ranked_volunteers
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/embedding/sync")
def sync_embedding(request_data: Dict[str, Any]):
    """
    Syncs individual volunteer embedding when a user registers or updates their profile.
    """
    try:
        data = dict(request_data)
        if "id" not in data and "_id" in data:
            data["id"] = str(data["_id"])
        elif "_id" not in data and "id" in data:
            data["_id"] = str(data["id"])

        volunteer = Volunteer(**data)
        embedding = service.embedding_generator.generate_volunteer_embedding(volunteer)
        service.vector_store.upsert_volunteer_embedding(embedding)
        
        print(f"[AI SERVICE - SYNC] Volunteer embedding synced: ID={volunteer.id} | Name={volunteer.name} | Skills={volunteer.skills} | Location={volunteer.location}")

        return {
            "success": True,
            "detail": "Volunteer embedding synced successfully",
            "volunteer_id": str(volunteer.id)
        }
    except Exception as exc:
        print(f"[AI SERVICE - SYNC ERROR] Failed to sync volunteer: {exc}")
        return {
            "success": False,
            "detail": f"Failed to sync embedding: {str(exc)}",
            "volunteer_id": str(request_data.get("id") or request_data.get("_id") or "unknown")
        }


