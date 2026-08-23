# Volunteer Matching Service

A volunteer recommendation engine built as a retrieval + ranking pipeline. This service provides intelligent matching between volunteers and civic campaigns using semantic search, business rules, and configurable scoring.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Backend Integration](#backend-integration)
- [Frontend Integration](#frontend-integration)
- [Features](#features)
- [API Endpoints](#api-endpoints)
- [Local Setup](#local-setup)
- [Testing](#testing)

## Overview

The Volunteer Matching Service is a dedicated microservice responsible for:
- Analyzing campaign requirements and volunteer profiles
- Generating semantic embeddings for intelligent matching
- Retrieving candidate volunteers using similarity search
- Applying business rules (availability, verification, campaign load limits)
- Scoring and ranking candidates
- Returning ranked recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                        │
│              (Dashboard, Campaign, Volunteer UI)            │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    ┌────────────────────────────────────────────┐
    │         Backend (Node.js/Express)          │
    │     API Gateway & Business Logic Layer     │
    └─────────────────┬──────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
        ▼                            ▼
    [Fetch API]             [Matching Service API]
    (User/Campaign          (Recommendations)
     Database)
```

## Backend Integration

### Requirements from Backend

The backend service **MUST** provide the following to support Volunteer Matching:

#### 1. Data Availability Endpoints
The backend must expose REST APIs to fetch raw data:

**Get Campaign by ID**
```
GET /api/campaigns/:id
Response: {
  "_id": "string",
  "title": "string",
  "description": "string",
  "category": "string" (environmental|social|health|education|disaster)
  "requirements": ["string"],
  "location": "string",
  "urgency": "string" (low|medium|high),
  "meta": {
    "impact_score": number,
    "participation_goal": number
  }
}
```

**Get All Volunteers**
```
GET /api/volunteers
Response: [{
  "_id": "string",
  "name": "string",
  "email": "string",
  "location": "string",
  "skills": ["string"],
  "availability": ["string"],
  "attendance_rate": number (0-1),
  "social_impact_score": number (0-1),
  "verified": boolean,
  "active_campaigns": number,
  "history_ids": ["string"]
}]
```

#### 2. Volunteer Metadata Consistency
Backend must ensure:
- Volunteer records have complete and accurate skill tags
- Location information is standardized (city/region format)
- Availability is always an array of days/times
- Social impact scores and attendance rates are normalized (0-1)
- Verified status is maintained accurately

#### 3. Campaign-Volunteer Relationship Management
When recommendations are used by frontend/backend:
- Backend must track which volunteers are recommended for which campaigns
- Backend must store participation history (`history_ids`)
- Backend must track active campaign count per volunteer
- Backend must support updating `active_campaigns` count when volunteer joins

#### 4. Embedding Sync Capability
Backend can trigger embedding updates via:
```
POST /embedding/sync
Content-Type: application/json

{
  "_id": "vol123",
  "name": "Alice Johnson",
  "email": "alice@example.com",
  "location": "Springfield",
  "skills": ["gardening", "community organizing"],
  "availability": ["weekends", "evenings"],
  "attendance_rate": 0.92,
  "social_impact_score": 0.82,
  "verified": true,
  "active_campaigns": 1,
  "history_ids": ["camp1"]
}
```

### Backend Integration Workflow

1. **Request Recommendations**
   - Frontend/Backend calls: `GET /recommend/:campaign_id`
   - Matching service retrieves campaign from backend
   - Matching service retrieves all volunteers from backend
   - Returns ranked list of recommendations

2. **Update Volunteer Profile**
   - When volunteer joins a campaign or updates profile
   - Backend should call: `POST /embedding/sync` to update vector store
   - This ensures future recommendations reflect latest data

3. **Campaign-Level Triggers**
   - When campaign is created/updated
   - Backend should trigger recommendation refresh if needed

## Frontend Integration

### Requirements from Frontend

#### 1. Connection Flow

**Step 1: Display Available Campaigns**
- Frontend fetches campaigns from backend API
- Shows campaign details to campaign managers/admins

**Step 2: Request Recommendations**
```javascript
// Call the matching service (via backend proxy or directly)
const response = await fetch(
  'http://localhost:8000/recommend/camp123?use_dummy=false'
);
const recommendations = await response.json();
// Returns: { candidates: [...], total: number }
```

**Step 3: Display Recommendations**
- Frontend receives ranked list of volunteer recommendations
- Shows volunteer profiles with match scores
- Displays why each volunteer was recommended

**Step 4: Handle Volunteer Selection**
- When campaign manager selects a volunteer
- Frontend notifies backend to create participation record
- Backend calls `/embedding/sync` to update service

#### 2. Frontend UI Requirements

The frontend **MUST** display the following to users:

**Recommendation Card Component**
```
┌─────────────────────────────────────┐
│ Volunteer Name                      │
├─────────────────────────────────────┤
│ Match Score: 92%                    │
│ Skills Match: ✓ gardening, ✓ events│
│ Availability: Weekends              │
│ Attendance Rate: 92%                │
│ Impact Score: 82%                   │
│ Verification: Verified ✓            │
├─────────────────────────────────────┤
│ [Select Volunteer] [View Profile]   │
└─────────────────────────────────────┘
```

**Integration Points**
```javascript
// 1. Fetch recommendations
async function getRecommendations(campaignId) {
  const response = await fetch(
    `/api/recommend/${campaignId}?use_dummy=false`
  );
  return response.json();
}

// 2. Display similarity score (0-1, show as percentage)
function displayScore(similarityScore) {
  return Math.round(similarityScore * 100) + '%';
}

// 3. Handle volunteer selection
async function selectVolunteer(volunteerId, campaignId) {
  // 1. Update backend participation record
  await fetch('/api/participations', {
    method: 'POST',
    body: JSON.stringify({ volunteerId, campaignId })
  });
  
  // 2. Trigger embedding sync (backend should do this)
  // This ensures future recommendations account for new campaign
}

// 4. Display match reasoning
function showReason(candidate) {
  const { similarity_score, business_score, final_score, reason } = candidate;
  return `
    Similarity: ${similarity_score.toFixed(2)}
    Business Score: ${business_score.toFixed(2)}
    Final Score: ${final_score.toFixed(2)}
    Reason: ${reason}
  `;
}
```

#### 3. Environment Configuration

Frontend should configure the matching service URL:
```javascript
// .env or config file
REACT_APP_MATCHING_SERVICE_URL=http://localhost:8000
// or through backend proxy
REACT_APP_RECOMMEND_ENDPOINT=/api/recommend
```

#### 4. Data Constraints

Frontend must be aware:
- Recommendations return up to `top_n_candidates` (default: 50, configurable)
- Scores range from 0.0 to 1.0 (display as 0-100%)
- `metadata` field may be empty (reserved for future use)
- `reason` field contains human-readable score breakdown
- Results are sorted by `final_score` (highest first)

## Features

- Pydantic models for strong typing and validation
- Fetching service for campaigns and volunteers
- Embedding generation with sentence transformers
- Vector store for semantic retrieval
- Candidate generation via cosine similarity
- Business rule filtering for availability, verification, and campaign load
- Configurable scoring and reranking
- FastAPI endpoints for recommendations and embedding sync

## API Endpoints

### Get Recommendations
```
GET /recommend/:campaign_id
Query Parameters:
  - use_dummy: boolean (default: false) - Use dummy data for testing
  
Response: {
  "candidates": [
    {
      "volunteer_id": "string",
      "similarity_score": number,
      "business_score": number,
      "final_score": number,
      "reason": "string",
      "metadata": {}
    }
  ],
  "total": number
}
```

### Sync Volunteer Embedding
```
POST /embedding/sync
Content-Type: application/json

Payload: Volunteer object with all fields
Response: { "status": "success", "volunteer_id": "string" }
```

### Health Check
```
GET /health
Response: { "status": "ok" }
```

## Local Setup

## Local Setup

### Prerequisites
- Python 3.9+
- pip or conda
- Backend service running (for real data mode)

### Installation

1. Create and activate a Python virtual environment

```powershell
cd c:\Users\User\CivicEngage\volunteer-matching-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure settings (optional)

Edit `config/settings.py` to adjust:
- `top_n_candidates`: Number of recommendations to return (default: 50)
- `top_k_recommendations`: Final recommendations after reranking (default: 10)
- `backend_url`: Backend service URL (default: http://localhost:3000)

3. Run the service

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The service will be available at `http://127.0.0.1:8000`

### Testing with Dummy Data

Request recommendations using dummy data:
```powershell
curl "http://127.0.0.1:8000/recommend/camp1?use_dummy=true"
```

### Testing with Real Data

Ensure backend is running, then:
```powershell
curl "http://127.0.0.1:8000/recommend/camp1?use_dummy=false"
```

### Syncing Volunteer Embeddings

When a volunteer profile is updated in the backend:
```powershell
curl -X POST "http://127.0.0.1:8000/embedding/sync" `
  -H "Content-Type: application/json" `
  -d '{
    "_id":"vol1",
    "name":"Alice",
    "email":"alice@example.com",
    "location":"Springfield",
    "skills":["gardening","community organizing"],
    "availability":["weekends","evenings"],
    "attendance_rate":0.92,
    "social_impact_score":0.82,
    "verified":true,
    "active_campaigns":1,
    "history_ids":["camp1"]
  }'
```

## Testing

Run all tests:
```powershell
python -m pytest tests/ -v
```

Run specific test file:
```powershell
python -m pytest tests/test_candidate_retriever.py -v
```

Run with coverage:
```powershell
python -m pytest tests/ --cov=. --cov-report=html
```

## Deployment

### Docker (Optional)

Build image:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```
BACKEND_URL=http://backend:3000
SERVICE_PORT=8000
LOG_LEVEL=INFO
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_N_CANDIDATES=50
TOP_K_RECOMMENDATIONS=10
```

## Troubleshooting

### KeyError: 'metadata'
- Ensure you're using the latest vector store implementation
- Vector store no longer stores metadata; pass empty dict `{}` instead

### Embedding Generation Slow
- First embedding generation caches the model (one-time cost)
- Subsequent calls are much faster
- Adjust model in `config/settings.py` for speed vs. accuracy tradeoff

### Backend Connection Failed
- Check backend is running on configured URL
- Verify volunteer and campaign endpoints return correct data format
- Use `use_dummy=true` to test matching service independently

## Integration Checklist

**Backend Team:**
- [ ] Expose `/api/campaigns/:id` endpoint
- [ ] Expose `/api/volunteers` endpoint
- [ ] Implement `/embedding/sync` handler (receive notifications)
- [ ] Maintain accurate volunteer metadata (skills, location, availability)
- [ ] Track `active_campaigns` and `attendance_rate` per volunteer
- [ ] Store recommendations in database for audit/analytics

**Frontend Team:**
- [ ] Create recommendations UI component
- [ ] Call `/recommend/:campaignId` endpoint
- [ ] Display `final_score` as percentage match
- [ ] Show volunteer skills and availability
- [ ] Implement "Select Volunteer" action
- [ ] Connect to backend to record participation
- [ ] Cache recommendation results if needed

**Matching Service Team:**
- [ ] Monitor embedding quality and performance
- [ ] Track recommendation accuracy metrics
- [ ] Maintain business rule configurations
- [ ] Handle edge cases (no volunteers, new campaigns, etc.)

## Project Structure

```
volunteer-matching-service/
├── main.py                           # FastAPI application entry point
├── pipeline.py                        # Main recommendation pipeline
├── requirements.txt                   # Python dependencies
├── config/
│   └── settings.py                   # Configuration and constants
├── models/                            # Pydantic data models
│   ├── campaign.py
│   ├── volunteer.py
│   ├── embedding.py
│   ├── candidate.py
│   ├── recommendation.py
│   └── ...
├── fetching_service/                 # Data retrieval from backend
│   ├── campaign_fetcher.py
│   └── volunteer_fetcher.py
├── embedding_service/                # Embedding generation & storage
│   ├── embedding_generator.py
│   ├── text_builder.py
│   └── vector_store.py
├── candidate_generation_service/     # Semantic search retrieval
│   └── retriever.py
├── business_rule_service/            # Business rule application
│   ├── rule_engine.py
│   ├── availability_rule.py
│   ├── verification_rule.py
│   └── max_campaign_rule.py
├── scoring_service/                  # Candidate scoring & ranking
│   ├── scorer.py
│   └── weight_config.py
├── reranking_service/                # Final ranking
│   └── reranker.py
├── tests/                            # Unit tests
│   ├── test_candidate_retriever.py
│   ├── test_pipeline.py
│   └── ...
└── utils/
    └── pipeline_context.py           # Shared context between stages
```

## Contributing

1. Follow PEP 8 style guide
2. Add tests for new features
3. Update README for API changes
4. Ensure all tests pass before submitting

## License

TBD
