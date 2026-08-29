# 🏛️ CivicEngage — AI-Powered Civic Engagement & NGO Platform

[![Node.js](https://img.shields.io/badge/Node.js-v18+-68a063.svg?logo=node.js)](https://nodejs.org)
[![React](https://img.shields.io/badge/React-v18-61dafb.svg?logo=react)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg?logo=flask)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-SentenceTransformers-ee4c2c.svg?logo=pytorch)](https://pytorch.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47a248.svg?logo=mongodb)](https://www.mongodb.com)

**CivicEngage** is a unified platform connecting volunteers with community-driven NGO campaigns through AI-powered semantic matching, proximity geofencing, skill affinity analysis, and automated natural-language justifications.

---

## 🏗️ System Architecture & Service Topology

```
                                  ┌────────────────────────┐
                                  │   React + Vite App     │
                                  │ (http://localhost:5173)│
                                  └───────────┬────────────┘
                                              │ HTTP / REST
                                              ▼
                                  ┌────────────────────────┐
                                  │   Express Backend API  │
                                  │ (http://localhost:5000)│
                                  └─────┬────────────┬─────┘
                                        │            │
            ┌───────────────────────────┘            └───────────────────────────┐
            │ POST /recommend                                                    │ POST /recommend-volunteers
            ▼                                                                    ▼
┌───────────────────────────────┐                                ┌───────────────────────────────┐
│ AI Event Recommender (Flask)  │                                │ AI Volunteer Matcher (FastAPI)│
│    (http://127.0.0.1:5001)    │                                │    (http://127.0.0.1:8000)    │
├───────────────────────────────┤                                ├───────────────────────────────┤
│ • all-MiniLM-L6-v2 Embeddings │                                │ • MPNet Semantic Retrieval    │
│ • Haversine Proximity (Dict)  │                                │ • Location Geofencing         │
│ • Multi-Factor Scoring        │                                │ • Attendance & Impact Weights │
│ • Groq LLM Explanations       │                                │ • Dynamic Explainer Engine    │
└───────────────────────────────┘                                └───────────────────────────────┘
```

| Service | Technology | Port / URL | Description |
|---|---|---|---|
| **Frontend** | React 18, Vite, TailwindCSS | `http://localhost:5173` | Volunteer & NGO Interactive Dashboards |
| **Backend API** | Node.js, Express, Mongoose | `http://localhost:5000` | REST API, Auth, Database Orchestration |
| **AI Event Recommender** | Python, Flask, SentenceTransformers | `http://127.0.0.1:5001` | Semantic Event Recommendation Engine |
| **AI Volunteer Matcher** | Python, FastAPI, Uvicorn, MPNet | `http://127.0.0.1:8000` | Multi-Stage Volunteer Matching Pipeline |
| **Database** | MongoDB Atlas | Cloud Cluster | Users, NGOs, Campaigns, Participation |

---

## 📁 Repository Structure

```
CivicEngage-final/
├── .env                              # Centralized environment configuration (Single Source of Truth)
├── .env.example                      # Template environment variables
├── package.json                      # Monorepo runner scripts
├── README.md                         # Project documentation
│
├── frontend/                         # React + Vite Frontend Application
│   ├── src/
│   │   ├── components/               # UI components (Navbar, Modals, Cards)
│   │   ├── pages/                    # User Dashboard, NGO Portal, Landing Page
│   │   ├── services/                 # Axios API clients
│   │   └── context/                  # AuthContext and state providers
│   ├── vite.config.js                # Vite config (configured to load root .env)
│   └── package.json
│
├── backend/                          # Express.js Backend REST API
│   ├── config/                       # Database (db.js) and Cloudinary (cloudinary.js)
│   ├── controllers/                  # Auth, User, Campaign, NGO, AI Controllers
│   ├── middleware/                   # JWT Authentication & Role Verification
│   ├── models/                       # Mongoose Schemas (User, NGO, Campaign)
│   ├── routes/                       # Express Route Definitions
│   ├── utils/                        # Coordinate Resolution & AI Sync helpers
│   ├── seed.js                       # Database Seeding Script
│   ├── server.js                     # Express entry point (loads root .env)
│   └── package.json
│
└── ai-service/                       # AI & Machine Learning Microservices
    ├── recommendation/               # Semantic Event Recommendation Microservice (Flask)
    │   ├── app.py                    # Flask API Server (Port 5001)
    │   ├── recommender.py            # Multi-factor Semantic Scoring Engine
    │   ├── explainer.py              # Groq LLM Explanation Generator
    │   ├── models.py                 # Pydantic data schemas
    │   └── venv/                     # Python Virtual Environment
    │
    └── volunteer-matching/           # Volunteer-to-Campaign Matching Microservice (FastAPI)
        ├── main.py                   # FastAPI Application (Port 8000)
        ├── pipeline.py               # Candidate retrieval & scoring pipeline
        ├── config/settings.py        # Settings loader (reads root .env)
        ├── explanation_service/      # AI Justification Service
        └── requirements.txt          # Python dependencies
```

---

## ⚙️ Unified Environment Configuration (`.env`)

All services are configured to fetch their configuration directly from the **`.env` file in the root folder** (`CivicEngage-final/.env`).

Create or edit `.env` in the root directory:

```env
# Server Ports & Database
PORT=5000
MONGO_URI="mongodb+srv://<username>:<password>@cluster0.tnrecqb.mongodb.net/?appName=Cluster0"
JWT_SECRET=mysecretkey123

# Cloudinary Media Storage
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Frontend API URL
VITE_API_URL=http://127.0.0.1:5000/api

# Backend Base URL (Used by AI Services)
BACKEND_BASE_URL=http://127.0.0.1:5000

# AI Microservices URLs (Used by Backend)
AI_RECOMMENDATION_SERVICE_URL=http://127.0.0.1:5001
AI_VOLUNTEER_MATCHING_SERVICE_URL=http://127.0.0.1:8000

# LLM Explanation Service (Groq)
GROQ_API_KEY="your_groq_api_key_here"
GROQ_MODEL="llama-3.3-70b-versatile"
LLM_PROVIDER=groq
```

---

## 🚀 Step-by-Step Execution Guide

Open **4 separate terminal windows** (or tabs) to start all components:

### Terminal 1: Backend Server (Node.js / Express)
```powershell
# Navigate to backend
cd backend

# Install dependencies (first time only)
npm install

# (Optional) Seed sample NGOs, Volunteers, and Campaigns into MongoDB
node seed.js

# Start backend server
npm run dev
```
* Backend runs at **`http://127.0.0.1:5000`**

---

### Terminal 2: Frontend Application (React / Vite)
```powershell
# Navigate to frontend
cd frontend

# Install dependencies (first time only)
npm install

# Start Vite development server
npm run dev
```
* Access the web UI at **`http://localhost:5173`**

---

### Terminal 3: AI Event Recommender Service (Flask)
```powershell
# Navigate to recommendation service
cd ai-service/recommendation

# Run using the dedicated virtual environment
.\venv\Scripts\python.exe app.py
```
* Service runs at **`http://127.0.0.1:5001`**
* Endpoints: `POST /recommend`, `GET /`

---

### Terminal 4: AI Volunteer Matching Service (FastAPI)
```powershell
# Navigate to volunteer matching service
cd ai-service/volunteer-matching

# (Optional) Install dependencies if setting up a new environment
# pip install -r requirements.txt

# Start FastAPI server with Uvicorn
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
* Service runs at **`http://127.0.0.1:8000`**
* Interactive Swagger Docs: **`http://127.0.0.1:8000/docs`**

---

## 🧠 AI Recommendation Pipeline Breakdown

When a user opens their dashboard or an NGO views event applicants:

1. **Semantic Text Generation**: Profiles and campaign details are encoded into dense 384-dimensional embeddings using `all-MiniLM-L6-v2`.
2. **Multi-Factor Weighted Scoring**:
   $$\text{Final Score} = (0.40 \times \text{Semantic}) + (0.30 \times \text{Skills}) + (0.15 \times \text{Interests}) + (0.10 \times \text{Proximity}) + (0.05 \times \text{Availability})$$
3. **Proximity Decay**: Calculated via spherical Haversine formula with continuous exponential decay ($e^{-d / 15}$). Supports both coordinate dictionaries (`{'lat': ..., 'lng': ...}`) and coordinate arrays.
4. **Natural Language Explanation**: Groq's high-speed inference engine generates concise, human-readable match justifications under 40 words, with an automatic deterministic rule fallback.
5. **Backend Document Enrichment**: The AI service returns scored event IDs, and Express joins the full MongoDB campaign documents with their score badges.

---

## 🛠️ Key API Endpoints

### Authentication & Profiles
* `POST /api/auth/user/send-otp` — Send OTP for volunteer phone authentication
* `POST /api/auth/user/verify-otp` — Verify OTP and issue JWT session
* `POST /api/auth/ngo/register` — Register a verified NGO profile
* `POST /api/auth/ngo/login` — NGO login with email & password

### Campaigns & Events
* `GET /api/events` — Retrieve all active campaigns with search and filtering
* `POST /api/campaigns` — Create a new campaign (NGO authorization required)
* `POST /api/events/:id/participate` — Register a volunteer for a campaign

### AI Microservices
* `POST /api/recommendations/events` — Fetch AI ranked campaign recommendations for a volunteer
* `POST /api/recommendations/volunteers` — Fetch AI ranked volunteer candidates for an NGO campaign
* `POST /api/tender/generate` — Generate structured RFQ/Tender documentation for NGO procurement

---

## 🛡️ Troubleshooting & FAQ

<details>
<summary><b>1. Python UnicodeEncodeError on Windows Console</b></summary>
All Python services in this project automatically reconfigure `sys.stdout` to UTF-8 on Windows. If running standalone scripts, ensure your PowerShell session supports UTF-8:
<code>[Console]::OutputEncoding = [System.Text.Encoding]::UTF8</code>
</details>

<details>
<summary><b>2. ModuleNotFoundError: No module named 'sentence_transformers'</b></summary>
Always execute Python services using the virtual environment Python interpreter:
<code>.\venv\Scripts\python.exe app.py</code> (or activate via <code>.\venv\Scripts\activate</code>).
</details>

<details>
<summary><b>3. AI Service Fallback Mode</b></summary>
If the AI services (ports 5001 or 8000) are offline, the Express backend automatically activates a deterministic fallback scoring algorithm without breaking the frontend experience.
</details>


