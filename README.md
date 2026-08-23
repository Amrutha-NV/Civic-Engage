# 🏛️ CivicEngage – Civic Engagement & NGO Platform

CivicEngage is an integrated platform facilitating civic engagement, NGO campaign orchestration, user participation, and AI-driven volunteer matching & recommendations.

---

## 📁 Clean Directory Structure

```
CivicEngage-final/
├── frontend/                     # React + Vite frontend application (User & NGO Dashboards)
│   ├── src/
│   ├── public/
│   ├── index.html
│   └── package.json
│
├── backend/                      # Node.js + Express REST API Backend
│   ├── config/                   # Database connection (MongoDB)
│   ├── controllers/              # Auth, User, Campaign, NGO, AI, Upload controllers
│   ├── middleware/               # Auth verification & security middleware
│   ├── models/                   # Mongoose schemas (User, NGO, Campaign, Event, etc.)
│   ├── routes/                   # API routes
│   ├── seed.js                   # Database seed script
│   ├── server.js                 # Express server entry point
│   └── package.json
│
├── ai-service/                   # AI Recommendation & Matching Services
│   ├── recommendation/           # Flask campaign/event recommendation engine
│   │   └── app.py
│   └── volunteer-matching/       # FastAPI volunteer matching pipeline service
│       └── main.py
│
├── .env                          # Global environment configuration
├── package.json                  # Root monorepo runner scripts
└── README.md                     # Documentation & setup guide
```

---

## 🚀 Steps to Start the Project

### 1. Database Seeding (Optional)
To populate sample Users, NGOs, and Campaigns:
```bash
# From root directory:
npm run seed:backend

# Or from backend directory:
cd backend
node seed.js
```

### 2. Start Backend API Server
Run the Express backend (default port `5000`):
```bash
# From root directory:
npm run dev:backend

# Or from backend directory:
cd backend
npm install
npm run dev
```

### 3. Start Frontend Web Application
Run the React + Vite frontend (default port `5173`):
```bash
# From root directory:
npm run start:frontend

# Or from frontend directory:
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 4. Start AI Recommendation Service (Optional)
Run the Python Flask recommendation engine:
```bash
# From root directory:
npm run start:ai-recommendation

# Or from ai-service/recommendation directory:
cd ai-service/recommendation
python app.py
```

### 5. Start Volunteer Matching AI Service (Optional)
Run the FastAPI volunteer matching pipeline service:
```bash
# From root directory:
npm run start:ai-matching

# Or from ai-service/volunteer-matching directory:
cd ai-service/volunteer-matching
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 🛠️ API & Core Features

* **Authentication:** JWT-based stateless auth for Users and NGOs.
* **Campaign & NGO Management:** Full CRUD operations for NGOs to launch and manage civic campaigns.
* **User Participation:** Task tracking, badges, impact scoring, event registration.
* **AI Engine:** Content & collaborative recommendation and FastAPI-powered scoring pipeline.

