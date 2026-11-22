# 🌍 CivicEngage – Backend (User Module)

## 🚀 Project Overview

CivicEngage is a platform that connects **volunteers** with **social campaigns and events**.

This backend handles:

* User authentication
* Profile management
* Event browsing
* Participation tracking
* Task management
* Dashboard analytics

---

## 🛠️ Tech Stack

* Node.js
* Express.js
* MongoDB (Mongoose)
* JWT Authentication
* bcrypt (Password Hashing)

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone <your-repo-link>
cd CivicEngage/backend
```

### 2. Install dependencies

```
npm install
```

### 3. Add `.env` file

```
PORT=5000
MONGO_URI=your_mongodb_connection
JWT_SECRET=your_secret_key
```

### 4. Run server

```
npm run dev
```

---

## 🔐 Authentication APIs

Volunteer signup and login use email OTP. Configure Gmail credentials and `OTP_SECRET` using [EMAIL_OTP_SETUP.md](EMAIL_OTP_SETUP.md) before testing these endpoints.

### 🔹 Start Volunteer Signup

**POST** `/api/auth/user/signup/send-otp`

Send the signup profile fields (`name`, `phone`, `email`, and optional profile fields). The backend emails an OTP and holds the pending profile until verification; it does not create the account at this step.

### 🔹 Verify Signup OTP and Create Account

**POST** `/api/auth/user/signup/verify-otp`

```
{
  "email": "volunteer@example.com",
  "otp": "six-digit-code"
}
```

On success, the backend creates the volunteer and returns the normal user JWT/profile. The frontend uses these to log the volunteer in automatically.

### 🔹 Start Volunteer Login

**POST** `/api/auth/user/login/send-otp`

Body: `{ "email": "volunteer@example.com" }`.

### 🔹 Verify Login OTP

**POST** `/api/auth/user/login/verify-otp`

Body: `{ "email": "volunteer@example.com", "otp": "six-digit-code" }`.

### 🔹 NGO Authentication

NGO signup and password login are unchanged at `/api/auth/ngo/register` and `/api/auth/ngo/login`.

---

### 🔹 Get Current User

**GET** `/api/auth/me`

Header:

```
Authorization: Bearer TOKEN
```

---

### 🔹 Update Profile

**PUT** `/api/auth/profile`

---

### 🔹 Change Password

**PUT** `/api/auth/change-password`

---

### 🔹 Dashboard

**GET** `/api/auth/dashboard`

Returns:

* user info
* total tasks
* completed tasks
* impact score

---

## 🎯 Event APIs

### 🔹 Get All Events

**GET** `/api/events`

---

### 🔹 Get Event Details

**GET** `/api/events/:id`

---

### 🔹 Search / Filter Events

**GET** `/api/events/search?location=Bangalore&skill=teaching`

---

### 🔹 Create Event (Testing Only)

**POST** `/api/events`

---

## 📋 Participation & Tasks APIs

### 🔹 Join Event

**POST** `/api/events/:id/join`

Creates a task with status = `pending`

---

### 🔹 Get Tasks

**GET** `/api/events/tasks`

Returns:

* pending
* ongoing
* completed

---

### 🔹 Update Task Status

**PUT** `/api/events/task/:id`

Body:

```
{
  "status": "completed"
}
```

---

## 🔄 Task Lifecycle

```
pending → ongoing → completed
```

* **pending** → user joined event
* **ongoing** → user started work
* **completed** → task finished

---

## 🔗 Frontend Integration Guide

### 🟢 Login Flow

1. Call `/login`
2. Store JWT token
3. Send token in all requests

---

### 🟢 Dashboard Page

Call:

```
GET /api/auth/dashboard
```

---

### 🟢 Events Page

Call:

```
GET /api/events
```

---

### 🟢 Participate Button

Call:

```
POST /api/events/:id/join
```

---

### 🟢 My Tasks Page

Call:

```
GET /api/events/tasks
```

---

### 🟢 Search Feature

Call:

```
GET /api/events/search?location=&skill=
```

---

## 📌 Notes

* All protected routes require JWT token
* Passwords are securely hashed
* Dashboard data is dynamically calculated
* Participation links users with events

---

## 🎯 Future Scope

* NGO module
* Real-time notifications
* AI-based recommendations
* QR-based check-in system

---
---

## 🤖 AI Recommendation API

This module recommends suitable volunteer activities based on user interests, especially useful during disaster situations.

---

### 🔹 Get Activity Recommendations

**POST** `/api/ai/recommend`

---

### 📥 Request Body

```json
{
  "interest": "disaster"
}
```
### 📤 Response
```json
[
  "Flood relief food distribution",
  "Rescue support for flood victims",
  "Relief material packing"
]
```
### 🔗 System Flow

Client → Node.js Backend → Flask AI Service → Response

## 👨‍💻 Author

Deepthi & Team 🚀


---

## 👨‍💻 Backend Contribution – Shashank Srinivas

This section outlines my contribution to the backend development of the CivicEngage platform.

### 🔧 Modules Implemented

#### 📌 Notification System
- Developed APIs to create, fetch, and update notifications
- Implemented read/unread status functionality

#### 💬 Messaging System
- Built APIs for sending and retrieving messages between users

#### 🖼️ Image Handling
- Implemented image upload and backend media handling

#### 🤖 AI Recommendation Integration
- Integrated AI-based volunteer recommendation module with backend services

---

### 📡 API Endpoints

**Notifications**
- GET /notifications  
- POST /notifications  
- PUT /notifications/read  

**Messages**
- POST /messages  
- GET /messages  

**Images**
- POST /upload  

---

### ⚙️ Technologies Used
- Node.js  
- Express.js  
- MongoDB  

---

### ✅ Key Highlights
- Designed modular backend architecture  
- Followed clean API structuring  
- Ensured scalability and maintainability
### 👥 Team Contributions
