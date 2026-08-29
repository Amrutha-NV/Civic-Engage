const express = require("express");
const cors = require("cors");
const path = require("path");
const dotenv = require("dotenv");

// 1. Load environment variables from root directory (.env), with fallback to local .env
const rootEnv = path.resolve(__dirname, "..", ".env");
const localEnv = path.resolve(__dirname, ".env");
dotenv.config({ path: rootEnv });
dotenv.config({ path: localEnv });

// 2. Import and run connectDB ONLY after environment variables are loaded
const connectDB = require("./config/db");
connectDB();

const cloudinary = require("cloudinary").v2;
const app = express();

app.use(cors());
app.use(express.json());

// Clean up Cloudinary keys globally on server initialization
const cloudName = (process.env.CLOUDINARY_CLOUD_NAME || "").replace(
  /["'\s]/g,
  "",
);
const apiKey = (process.env.CLOUDINARY_API_KEY || "").replace(/["'\s]/g, "");
const apiSecret = (process.env.CLOUDINARY_API_SECRET || "").replace(
  /["'\s]/g,
  "",
);

if (cloudName && apiKey && apiSecret) {
  cloudinary.config({
    cloud_name: cloudName,
    api_key: apiKey,
    api_secret: apiSecret,
  });

  console.log("✅ Cloudinary Global Config Initialization Successful");
} else {
  console.warn(
    "⚠️ Cloudinary initialization skipped: Keys missing in environment configuration.",
  );
}

// ===============================
// API Routes
// ===============================

app.use("/api/auth", require("./routes/authRoutes"));
app.use("/api/users", require("./routes/userRoutes"));
app.use("/api", require("./routes/campaignroutes"));
app.use("/api/ngos", require("./routes/ngoroutes"));
app.use("/api", require("./routes/aiRoutes"));
app.use("/api", require("./routes/uploadRoutes"));
app.use("/api/events", require("./routes/eventRoutes"));
app.use("/api/notifications", require("./routes/notificationRoutes"));
app.use("/api/messages", require("./routes/messageRoutes"));
app.use("/api/settings", require("./routes/settingsRoutes"));
app.use("/api/impact", require("./routes/impactRoutes"));
app.use("/api/history", require("./routes/historyRoutes"));

// ===============================
// Root Health Check
// ===============================

app.get("/", (req, res) => {
  res.json({
    status: "online",
    message: "CivicEngage Backend API is running cleanly",
    timestamp: new Date().toISOString(),
  });
});

// ===============================
// Centralized Error Handler
// ===============================

app.use((err, req, res, next) => {
  console.error("Server Error:", err);

  res.status(err.status || 500).json({
    message: err.message || "Internal Server Error",
    error: process.env.NODE_ENV === "development" ? err : {},
  });
});

// ===============================
// Start Server
// ===============================

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`🚀 CivicEngage Backend Server listening on port ${PORT}`);
});
