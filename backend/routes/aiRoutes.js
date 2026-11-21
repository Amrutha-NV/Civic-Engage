const express = require("express");
const router = express.Router();

const {
  getEventRecommendations,
  getVolunteerRecommendations,
  generateTender: generateAITender,
} = require("../controllers/aiController");

const {
  generateTender: generateMLTender,
} = require("../controllers/tenderController");

const { protectNGO } = require("../middleware/authMiddleware");

router.post("/recommendations/events", getEventRecommendations);
router.post("/recommendations/volunteers", getVolunteerRecommendations);

router.post("/tender/generate", generateAITender);

// ML price-prediction tender route
router.post("/tender/ml-generate", protectNGO, generateMLTender);

module.exports = router;