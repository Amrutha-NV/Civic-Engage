const express = require("express");
const router = express.Router();
const {
  getEventRecommendations,
  getVolunteerRecommendations,
  generateTender,
} = require("../controllers/aiController");

router.post("/recommendations/events", getEventRecommendations);
router.post("/recommendations/volunteers", getVolunteerRecommendations);
router.post("/tender/generate", generateTender);

module.exports = router;
