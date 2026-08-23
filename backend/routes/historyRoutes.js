const express = require("express");
const router = express.Router();
const protect = require("../middleware/authMiddleware");
const { getParticipationHistory, getEventHistory } = require("../controllers/historyController");

router.get("/participations", protect, getParticipationHistory);
router.get("/events", protect, getEventHistory);

module.exports = router;
