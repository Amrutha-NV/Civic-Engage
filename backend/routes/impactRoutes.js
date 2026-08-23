const express = require("express");
const router = express.Router();
const protect = require("../middleware/authMiddleware");
const { calculateImpactScore, getBadges, assignBadge } = require("../controllers/impactController");

router.get("/score", protect, calculateImpactScore);
router.get("/badges", protect, getBadges);
router.post("/badges/:userId", protect, assignBadge);

module.exports = router;
