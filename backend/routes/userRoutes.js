const express = require("express");
const router = express.Router();
const { getUserProfile, updateUserProfile, getAllVolunteers } = require("../controllers/userController");
const { protectUser, protect } = require("../middleware/authMiddleware");

router.get("/profile", protectUser, getUserProfile);
router.put("/profile", protectUser, updateUserProfile);
router.get("/volunteers", protect, getAllVolunteers);

module.exports = router;
