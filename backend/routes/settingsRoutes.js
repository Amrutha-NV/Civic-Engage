const express = require("express");
const router = express.Router();
const protect = require("../middleware/authMiddleware");
const { changePassword, updatePreferences } = require("../controllers/settingsController");

router.put("/password", protect, changePassword);
router.put("/preferences", protect, updatePreferences);

module.exports = router;
