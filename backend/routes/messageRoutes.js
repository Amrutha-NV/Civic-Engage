const express = require("express");
const router = express.Router();
const protect = require("../middleware/authMiddleware");
const { sendMessage, getConversation } = require("../controllers/messageController");

router.post("/", protect, sendMessage);
router.get("/:userId", protect, getConversation);

module.exports = router;
