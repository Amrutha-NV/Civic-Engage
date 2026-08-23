const express = require("express");
const router = express.Router();
const {
  getAllEvents,
  getEventById,
  createEvent,
  updateEvent,
  deleteEvent,
  registerUserForEvent,
} = require("../controllers/campaignController");
const { protectNGO, protectUser } = require("../middleware/authMiddleware");

// Public / General Event routes
router.get("/events", getAllEvents);
router.get("/events/:id", getEventById);
router.post("/events/:id/register", protectUser, registerUserForEvent);

// NGO specific routes
router.post("/ngos/events", protectNGO, createEvent);
router.put("/ngos/events/:id", protectNGO, updateEvent);
router.delete("/ngos/events/:id", protectNGO, deleteEvent);

module.exports = router;