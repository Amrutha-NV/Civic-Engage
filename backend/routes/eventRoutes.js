const express = require("express");
const router = express.Router();

const protect = require("../middleware/authMiddleware");

const {
  getEvents,
  joinEvent,
  getTasks,
  searchEvents,
  getEventById,
  updateTaskStatus,
  createEvent
} = require("../controllers/eventController");

// CREATE EVENT (for testing)
router.post("/", protect, createEvent);

// SEARCH (put BEFORE :id)
router.get("/search", protect, searchEvents);

// TASKS
router.get("/tasks", protect, getTasks);

// GET ALL EVENTS
router.get("/", protect, getEvents);

// GET EVENT BY ID (LAST)
router.get("/:id", protect, getEventById);

// JOIN EVENT
router.post("/:id/join", protect, joinEvent);

// UPDATE TASK STATUS
router.put("/task/:id", protect, updateTaskStatus);

module.exports = router;