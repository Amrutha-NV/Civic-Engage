const Event = require("../models/Event");
const Participation = require("../models/Participation");

// GET EVENTS
exports.getEvents = async (req, res) => {
  try {
    const events = await Event.find();
    res.json({
      success: true,
      events
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// JOIN EVENT
exports.joinEvent = async (req, res) => {
  try {
    const eventId = req.params.id;

    const existing = await Participation.findOne({
      user: req.user._id,
      event: eventId
    });

    if (existing) {
      return res.status(400).json({ message: "Already joined" });
    }

    const participation = await Participation.create({
      user: req.user._id,
      event: eventId
    });

    res.json({
      success: true,
      message: "Joined successfully",
      participation
    });

  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// TASKS
exports.getTasks = async (req, res) => {
  try {
    const tasks = await Participation.find({
      user: req.user._id
    }).populate("event");

    const pending = tasks.filter(t => t.status === "pending");
    const ongoing = tasks.filter(t => t.status === "ongoing");
    const completed = tasks.filter(t => t.status === "completed");

    res.json({
      success: true,
      pending,
      ongoing,
      completed
    });

  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// SEARCH
exports.searchEvents = async (req, res) => {
  try {
    const { location, skill } = req.query;

    let query = {};

    if (location) query.location = location;
    if (skill) query.requiredSkills = skill;

    const events = await Event.find(query);

    res.json({
      success: true,
      events
    });

  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

//temporary for testing
exports.createEvent = async (req, res) => {
  try {
    const event = await Event.create({
      title: req.body.title,
      description: req.body.description,
      location: req.body.location,
      requiredSkills: req.body.requiredSkills,
      createdBy: req.user._id
    });

    res.json({
      success: true,
      event
    });

  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

exports.getEventById = async (req, res) => {
  try {
    const event = await Event.findById(req.params.id);

    res.json({
      success: true,
      event
    });

  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

exports.updateTaskStatus = async (req, res) => {
  try {
    const { status } = req.body;

    const task = await Participation.findById(req.params.id);

    if (!task) {
      return res.status(404).json({ message: "Task not found" });
    }

    task.status = status;
    await task.save();

    res.json({
      success: true,
      message: "Status updated",
      task
    });

  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};