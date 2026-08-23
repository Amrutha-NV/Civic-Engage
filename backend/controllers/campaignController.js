const Campaign = require("../models/Campaign");
const User = require("../models/User");
const { resolveLocationCoordinatesAsync } = require("../utils/aiSync");


// @desc Get all events / campaigns
// @route GET /api/events
exports.getAllEvents = async (req, res) => {
  try {
    const { search, category, status, ngoId } = req.query;
    let query = {};

    if (ngoId) {
      query.ngoId = ngoId;
    }
    if (status && status !== "All") {
      query.status = status;
    }
    if (category && category !== "All") {
      query.category = category;
    }
    if (search) {
      query.$or = [
        { title: { $regex: search, $options: "i" } },
        { location: { $regex: search, $options: "i" } },
        { description: { $regex: search, $options: "i" } },
      ];
    }

    const events = await Campaign.find(query)
      .populate("ngoId", "ngoName email phone address logo category")
      .sort({ createdAt: -1 });

    return res.status(200).json({ success: true, count: events.length, events });
  } catch (error) {
    console.error("getAllEvents Error:", error);
    return res.status(500).json({ message: "Failed to fetch events", error: error.message });
  }
};

// @desc Get single event by ID
// @route GET /api/events/:id
exports.getEventById = async (req, res) => {
  try {
    const event = await Campaign.findById(req.params.id)
      .populate("ngoId", "ngoName email phone address logo category description")
      .populate("registeredUsers", "name email phone skills availability avatar");

    if (!event) {
      return res.status(404).json({ message: "Event not found" });
    }

    return res.status(200).json({ success: true, event });
  } catch (error) {
    return res.status(500).json({ message: "Failed to fetch event", error: error.message });
  }
};

// @desc Create new event (NGO Protected)
// @route POST /api/ngos/events
exports.createEvent = async (req, res) => {
  try {
    const {
      title,
      description,
      shortDesc,
      category,
      requiredSkills,
      location,
      date,
      startTime,
      endTime,
      banner,
      capacity,
    } = req.body;

    if (!title || !description || !location || !date) {
      return res.status(400).json({ message: "Title, description, location, and date are required." });
    }

    const skillsArray = Array.isArray(requiredSkills)
      ? requiredSkills
      : typeof requiredSkills === "string"
      ? requiredSkills.split(",").map((s) => s.trim())
      : [];

    const ngoId = req.ngo?._id || req.user?._id || req.auth?.id;
    if (!ngoId) {
      return res.status(401).json({ message: "Not authorized as NGO" });
    }

    const coords = await resolveLocationCoordinatesAsync(location);

    const event = await Campaign.create({
      ngoId,
      title,
      description,
      shortDesc: shortDesc || description.substring(0, 120),
      category: category || "General",
      requiredSkills: skillsArray,
      location,
      coordinates: coords,
      date,
      startTime: startTime || "08:00 AM",
      endTime: endTime || "04:00 PM",
      banner: banner || "https://images.unsplash.com/photo-1559027615-cd4628902d4a?w=800",
      capacity: capacity ? Number(capacity) : 50,
      volunteersJoined: 0,
      status: "Active",
    });

    return res.status(201).json({ success: true, event });

  } catch (error) {
    console.error("createEvent Error:", error);
    return res.status(500).json({ message: "Failed to create event", error: error.message });
  }
};

// @desc Update event (NGO Protected)
// @route PUT /api/ngos/events/:id
exports.updateEvent = async (req, res) => {
  try {
    const event = await Campaign.findById(req.params.id);
    if (!event) {
      return res.status(404).json({ message: "Event not found" });
    }

    // Verify ownership
    if (event.ngoId.toString() !== req.ngo._id.toString()) {
      return res.status(403).json({ message: "Forbidden: You do not own this event" });
    }

    const {
      title,
      description,
      shortDesc,
      category,
      requiredSkills,
      location,
      date,
      startTime,
      endTime,
      banner,
      capacity,
      status,
    } = req.body;

    if (title) event.title = title;
    if (description) event.description = description;
    if (shortDesc) event.shortDesc = shortDesc;
    if (category) event.category = category;
    if (requiredSkills) {
      event.requiredSkills = Array.isArray(requiredSkills)
        ? requiredSkills
        : requiredSkills.split(",").map((s) => s.trim());
    }
    if (location) event.location = location;
    if (date) event.date = date;
    if (startTime) event.startTime = startTime;
    if (endTime) event.endTime = endTime;
    if (banner) event.banner = banner;
    if (capacity) event.capacity = Number(capacity);
    if (status) event.status = status;

    await event.save();
    return res.status(200).json({ success: true, event });
  } catch (error) {
    return res.status(500).json({ message: "Failed to update event", error: error.message });
  }
};

// @desc Delete event (NGO Protected)
// @route DELETE /api/ngos/events/:id
exports.deleteEvent = async (req, res) => {
  try {
    const event = await Campaign.findById(req.params.id);
    if (!event) {
      return res.status(404).json({ message: "Event not found" });
    }

    if (event.ngoId.toString() !== req.ngo._id.toString()) {
      return res.status(403).json({ message: "Forbidden: You do not own this event" });
    }

    await Campaign.findByIdAndDelete(req.params.id);
    return res.status(200).json({ success: true, message: "Event deleted successfully" });
  } catch (error) {
    return res.status(500).json({ message: "Failed to delete event", error: error.message });
  }
};

// @desc Register User for Event
// @route POST /api/events/:id/register
exports.registerUserForEvent = async (req, res) => {
  try {
    const event = await Campaign.findById(req.params.id);
    if (!event) {
      return res.status(404).json({ message: "Event not found" });
    }

    const userId = req.user._id;

    // Check duplicate
    const alreadyRegistered = event.registeredUsers.some(
      (id) => id.toString() === userId.toString()
    );
    if (alreadyRegistered) {
      return res.status(400).json({ message: "You are already registered for this event." });
    }

    // Check capacity
    if (event.volunteersJoined >= event.capacity) {
      return res.status(400).json({ message: "Event capacity has been reached." });
    }

    event.registeredUsers.push(userId);
    event.volunteersJoined += 1;
    await event.save();

    // Update user registeredEvents
    const user = await User.findById(userId);
    if (user && !user.registeredEvents.includes(event._id)) {
      user.registeredEvents.push(event._id);
      await user.save();
    }

    return res.status(200).json({
      success: true,
      message: "Successfully registered for event!",
      event,
    });
  } catch (error) {
    console.error("registerUserForEvent Error:", error);
    return res.status(500).json({ message: "Event registration failed", error: error.message });
  }
};