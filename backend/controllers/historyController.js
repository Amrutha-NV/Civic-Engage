const Participation = require("../models/Participation");
const Event = require("../models/Event");

exports.getParticipationHistory = async (req, res) => {
  try {
    const participations = await Participation.find({ 
      user: req.user.id, 
      status: { $in: ["completed", "past"] } 
    }).populate("event");
    
    res.json(participations);
  } catch (error) {
    res.status(500).json({ message: "Server error", error: error.message });
  }
};

exports.getEventHistory = async (req, res) => {
  try {
    // Assuming Event has an 'organizer' field pointing to User
    const events = await Event.find({ organizer: req.user.id, date: { $lt: new Date() } });
    res.json(events);
  } catch (error) {
    res.status(500).json({ message: "Server error", error: error.message });
  }
};
