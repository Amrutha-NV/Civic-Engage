const Participation = require("../models/Participation");
const User = require("../models/User");

exports.calculateImpactScore = async (req, res) => {
  try {
    const participations = await Participation.find({ user: req.user.id, status: "completed" });
    const totalHours = participations.reduce((acc, curr) => acc + (curr.hoursWorked || 0), 0);
    const score = totalHours * 10;
    
    res.json({ score, totalHours });
  } catch (error) {
    res.status(500).json({ message: "Server error", error: error.message });
  }
};

exports.getBadges = async (req, res) => {
  try {
    const user = await User.findById(req.user.id);
    res.json(user.badges || []);
  } catch (error) {
    res.status(500).json({ message: "Server error", error: error.message });
  }
};

exports.assignBadge = async (req, res) => {
  try {
    const { name } = req.body;
    const targetUserId = req.params.userId;
    
    const user = await User.findById(targetUserId);
    if (!user) return res.status(404).json({ message: "User not found" });

    if (!user.badges) user.badges = [];
    user.badges.push({ name, awardedAt: new Date() });
    await user.save();

    res.json(user.badges);
  } catch (error) {
    res.status(500).json({ message: "Server error", error: error.message });
  }
};
