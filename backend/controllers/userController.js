const User = require("../models/User");
const { syncVolunteerToAIService, resolveLocationCoordinates, resolveLocationCoordinatesAsync } = require("../utils/aiSync");


// @desc Get user profile
// @route GET /api/users/profile
exports.getUserProfile = async (req, res) => {
  try {
    const user = await User.findById(req.user._id).populate("registeredEvents");
    if (!user) {
      return res.status(404).json({ message: "User profile not found" });
    }
    return res.status(200).json({ success: true, user });
  } catch (error) {
    return res.status(500).json({ message: "Error fetching user profile", error: error.message });
  }
};

// @desc Update user profile
// @route PUT /api/users/profile
exports.updateUserProfile = async (req, res) => {
  try {
    const { name, email, skills, interests, availability, location, occupation, avatar } = req.body;
    const user = await User.findById(req.user._id);

    if (!user) {
      return res.status(404).json({ message: "User profile not found" });
    }

    if (name) user.name = name;
    if (email) user.email = email;
    if (skills && Array.isArray(skills)) user.skills = skills;
    if (interests && Array.isArray(interests)) user.interests = interests;
    if (availability) user.availability = availability;
    if (location) {
      user.location = location;
      user.coordinates = await resolveLocationCoordinatesAsync(location);
    }


    if (occupation) user.occupation = occupation;
    if (avatar) user.avatar = avatar;

    await user.save();


    // Trigger AI matching sync for updated volunteer profile
    syncVolunteerToAIService(user);

    return res.status(200).json({ success: true, user });
  } catch (error) {
    return res.status(500).json({ message: "Error updating user profile", error: error.message });
  }
};

// @desc List all volunteers (for NGO / AI services)
// @route GET /api/users/volunteers
exports.getAllVolunteers = async (req, res) => {
  try {
    const volunteers = await User.find({
      $or: [{ role: "user" }, { role: "volunteer" }, { role: { $exists: false } }, { role: null }],
    }).select("-otp -otpExpires");
    return res.status(200).json({ success: true, count: volunteers.length, volunteers });
  } catch (error) {
    return res.status(500).json({ message: "Error fetching volunteers", error: error.message });
  }
};

