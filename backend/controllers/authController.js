const User = require("../models/User");
const NGO = require("../models/NGO");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs");

const DUMMY_OTP = "123456";
const JWT_SECRET = process.env.JWT_SECRET || "mysecretkey123";

const generateToken = (id, role) => {
  return jwt.sign({ id, role }, JWT_SECRET, { expiresIn: "30d" });
};

// @desc Send Dummy OTP for user
// @route POST /api/auth/user/send-otp
exports.sendUserOTP = async (req, res) => {
  try {
    const { phone } = req.body;
    if (!phone) {
      return res.status(400).json({ message: "Mobile number is required" });
    }

    const cleanPhone = phone.replace(/\D/g, "");
    if (cleanPhone.length < 7) {
      return res.status(400).json({ message: "Invalid mobile number format" });
    }

    let user = await User.findOne({ phone });
    if (!user) {
      user = new User({ phone });
    }

    user.otp = DUMMY_OTP;
    user.otpExpires = new Date(Date.now() + 10 * 60 * 1000); // 10 mins
    await user.save();

    return res.status(200).json({
      success: true,
      message: "OTP sent successfully to mobile number.",
      dummyOtp: DUMMY_OTP,
      phone,
    });
  } catch (error) {
    console.error("sendUserOTP Error:", error);
    return res.status(500).json({ message: "Failed to send OTP", error: error.message });
  }
};

const { syncVolunteerToAIService, resolveLocationCoordinates, resolveLocationCoordinatesAsync } = require("../utils/aiSync");

// @desc Verify Dummy OTP for user
exports.verifyUserOTP = async (req, res) => {
  try {
    const { phone, otp, name, email, skills, interests, availability, location, occupation } = req.body;
    if (!phone || !otp) {
      return res.status(400).json({ message: "Phone number and OTP are required" });
    }

    if (otp !== DUMMY_OTP) {
      return res.status(400).json({ message: "Invalid OTP. Use 123456 for demo." });
    }

    let user = await User.findOne({ phone });
    if (!user) {
      user = new User({ phone });
    }

    if (name) user.name = name;
    if (email) user.email = email;
    if (skills && Array.isArray(skills)) user.skills = skills;
    if (interests && Array.isArray(interests)) user.interests = interests;
    if (availability) user.availability = availability;
    if (location) user.location = location;
    if (occupation) user.occupation = occupation;

    // Resolve geographic coordinates dynamically during signup (with OpenStreetMap / India dataset)
    user.coordinates = await resolveLocationCoordinatesAsync(user.location || location);
    user.otp = null;
    user.otpExpires = null;
    user.role = "user";
    await user.save();

    // Trigger AI volunteer matching embedding sync asynchronously
    syncVolunteerToAIService(user);


    const token = generateToken(user._id, "user");

    return res.status(200).json({
      success: true,
      token,
      user,
    });

  } catch (error) {
    console.error("verifyUserOTP Error:", error);
    return res.status(500).json({ message: "OTP verification failed", error: error.message });
  }
};

// @desc Register NGO
// @route POST /api/auth/ngo/register
exports.registerNGO = async (req, res) => {
  try {
    const { ngoName, email, phone, address, password, description, category, logo } = req.body;
    if (!ngoName || !email || !phone || !address || !password) {
      return res.status(400).json({ message: "All required NGO fields must be provided." });
    }

    const existing = await NGO.findOne({ email: email.toLowerCase() });
    if (existing) {
      return res.status(400).json({ message: "An NGO with this email already exists." });
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const ngo = await NGO.create({
      ngoName,
      email: email.toLowerCase(),
      phone,
      address,
      password: hashedPassword,
      description: description || "Dedicated community NGO",
      category: category || "Community Development",
      logo: logo || "https://images.unsplash.com/photo-1599305445671-ac291c95aaa9?w=150",
    });

    const token = generateToken(ngo._id, "ngo");

    const ngoResponse = ngo.toObject();
    delete ngoResponse.password;

    return res.status(201).json({
      success: true,
      token,
      ngo: ngoResponse,
    });
  } catch (error) {
    console.error("registerNGO Error:", error);
    return res.status(500).json({ message: "NGO registration failed", error: error.message });
  }
};

// @desc Login NGO
// @route POST /api/auth/ngo/login
exports.loginNGO = async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({ message: "Email and password are required." });
    }

    const ngo = await NGO.findOne({ email: email.toLowerCase() });
    if (!ngo) {
      return res.status(401).json({ message: "Invalid email or password." });
    }

    const isMatch = await bcrypt.compare(password, ngo.password);
    if (!isMatch && password !== ngo.password) { // fallback for plain text in legacy seeds
      return res.status(401).json({ message: "Invalid email or password." });
    }

    const token = generateToken(ngo._id, "ngo");

    const ngoResponse = ngo.toObject();
    delete ngoResponse.password;

    return res.status(200).json({
      success: true,
      token,
      ngo: ngoResponse,
    });
  } catch (error) {
    console.error("loginNGO Error:", error);
    return res.status(500).json({ message: "NGO login failed", error: error.message });
  }
};

// @desc Get authenticated user/ngo info
// @route GET /api/auth/me
exports.getMe = async (req, res) => {
  try {
    const role = (req.auth?.role || (req.ngo ? "ngo" : "user")).toLowerCase();
    if (req.ngo || role === "ngo") {
      const profile = req.ngo || req.user;
      return res.status(200).json({ success: true, role: "ngo", profile });
    } else if (req.user) {
      return res.status(200).json({ success: true, role: "user", profile: req.user });
    } else {
      return res.status(401).json({ message: "Not authenticated" });
    }
  } catch (error) {
    return res.status(500).json({ message: "Failed to fetch session info" });
  }
};

