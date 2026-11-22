const User = require("../models/User");
const NGO = require("../models/NGO");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs");

const JWT_SECRET = process.env.JWT_SECRET || "mysecretkey123";

const generateToken = (id, role) => {
  return jwt.sign({ id, role }, JWT_SECRET, { expiresIn: "30d" });
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

