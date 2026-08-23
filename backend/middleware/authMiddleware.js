const jwt = require("jsonwebtoken");
const User = require("../models/User");
const NGO = require("../models/NGO");

const JWT_SECRET = process.env.JWT_SECRET || "mysecretkey123";

const protect = async (req, res, next) => {
  try {
    let token;

    if (req.headers.authorization && req.headers.authorization.startsWith("Bearer")) {
      token = req.headers.authorization.split(" ")[1];
    }

    if (!token) {
      return res.status(401).json({ message: "No token provided, authorization denied" });
    }

    const decoded = jwt.verify(token, JWT_SECRET);
    req.auth = decoded;

    const decodedRole = (decoded.role || "").toLowerCase();

    if (decodedRole === "ngo") {
      const ngo = await NGO.findById(decoded.id).select("-password");
      if (!ngo) return res.status(401).json({ message: "NGO not found" });
      req.ngo = ngo;
      req.user = ngo;
    } else {
      const user = await User.findById(decoded.id).select("-password");
      if (!user) return res.status(401).json({ message: "User not found" });
      req.user = user;
    }

    next();
  } catch (error) {
    if (error.name === "TokenExpiredError") {
      return res.status(401).json({ message: "Token expired" });
    }
    return res.status(401).json({ message: "Not authorized" });
  }
};

const protectUser = async (req, res, next) => {
  return protect(req, res, next);
};

const protectNGO = async (req, res, next) => {
  try {
    let token;

    if (req.headers.authorization && req.headers.authorization.startsWith("Bearer")) {
      token = req.headers.authorization.split(" ")[1];
    }

    if (!token) {
      return res.status(401).json({ message: "No token provided, authorization denied" });
    }

    const decoded = jwt.verify(token, JWT_SECRET);
    req.auth = decoded;

    const ngo = await NGO.findById(decoded.id).select("-password");

    if (!ngo) {
      return res.status(401).json({ message: "Not authorized as NGO" });
    }

    req.ngo = ngo;
    req.user = ngo;
    next();
  } catch (error) {
    return res.status(401).json({ message: "Not authorized as NGO" });
  }
};


protect.protect = protect;
protect.protectUser = protectUser;
protect.protectNGO = protectNGO;

module.exports = protect;