const express = require("express");
const router = express.Router();
const {
  registerNGO,
  loginNGO,
  getMe,
} = require("../controllers/authController");
const {
  sendSignupOtp,
  verifySignupOtp,
  sendLoginOtp,
  verifyLoginOtp,
} = require("../controllers/emailOtpController");
const { protect } = require("../middleware/authMiddleware");
const rateLimit = require("express-rate-limit");

const sendOtpLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 5,
  standardHeaders: "draft-8",
  legacyHeaders: false,
  message: { message: "Too many OTP requests. Please try again later." },
});
const verifyOtpLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 10,
  standardHeaders: "draft-8",
  legacyHeaders: false,
  message: { message: "Too many verification attempts. Please try again later." },
});

router.post("/user/signup/send-otp", sendOtpLimiter, sendSignupOtp);
router.post("/user/signup/verify-otp", verifyOtpLimiter, verifySignupOtp);
router.post("/user/login/send-otp", sendOtpLimiter, sendLoginOtp);
router.post("/user/login/verify-otp", verifyOtpLimiter, verifyLoginOtp);

router.post("/ngo/register", registerNGO);
router.post("/ngo/login", loginNGO);

router.get("/me", protect, getMe);

module.exports = router;
