const express = require("express");
const router = express.Router();
const {
  sendUserOTP,
  verifyUserOTP,
  registerNGO,
  loginNGO,
  getMe,
} = require("../controllers/authController");
const { protect } = require("../middleware/authMiddleware");

router.post("/user/send-otp", sendUserOTP);
router.post("/user/verify-otp", verifyUserOTP);

router.post("/ngo/register", registerNGO);
router.post("/ngo/login", loginNGO);

router.get("/me", protect, getMe);

module.exports = router;
