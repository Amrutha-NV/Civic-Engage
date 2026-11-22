const crypto = require("crypto");
const jwt = require("jsonwebtoken");
const nodemailer = require("nodemailer");
const otpGenerator = require("otp-generator");
const EmailOtp = require("../models/EmailOtp");
const User = require("../models/User");
const {
  syncVolunteerToAIService,
  resolveLocationCoordinatesAsync,
} = require("../utils/aiSync");

const OTP_TTL_MS = 5 * 60 * 1000;
const RESEND_COOLDOWN_MS = 60 * 1000;
const MAX_OTP_ATTEMPTS = 5;
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const GENERIC_LOGIN_MESSAGE = "If an account exists for this email, an OTP has been sent.";

const normalizeEmail = (email) => String(email || "").trim().toLowerCase();
const createToken = (user) =>
  jwt.sign(
    { id: user._id, role: "user" },
    process.env.JWT_SECRET || "mysecretkey123",
    { expiresIn: "30d" }
  );

const createOtpHash = (email, purpose, otp) => {
  if (!process.env.OTP_SECRET) {
    throw new Error("OTP_SECRET is not configured");
  }
  return crypto
    .createHmac("sha256", process.env.OTP_SECRET)
    .update(`${email}:${purpose}:${otp}`)
    .digest("hex");
};

const sendEmailOtp = async (email, otp) => {
  const { GMAIL_USER, GMAIL_APP_PASSWORD } = process.env;
  if (!GMAIL_USER || !GMAIL_APP_PASSWORD) {
    throw new Error("Gmail SMTP credentials are not configured");
  }

  const transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
      user: GMAIL_USER,
      pass: GMAIL_APP_PASSWORD.replace(/\s/g, ""),
    },
  });

  await transporter.sendMail({
    from: process.env.GMAIL_FROM || `CivicEngage <${GMAIL_USER}>`,
    to: email,
    subject: "CivicEngage email verification code",
    text: `Your CivicEngage verification code is ${otp}. It expires in 5 minutes. If you did not request this code, you can ignore this email.`,
    html: `<p>Your CivicEngage verification code is:</p><p style="font-size:24px;font-weight:bold;letter-spacing:6px">${otp}</p><p>This code expires in 5 minutes. If you did not request it, you can ignore this email.</p>`,
  });
};

const validateEmail = (email) =>
  email.length <= 254 && EMAIL_PATTERN.test(email);

const validateSignupData = (body) => {
  const email = normalizeEmail(body.email);
  const name = String(body.name || "").trim();
  const phone = String(body.phone || "").trim();
  const phoneDigits = phone.replace(/\D/g, "");

  if (!validateEmail(email)) return { error: "Enter a valid email address." };
  if (!name || name.length > 100) return { error: "Enter your name (up to 100 characters)." };
  if (phoneDigits.length < 7 || phoneDigits.length > 15) {
    return { error: "Enter a valid mobile number." };
  }

  const stringList = (value) =>
    Array.isArray(value)
      ? value.filter((item) => typeof item === "string").slice(0, 30)
      : undefined;

  return {
    data: {
      email,
      name,
      phone,
      location: String(body.location || "Bengaluru, Karnataka").trim().slice(0, 200),
      occupation: String(body.occupation || "Student / Volunteer").trim().slice(0, 100),
      availability: ["Available", "Weekends", "Weekdays", "Flexible"].includes(body.availability)
        ? body.availability
        : "Available",
      skills: stringList(body.skills),
      interests: stringList(body.interests),
    },
  };
};

const issueOtp = async ({ email, purpose, signupData }) => {
  const now = new Date();
  const existing = await EmailOtp.findOne({ email, purpose }).select("sentAt");
  if (existing && now.getTime() - existing.sentAt.getTime() < RESEND_COOLDOWN_MS) {
    const error = new Error("Please wait a minute before requesting another code.");
    error.status = 429;
    throw error;
  }

  const otp = otpGenerator.generate(6, {
    digits: true,
    lowerCaseAlphabets: false,
    upperCaseAlphabets: false,
    specialChars: false,
  });
  const challenge = await EmailOtp.findOneAndUpdate(
    { email, purpose },
    {
      $set: {
        email,
        purpose,
        codeHash: createOtpHash(email, purpose, otp),
        expiresAt: new Date(now.getTime() + OTP_TTL_MS),
        sentAt: now,
        attempts: 0,
        signupData,
      },
    },
    { new: true, upsert: true, setDefaultsOnInsert: true }
  );

  try {
    await sendEmailOtp(email, otp);
  } catch (error) {
    await EmailOtp.deleteOne({ _id: challenge._id });
    throw error;
  }
};

const consumeOtp = async (email, purpose, submittedOtp) => {
  if (!/^\d{6}$/.test(String(submittedOtp || ""))) return null;
  const now = new Date();
  const challenge = await EmailOtp.findOne({
    email,
    purpose,
    expiresAt: { $gt: now },
  }).select("+codeHash +signupData");
  if (!challenge) return null;

  const expectedHash = createOtpHash(email, purpose, String(submittedOtp));
  const expected = Buffer.from(challenge.codeHash, "hex");
  const actual = Buffer.from(expectedHash, "hex");
  const isValid = expected.length === actual.length && crypto.timingSafeEqual(expected, actual);

  if (!isValid) {
    const updated = await EmailOtp.findOneAndUpdate(
      { _id: challenge._id, expiresAt: { $gt: now } },
      { $inc: { attempts: 1 } },
      { new: true }
    );
    if (updated && updated.attempts >= MAX_OTP_ATTEMPTS) {
      await EmailOtp.deleteOne({ _id: updated._id });
    }
    return null;
  }

  // Conditional deletion makes a code single-use and prevents races with resend.
  return EmailOtp.findOneAndDelete({
    _id: challenge._id,
    codeHash: challenge.codeHash,
    expiresAt: { $gt: now },
  }).select("+signupData");
};

exports.sendSignupOtp = async (req, res) => {
  try {
    const { data, error } = validateSignupData(req.body);
    if (error) return res.status(400).json({ message: error });

    const duplicate = await User.findOne({
      $or: [{ email: data.email }, { phone: data.phone }],
    }).select("_id email phone");
    if (duplicate) {
      const field = duplicate.email === data.email ? "email" : "mobile number";
      return res.status(409).json({ message: `An account already uses this ${field}. Please log in or use another value.` });
    }

    await issueOtp({ email: data.email, purpose: "signup", signupData: data });
    return res.status(200).json({ message: "Verification code sent to your email." });
  } catch (error) {
    if (error.status === 429) return res.status(429).json({ message: error.message });
    if (error.message.includes("not configured")) {
      console.error("Email OTP configuration error:", error.message);
      return res.status(503).json({ message: "Email verification is temporarily unavailable. Please contact support." });
    }
    console.error("sendSignupOtp failed:", error.message);
    return res.status(500).json({ message: "Could not send the verification email." });
  }
};

exports.verifySignupOtp = async (req, res) => {
  try {
    const email = normalizeEmail(req.body.email);
    if (!validateEmail(email) || !req.body.otp) {
      return res.status(400).json({ message: "A valid email and 6-digit code are required." });
    }

    const challenge = await consumeOtp(email, "signup", req.body.otp);
    if (!challenge) return res.status(400).json({ message: "Invalid or expired verification code." });

    const data = challenge.signupData;
    if (!data || data.email !== email) {
      return res.status(400).json({ message: "Invalid or expired verification code." });
    }

    const duplicate = await User.findOne({
      $or: [{ email: data.email }, { phone: data.phone }],
    }).select("_id");
    if (duplicate) {
      return res.status(409).json({ message: "An account already exists with these details. Please log in." });
    }

    const user = new User(data);
    user.coordinates = await resolveLocationCoordinatesAsync(user.location);
    user.role = "user";
    await user.save();
    syncVolunteerToAIService(user);

    return res.status(201).json({ success: true, token: createToken(user), user });
  } catch (error) {
    if (error.code === 11000) {
      return res.status(409).json({ message: "An account already exists with these details. Please log in." });
    }
    console.error("verifySignupOtp failed:", error.message);
    return res.status(500).json({ message: "Could not complete signup. Please request a new code and try again." });
  }
};

exports.sendLoginOtp = async (req, res) => {
  try {
    const email = normalizeEmail(req.body.email);
    if (!validateEmail(email)) return res.status(400).json({ message: "Enter a valid email address." });

    const user = await User.findOne({ email }).select("_id");
    if (user) await issueOtp({ email, purpose: "login" });
    return res.status(200).json({ message: GENERIC_LOGIN_MESSAGE });
  } catch (error) {
    if (error.status === 429) return res.status(429).json({ message: error.message });
    if (error.message.includes("not configured")) {
      console.error("Email OTP configuration error:", error.message);
      return res.status(503).json({ message: "Email verification is temporarily unavailable. Please contact support." });
    }
    console.error("sendLoginOtp failed:", error.message);
    return res.status(500).json({ message: "Could not send the verification email." });
  }
};

exports.verifyLoginOtp = async (req, res) => {
  try {
    const email = normalizeEmail(req.body.email);
    if (!validateEmail(email) || !req.body.otp) {
      return res.status(400).json({ message: "A valid email and 6-digit code are required." });
    }

    const challenge = await consumeOtp(email, "login", req.body.otp);
    if (!challenge) return res.status(400).json({ message: "Invalid or expired verification code." });

    const user = await User.findOne({ email });
    if (!user) return res.status(400).json({ message: "Invalid or expired verification code." });

    return res.status(200).json({ success: true, token: createToken(user), user });
  } catch (error) {
    console.error("verifyLoginOtp failed:", error.message);
    return res.status(500).json({ message: "Could not verify the code. Please request a new one." });
  }
};