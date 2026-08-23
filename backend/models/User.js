const mongoose = require("mongoose");

const userSchema = new mongoose.Schema(
  {
    phone: {
      type: String,
      required: true,
      unique: true,
      trim: true,
    },
    name: {
      type: String,
      default: "Volunteer",
      trim: true,
    },
    email: {
      type: String,
      trim: true,
      lowercase: true,
      default: "",
    },
    avatar: {
      type: String,
      default: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
    },
    skills: {
      type: [String],
      default: ["Community Service"],
    },
    interests: {
      type: [String],
      default: ["Environment", "Education"],
    },
    availability: {
      type: String,
      enum: ["Available", "Weekends", "Weekdays", "Flexible"],
      default: "Available",
    },
    location: {
      type: String,
      default: "Bengaluru, Karnataka, India",
      trim: true,
    },
    coordinates: {
      lat: { type: Number },
      lng: { type: Number },
    },
    occupation: {
      type: String,
      default: "Student / Volunteer",
      trim: true,
    },
    impactScore: {
      type: Number,
      default: 85,
    },
    otp: {
      type: String,
      default: null,
    },
    otpExpires: {
      type: Date,
      default: null,
    },
    role: {
      type: String,
      default: "user",
    },
    registeredEvents: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Campaign",
      },
    ],
  },
  {
    timestamps: true,
  }
);

const { resolveLocationCoordinates } = require("../utils/aiSync");

userSchema.pre("save", function (next) {
  if (this.isModified("location") || !this.coordinates || typeof this.coordinates.lat !== "number") {
    this.coordinates = resolveLocationCoordinates(this.location);
  }
  next();
});

module.exports = mongoose.model("User", userSchema);



