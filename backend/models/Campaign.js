const mongoose = require("mongoose");

const campaignSchema = new mongoose.Schema(
  {
    ngoId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "NGO",
      required: true,
    },
    title: {
      type: String,
      required: true,
      trim: true,
    },
    description: {
      type: String,
      required: true,
    },
    shortDesc: {
      type: String,
      default: "",
    },
    category: {
      type: String,
      required: true,
      default: "General",
    },
    requiredSkills: {
      type: [String],
      default: [],
    },
    location: {
      type: String,
      required: true,
      trim: true,
    },
    coordinates: {
      lat: { type: Number },
      lng: { type: Number },
    },
    date: {
      type: String,
      required: true,
    },
    startTime: {
      type: String,
      default: "08:00 AM",
    },
    endTime: {
      type: String,
      default: "04:00 PM",
    },
    banner: {
      type: String,
      default: "https://images.unsplash.com/photo-1559027615-cd4628902d4a?w=800",
    },
    capacity: {
      type: Number,
      default: 50,
    },
    volunteersJoined: {
      type: Number,
      default: 0,
    },
    status: {
      type: String,
      enum: ["Active", "Upcoming", "Completed", "Cancelled"],
      default: "Active",
    },
    registeredUsers: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
      },
    ],
  },
  {
    timestamps: true,
  }
);

const { resolveLocationCoordinates } = require("../utils/aiSync");

campaignSchema.pre("save", function (next) {
  if (this.isModified("location") || !this.coordinates || typeof this.coordinates.lat !== "number") {
    this.coordinates = resolveLocationCoordinates(this.location);
  }
  next();
});

module.exports = mongoose.model("Campaign", campaignSchema);