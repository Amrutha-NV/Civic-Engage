const mongoose = require("mongoose");

const participationSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  event: { type: mongoose.Schema.Types.ObjectId, ref: "Event" },

  status: {
    type: String,
    enum: ["pending", "approved", "ongoing", "completed"],
    default: "pending"
  },

  checkInTime: Date,
  checkOutTime: Date,
  hoursWorked: Number

}, { timestamps: true });

module.exports = mongoose.model("Participation", participationSchema);