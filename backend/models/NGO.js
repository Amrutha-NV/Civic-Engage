const mongoose = require("mongoose");

const ngoSchema = new mongoose.Schema(
  {
    ngoName: {
      type: String,
      required: true,
      trim: true,
    },
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
    },
    phone: {
      type: String,
      required: true,
      trim: true,
    },
    address: {
      type: String,
      required: true,
      trim: true,
    },
    password: {
      type: String,
      required: true,
    },
    description: {
      type: String,
      default: "Dedicated NGO working towards community improvement and youth empowerment.",
    },
    category: {
      type: String,
      default: "Community Development",
    },
    logo: {
      type: String,
      default: "https://images.unsplash.com/photo-1599305445671-ac291c95aaa9?w=150",
    },
    role: {
      type: String,
      default: "ngo",
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("NGO", ngoSchema);