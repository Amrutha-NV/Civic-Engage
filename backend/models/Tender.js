const mongoose = require("mongoose");

const tenderSchema = new mongoose.Schema(
  {
    ngoId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "NGO",
      required: true,
    },

    procurementType: {
      type: String,
      required: true,
      trim: true,
    },

    requirements: {
      type: String,
      required: true,
      trim: true,
    },

    items: {
      type: String,
      required: true,
      trim: true,
    },

    quantity: {
      type: Number,
      required: true,
    },

    budget: {
      type: Number,
      required: true,
    },

    location: {
      type: String,
      required: true,
      trim: true,
    },

    benchmarkPrice: {
      type: Number,
      required: true,
    },

    recommendedMinPrice: {
      type: Number,
      required: true,
    },

    recommendedMaxPrice: {
      type: Number,
      required: true,
    },

    reliability: {
      type: String,
      required: true,
      trim: true,
    },

    generatedTender: {
      type: String,
      required: true,
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Tender", tenderSchema);