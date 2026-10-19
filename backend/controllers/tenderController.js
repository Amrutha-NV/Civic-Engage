
const axios = require("axios");
const Tender = require("../models/Tender");

// --------------------------------------------------
// ML SERVICE URLS
// --------------------------------------------------

const ML_SERVICE_URL =
  process.env.ML_PRICE_PREDICTION_SERVICE_URL ||
  "http://127.0.0.1:8000";

const MARKET_EVIDENCE_SERVICE_URL =
  process.env.ML_MARKET_EVIDENCE_SERVICE_URL ||
  "http://127.0.0.1:8001";

// --------------------------------------------------
// VALIDATE PRICE RESULT
// --------------------------------------------------

function isValidPriceResult(result) {
  if (!result || typeof result !== "object") {
    return false;
  }

  const benchmark = Number(result.benchmarkUnitPrice);
  const lower = Number(result.expectedRange?.lower);
  const upper = Number(result.expectedRange?.upper);

  return (
    Number.isFinite(benchmark) &&
    Number.isFinite(lower) &&
    Number.isFinite(upper) &&
    benchmark > 0 &&
    lower > 0 &&
    upper >= lower &&
    benchmark >= lower &&
    benchmark <= upper &&
    typeof result.reliability === "string" &&
    result.reliability.trim().length > 0
  );
}

// --------------------------------------------------
// CURRENT-MARKET EVIDENCE FALLBACK
// --------------------------------------------------

async function getMarketFallback(items) {
  const response = await axios.post(
    `${MARKET_EVIDENCE_SERVICE_URL}/v1/market-benchmark`,
    {
      item_name: String(items).trim(),
      currency: "INR",
      unit: "each",
    },
    {
      timeout: 15000,
      headers: {
        "Content-Type": "application/json",
      },
    }
  );

  return response.data;
}

// --------------------------------------------------
// GENERATE TENDER + PRICE BENCHMARK
// --------------------------------------------------

exports.generateTender = async (req, res) => {
  try {
    // --------------------------------------------------
    // NGO AUTHENTICATION
    // --------------------------------------------------

    if (!req.ngo) {
      return res.status(401).json({
        success: false,
        message: "NGO authentication required",
      });
    }

    const {
      procurementType,
      requirements,
      items,
      quantity,
      budget,
      location,
    } = req.body;

    // --------------------------------------------------
    // VALIDATE REQUEST
    // --------------------------------------------------

    if (
      !procurementType ||
      !requirements ||
      !items ||
      quantity === undefined ||
      budget === undefined ||
      !location
    ) {
      return res.status(400).json({
        success: false,
        message:
          "procurementType, requirements, items, quantity, budget and location are required",
      });
    }

    const quantityValue = Number(quantity);
    const budgetValue = Number(budget);

    if (
      !Number.isFinite(quantityValue) ||
      quantityValue <= 0 ||
      !Number.isFinite(budgetValue) ||
      budgetValue < 0 ||
      !String(items).trim()
    ) {
      return res.status(400).json({
        success: false,
        message:
          "Items must be provided, quantity must be positive, and budget must be non-negative",
      });
    }

    // --------------------------------------------------
    // PREPARE REQUEST FOR HISTORICAL ML SERVICE
    // --------------------------------------------------

    const pythonRequest = {
      ITEM_DESCRIPTION: items,
      COMMODITY_DESCRIPTION: procurementType,
      EXTENDED_DESCRIPTION: requirements,

      product_text_normalized:
        `${items} ${requirements}`.trim(),

      COMMODITY: "",
      commodity_code: "",
      commodity_family: "",

      UNIT_OF_MEASURE: "EA",
      uom_standardized: "EA",

      quantity_numeric: quantityValue,

      // Retained from the existing frozen-model integration.
      award_date_parsed: "2024-06-15",

      PURCHASE_ORDER: "CIVICENGAGE-TENDER",
      VENDOR_CODE: "UNKNOWN",

      BRAND_NAME: "",
      MODEL_NUMBER: "",
      PART_NUMBER: "",
      MASTER_AGREEMENT: null,

      PRODUCT_TYPE: procurementType,
    };

    console.log(
      "Sending request to historical ML service:",
      JSON.stringify(pythonRequest, null, 2)
    );

    // --------------------------------------------------
    // TRY HISTORICAL ML PREDICTION FIRST
    // --------------------------------------------------

    let mlResult = null;
    let historicalError = null;

    try {
      const response = await axios.post(
        `${ML_SERVICE_URL}/predict`,
        pythonRequest,
        {
          timeout: 60000,
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      mlResult = response.data;

      console.log(
        "Historical ML response:",
        JSON.stringify(mlResult, null, 2)
      );
    } catch (error) {
      historicalError =
        error.response?.data || error.message;

      console.error(
        "Historical ML service failed:",
        historicalError
      );
    }

    // --------------------------------------------------
    // USE MARKET FALLBACK IF NEEDED
    // --------------------------------------------------

    let predictionSource = "Historical ML service";

    if (!isValidPriceResult(mlResult)) {
      console.warn(
        "Historical ML result unavailable or invalid. Trying current-market evidence."
      );

      try {
        mlResult = await getMarketFallback(items);

        if (!isValidPriceResult(mlResult)) {
          throw new Error(
            "Market-evidence service returned an invalid benchmark or price range."
          );
        }

        predictionSource = "Current-market evidence fallback";

        console.log(
          "Market-evidence response:",
          JSON.stringify(mlResult, null, 2)
        );
      } catch (error) {
        const marketEvidenceError =
          error.response?.data || error.message;

        console.error(
          "Market-evidence fallback failed:",
          marketEvidenceError
        );

        return res.status(422).json({
          success: false,
          message:
            "A valid price benchmark could not be generated. The historical prediction was unavailable or invalid, and the market-evidence fallback could not provide a valid result.",
          historicalError,
          marketEvidenceError,
        });
      }
    }

    // --------------------------------------------------
    // EXTRACT AND VALIDATE FINAL PRICE RESULT
    // --------------------------------------------------

    const benchmarkPrice = Number(
      mlResult.benchmarkUnitPrice
    );

    const recommendedMinPrice = Number(
      mlResult.expectedRange.lower
    );

    const recommendedMaxPrice = Number(
      mlResult.expectedRange.upper
    );

    const reliability =
      `${mlResult.reliability} | Source: ${predictionSource}`;

    // --------------------------------------------------
    // GENERATE TENDER DOCUMENT
    // --------------------------------------------------

    const generatedTender = `
CIVICENGAGE TENDER DOCUMENT
============================

Procurement Type:
${procurementType}

Requirements:
${requirements}

Items:
${items}

Quantity:
${quantityValue}

Budget:
₹${budgetValue}

Location:
${location}

PRICE BENCHMARK
============================

Benchmark Unit Price:
₹${benchmarkPrice}

Recommended Minimum Unit Price:
₹${recommendedMinPrice}

Recommended Maximum Unit Price:
₹${recommendedMaxPrice}

Prediction Reliability:
${reliability}

Prediction Source:
${predictionSource}

Note:
The benchmark is an estimate and should be reviewed before
final procurement decisions are made.

============================
END OF TENDER
============================
`.trim();

    // --------------------------------------------------
    // SAVE TENDER TO MONGODB
    // --------------------------------------------------

    const tender = await Tender.create({
      ngoId: req.ngo._id,

      procurementType,
      requirements,
      items,

      quantity: quantityValue,
      budget: budgetValue,

      location,

      benchmarkPrice,
      recommendedMinPrice,
      recommendedMaxPrice,
      reliability,

      generatedTender,
    });

    // --------------------------------------------------
    // RETURN SUCCESS RESPONSE
    // --------------------------------------------------

    return res.status(201).json({
      success: true,
      message: "Tender generated successfully",

      tender,

      prediction: {
        benchmarkUnitPrice: benchmarkPrice,

        expectedRange: {
          min: recommendedMinPrice,
          max: recommendedMaxPrice,
        },

        reliability,
        source: predictionSource,
      },
    });
  } catch (error) {
    console.error("generateTender Error:", error);

    return res.status(500).json({
      success: false,
      message: "Failed to generate tender",
      error: error.message,
    });
  }
};
