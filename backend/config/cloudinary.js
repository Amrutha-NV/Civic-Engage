const cloudinary = require("cloudinary").v2;

const configureCloudinary = () => {
  const rawCloudName = process.env.CLOUDINARY_CLOUD_NAME || "";
  const rawApiKey = process.env.CLOUDINARY_API_KEY || "";
  const rawApiSecret = process.env.CLOUDINARY_API_SECRET || "";

  const cloudName = rawCloudName.replace(/["'\s]/g, "");
  const apiKey = rawApiKey.replace(/["'\s]/g, "");
  const apiSecret = rawApiSecret.replace(/["'\s]/g, "");

  if (cloudName && apiKey && apiSecret) {
    cloudinary.config({
      cloud_name: cloudName,
      api_key: apiKey,
      api_secret: apiSecret,
    });
    console.log(`🌩️ Cloudinary initialized with Cloud Name: [${cloudName}]`);
  } else {
    console.warn("⚠️ Cloudinary credentials missing or incomplete in environment variables.");
  }
};

module.exports = { cloudinary, configureCloudinary };
