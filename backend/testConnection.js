const mongoose = require('mongoose');

// Connects entirely inside your computer—no internet or cloud required!
const LOCAL_MONGO_URI = "mongodb://127.0.0.1:27017/ngo_database";

console.log("Connecting to your local MongoDB database...");

mongoose.connect(LOCAL_MONGO_URI)
  .then(() => {
    console.log("🎉 SUCCESS! Connected to your local MongoDB perfectly!");
    process.exit(0);
  })
  .catch((err) => {
    console.error("❌ Local connection failed:", err.message);
    process.exit(1);
  });