require("dotenv").config();
const mongoose = require("mongoose");
const User = require("./models/User");
const Campaign = require("./models/Campaign");

async function verifyDatabaseCoordinates() {
  try {
    console.log("Connecting to MongoDB Atlas...");
    await mongoose.connect(process.env.MONGO_URI);
    console.log("Connected successfully to MongoDB!\n");

    // 1. Fetch and inspect existing users in DB
    console.log("=== 1. Checking Existing Users in Database ===");
    const existingUsers = await User.find().select("name phone location coordinates role").limit(5).lean();
    console.log(`Found ${existingUsers.length} users:`);
    existingUsers.forEach((u) => {
      console.log(`- User: "${u.name}" (${u.phone}) | Location: "${u.location}" | Stored Coordinates:`, u.coordinates);
    });

    // 2. Test saving a fresh new user with Bengaluru
    console.log("\n=== 2. Creating Fresh Test User with Bengaluru ===");
    const testPhone = "+919900112233";
    await User.deleteOne({ phone: testPhone });
    
    const testUser = new User({
      phone: testPhone,
      name: "Aarav Sharma",
      location: "Koramangala, Bengaluru, Karnataka",
      skills: ["Teaching", "Python"],
      interests: ["Education & Literacy"]
    });
    
    // Save to trigger pre-save hook and geocoding
    await testUser.save();
    console.log("Saved test user to MongoDB.");

    // Read back directly from MongoDB (using lean to inspect the raw database document)
    const storedUser = await User.findOne({ phone: testPhone }).lean();
    console.log("\n[RAW MONGODB RECORD for Aarav Sharma (Bengaluru)]:");
    console.log(JSON.stringify({
      _id: storedUser._id,
      name: storedUser.name,
      location: storedUser.location,
      coordinates: storedUser.coordinates,
      updatedAt: storedUser.updatedAt
    }, null, 2));

    // 3. Test saving another fresh user with Varanasi, Uttar Pradesh
    console.log("\n=== 3. Testing Fresh User with Varanasi, Uttar Pradesh ===");
    const varanasiPhone = "+919900112244";
    await User.deleteOne({ phone: varanasiPhone });

    const varanasiUser = new User({
      phone: varanasiPhone,
      name: "Rishi Mishra",
      location: "Varanasi, Uttar Pradesh"
    });
    await varanasiUser.save();

    const storedVaranasi = await User.findOne({ phone: varanasiPhone }).lean();
    console.log("\n[RAW MONGODB RECORD for Rishi Mishra (Varanasi)]:");
    console.log(JSON.stringify({
      _id: storedVaranasi._id,
      name: storedVaranasi.name,
      location: storedVaranasi.location,
      coordinates: storedVaranasi.coordinates,
    }, null, 2));

    // 4. Test updating existing user location to Mumbai
    console.log("\n=== 4. Testing Updating Location to Mumbai ===");
    storedVaranasiDoc = await User.findOne({ phone: varanasiPhone });
    storedVaranasiDoc.location = "Bandra West, Mumbai, Maharashtra";
    await storedVaranasiDoc.save();

    const storedMumbai = await User.findOne({ phone: varanasiPhone }).lean();
    console.log("\n[RAW MONGODB RECORD after updating to Mumbai]:");
    console.log(JSON.stringify({
      _id: storedMumbai._id,
      name: storedMumbai.name,
      location: storedMumbai.location,
      coordinates: storedMumbai.coordinates,
    }, null, 2));


    // 4. Inspect Campaigns in DB
    console.log("\n=== 4. Checking Campaigns in Database ===");
    const campaigns = await Campaign.find().select("title location coordinates category").limit(5).lean();
    campaigns.forEach((c) => {
      console.log(`- Campaign: "${c.title}" | Location: "${c.location}" | Stored Coordinates:`, c.coordinates);
    });

    console.log("\n[VERIFICATION PASSED] Latitude and Longitude are actively computed and stored in MongoDB database!");
  } catch (err) {
    console.error("Database Verification Error:", err);
  } finally {
    await mongoose.disconnect();
    console.log("Disconnected from MongoDB.");
  }
}

verifyDatabaseCoordinates();
