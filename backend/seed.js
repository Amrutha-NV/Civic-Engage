
const path = require("path");
const mongoose = require("mongoose");
const dotenv = require("dotenv");
const bcrypt = require("bcryptjs");

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });
dotenv.config({ path: path.resolve(__dirname, ".env") });


const User = require("./models/User");
const NGO = require("./models/NGO");
const Campaign = require("./models/Campaign");

const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log("Database connected for seeding...");
  } catch (err) {
    console.error("Database connection failed for seeding:", err.message);
    process.exit(1);
  }
};

const seedData = async () => {
  await connectDB();

  try {
    // Clear existing data
    await User.deleteMany({});
    await NGO.deleteMany({});
    await Campaign.deleteMany({});
    console.log("Cleared existing collection data.");

    // Password hashing for NGOs
    const hashedPassword = await bcrypt.hash("password123", 10);

    // 1. Create NGOs
    const ngos = await NGO.create([
      {
        ngoName: "GreenFuture Philippines Foundation",
        email: "contact@greenfutureph.org",
        phone: "+63 2 8123 4567",
        address: "12F Ayala Avenue, Makati City, Metro Manila",
        password: hashedPassword,
        description: "Leading environmental conservation NGO dedicated to coastal cleanups, reforestation, and urban greening.",
        category: "Environmental Protection",
        logo: "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=300",
      },
      {
        ngoName: "EduReach Philippines Initiative",
        email: "info@edureach.org.ph",
        phone: "+63 2 8987 6543",
        address: "45 Katipunan Avenue, Quezon City",
        password: hashedPassword,
        description: "Empowering underprivileged children through digital literacy, after-school tutoring, and book donation drives.",
        category: "Education & Literacy",
        logo: "https://images.unsplash.com/photo-1509062522246-3755977927d7?w=300",
      },
      {
        ngoName: "MetroHealth Community Care",
        email: "support@metrohealth.org",
        phone: "+63 2 8765 4321",
        address: "78 Taft Avenue, Manila",
        password: hashedPassword,
        description: "Providing free medical missions, health awareness workshops, and first-aid response training across Metro Manila.",
        category: "Healthcare & Wellness",
        logo: "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=300",
      },
    ]);
    console.log(`Seeded ${ngos.length} NGOs.`);

    // 2. Create Users / Volunteers
    const users = await User.create([
      {
        phone: "+639171234567",
        name: "Maria Santos",
        email: "maria.santos@gmail.com",
        avatar: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150",
        skills: ["Environmental Science", "Event Planning", "Community Outreach"],
        interests: ["Environmental Protection", "Recycling", "Tree Planting"],
        availability: "Available",
        location: "Marikina City, Metro Manila",
        occupation: "Environmental Researcher",
        impactScore: 94,
        role: "user",
      },
      {
        phone: "+639189876543",
        name: "Juan Dela Cruz",
        email: "juan.delacruz@yahoo.com",
        avatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150",
        skills: ["Teaching", "Mathematics", "Communication"],
        interests: ["Education & Literacy", "Youth Mentorship"],
        availability: "Weekends",
        location: "Quezon City, Metro Manila",
        occupation: "High School Teacher",
        impactScore: 88,
        role: "user",
      },
      {
        phone: "+639195551234",
        name: "Elena Reyes, RN",
        email: "elena.reyes@healthnet.ph",
        avatar: "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=150",
        skills: ["Healthcare", "First Aid", "Patient Care", "Nutrition"],
        interests: ["Healthcare & Wellness", "Disaster Relief"],
        availability: "Flexible",
        location: "Manila City",
        occupation: "Registered Nurse",
        impactScore: 92,
        role: "user",
      },
      {
        phone: "+639201112233",
        name: "Alex Rivera",
        email: "alex.rivera@techcorp.io",
        avatar: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150",
        skills: ["Technology", "Graphic Design", "Social Media", "Marketing"],
        interests: ["Digital Literacy", "Social Awareness"],
        availability: "Available",
        location: "Makati City",
        occupation: "Software Engineer",
        impactScore: 85,
        role: "user",
      },
      {
        phone: "+639223334455",
        name: "Sarah Lim",
        email: "sarah.lim@outlook.com",
        avatar: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
        skills: ["Writing", "Photography", "Event Planning"],
        interests: ["Community Development", "Environmental Protection"],
        availability: "Weekdays",
        location: "Pasig City",
        occupation: "Content Creator",
        impactScore: 80,
        role: "user",
      },
    ]);
    console.log(`Seeded ${users.length} Users/Volunteers.`);

    // 3. Create Events / Campaigns
    const campaigns = await Campaign.create([
      {
        ngoId: ngos[0]._id, // GreenFuture
        title: "Marikina Watershed Reforestation Drive 2026",
        description: "Join us in planting 1,000 native tree saplings to prevent soil erosion and restore the Marikina River watershed ecosystem. Tools, gloves, and lunch will be provided to all volunteers.",
        shortDesc: "Plant 1,000 tree saplings along the Marikina River watershed to mitigate flooding.",
        category: "Environmental Protection",
        requiredSkills: ["Environmental Science", "Event Planning", "Community Outreach"],
        location: "Marikina Watershed Reserve, Marikina City",
        date: "Saturday, Sep 20, 2026",
        startTime: "07:00 AM",
        endTime: "01:00 PM",
        banner: "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=800",
        capacity: 100,
        volunteersJoined: 34,
        status: "Active",
        registeredUsers: [users[0]._id],
      },
      {
        ngoId: ngos[1]._id, // EduReach
        title: "Digital Literacy & Coding Workshop for Kids",
        description: "Help teach basic computer skills, Scratch programming, and internet safety to public school students in Quezon City. Mentors will be assigned 2 students each.",
        shortDesc: "Teach basic digital literacy and programming logic to elementary school students.",
        category: "Education & Literacy",
        requiredSkills: ["Teaching", "Technology", "Communication"],
        location: "Quezon City Public Library, QC",
        date: "Saturday, Sep 27, 2026",
        startTime: "09:00 AM",
        endTime: "03:00 PM",
        banner: "https://images.unsplash.com/photo-1509062522246-3755977927d7?w=800",
        capacity: 40,
        volunteersJoined: 18,
        status: "Active",
        registeredUsers: [users[1]._id, users[3]._id],
      },
      {
        ngoId: ngos[2]._id, // MetroHealth
        title: "Community First-Aid & Wellness Health Fair",
        description: "Providing free health checkups, blood pressure monitoring, dental triage, and basic first-aid training for barangay residents.",
        shortDesc: "Medical triage, blood pressure screening, and wellness education for local barangay.",
        category: "Healthcare & Wellness",
        requiredSkills: ["Healthcare", "First Aid", "Patient Care"],
        location: "Taft Avenue Barangay Hall, Manila",
        date: "Sunday, Oct 04, 2026",
        startTime: "08:00 AM",
        endTime: "04:00 PM",
        banner: "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800",
        capacity: 60,
        volunteersJoined: 25,
        status: "Active",
        registeredUsers: [users[2]._id],
      },
      {
        ngoId: ngos[0]._id, // GreenFuture
        title: "Manila Bay Coastal Clean-up Initiative",
        description: "Gathering volunteers to clear non-biodegradable plastic waste along the coastal shoreline and record marine debris metrics.",
        shortDesc: "Coastal plastic cleanup and microplastic data recording initiative.",
        category: "Environmental Protection",
        requiredSkills: ["Environmental Science", "Writing", "Photography"],
        location: "Baywalk, Roxas Boulevard, Manila",
        date: "Saturday, Oct 11, 2026",
        startTime: "06:00 AM",
        endTime: "11:00 AM",
        banner: "https://images.unsplash.com/photo-1621451537084-482c73073a0f?w=800",
        capacity: 150,
        volunteersJoined: 62,
        status: "Active",
        registeredUsers: [users[0]._id, users[4]._id],
      },
    ]);
    console.log(`Seeded ${campaigns.length} Events/Campaigns.`);

    // Update user registeredEvents refs
    users[0].registeredEvents.push(campaigns[0]._id, campaigns[3]._id);
    users[1].registeredEvents.push(campaigns[1]._id);
    users[2].registeredEvents.push(campaigns[2]._id);
    users[3].registeredEvents.push(campaigns[1]._id);
    users[4].registeredEvents.push(campaigns[3]._id);

    await Promise.all(users.map((u) => u.save()));
    console.log("Updated user registered event references.");

    console.log("✅ Seed database process completed successfully!");
    process.exit(0);
  } catch (error) {
    console.error("Seeding Error:", error);
    process.exit(1);
  }
};

seedData();
