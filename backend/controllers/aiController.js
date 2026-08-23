const axios = require("axios");
const Campaign = require("../models/Campaign");
const User = require("../models/User");

const AI_RECOMMENDER_URL = process.env.AI_RECOMMENDATION_SERVICE_URL || "http://127.0.0.1:5001";
const AI_VOLUNTEER_SERVICE_URL = process.env.AI_VOLUNTEER_MATCHING_SERVICE_URL || "http://127.0.0.1:8000";

// Fallback logic for user event recommendations
function computeUserEventMatch(user, event) {
  let score = 60; // base score
  const userSkills = (user.skills || []).map((s) => s.toLowerCase());
  const userInterests = (user.interests || []).map((i) => i.toLowerCase());
  const requiredSkills = (event.requiredSkills || []).map((s) => s.toLowerCase());
  const category = (event.category || "").toLowerCase();

  let matchedSkills = [];
  requiredSkills.forEach((reqSkill) => {
    if (userSkills.some((us) => us.includes(reqSkill) || reqSkill.includes(us))) {
      score += 15;
      matchedSkills.push(reqSkill);
    }
  });

  if (userInterests.some((ui) => category.includes(ui) || ui.includes(category))) {
    score += 15;
  }

  score = Math.min(98, Math.max(65, score));

  let reason = `Matches your skills in ${matchedSkills.length > 0 ? matchedSkills.join(", ") : "community service"} and location.`;
  return { matchScore: score, reason };
}

function haversineDistance(coord1, coord2) {
  if (!coord1 || !coord2) return 5.0;
  const lat1 = coord1.lat;
  const lon1 = coord1.lng;
  const lat2 = coord2.lat;
  const lon2 = coord2.lng;
  if (lat1 === undefined || lon1 === undefined || lat2 === undefined || lon2 === undefined) return 5.0;

  const R = 6371.0;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return Math.round(R * c * 10) / 10;
}

// Fallback logic for volunteer matching
function computeVolunteerMatch(event, volunteer, eventCoords) {
  let score = 65;
  const volSkills = (volunteer.skills || []).map((s) => s.toLowerCase());
  const volInterests = (volunteer.interests || []).map((i) => i.toLowerCase());
  const requiredSkills = (event.requiredSkills || []).map((s) => s.toLowerCase());
  const category = (event.category || "").toLowerCase();

  let skillMatchCount = 0;
  let matchedSkillsList = [];

  requiredSkills.forEach((reqSkill) => {
    if (volSkills.some((vs) => vs.includes(reqSkill) || reqSkill.includes(vs))) {
      skillMatchCount++;
      matchedSkillsList.push(reqSkill);
    }
  });

  score += skillMatchCount * 12;
  if (volInterests.some((vi) => category.includes(vi))) {
    score += 10;
  }

  if (volunteer.availability === "Available") score += 5;

  const vCoords = volunteer.coordinates || resolveLocationCoordinates(volunteer.location);
  const eCoords = eventCoords || resolveLocationCoordinates(event.location, event.coordinates);
  const distKm = haversineDistance(eCoords, vCoords);
  const proximityScore = Math.min(99, Math.max(65, Math.round(100 / (1 + distKm / 15))));

  score = Math.min(99, Math.max(68, score));

  const breakdown = {
    skills: Math.min(98, 70 + skillMatchCount * 10),
    interests: Math.min(95, 75 + (volInterests.length > 0 ? 10 : 0)),
    availability: volunteer.availability === "Available" ? 95 : 80,
    proximity: proximityScore,
  };

  const whyRecommended = `Strong match for ${event.title || "this event"} with verified skills in ${
    matchedSkillsList.length > 0 ? matchedSkillsList.join(", ") : "community outreach"
  }, located ${distKm} km away.`;

  return {
    id: volunteer._id || volunteer.id,
    userId: volunteer._id || volunteer.id,
    name: volunteer.name,
    email: volunteer.email,
    phone: volunteer.phone,
    avatar: volunteer.avatar || "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
    skills: volunteer.skills,
    interests: volunteer.interests,
    availability: volunteer.availability,
    location: volunteer.location || "Manila",
    coordinates: vCoords,
    distanceKm: distKm,
    impactScore: volunteer.impactScore || volunteer.social_impact_score || 85,
    matchPercent: score,
    breakdown,
    whyRecommended,
    matchedSkills: matchedSkillsList,
  };
}


// @desc Get AI Recommended Events for User
// @route POST /api/recommendations/events
exports.getEventRecommendations = async (req, res) => {
  try {
    let user = req.user;
    if (!user) {
      const { userId } = req.body;
      if (userId) {
        user = await User.findById(userId);
      }
    }

    if (!user) {
      user = {
        skills: ["Teaching", "Environmental Science", "Communication"],
        interests: ["Environment", "Education"],
        location: "Manila",
      };
    }

    const events = await Campaign.find({ status: "Active" }).populate("ngoId", "ngoName logo");

    try {
      // Try calling FastAPI / Flask AI service
      const response = await axios.post(
        `${AI_RECOMMENDER_URL}/recommend`,
        {
          user: {
            skills: user.skills,
            interests: user.interests,
            location: user.location,
          },
          events: events.map((e) => ({
            id: e._id.toString(),
            title: e.title,
            category: e.category,
            requiredSkills: e.requiredSkills,
            location: e.location,
          })),
        },
        { timeout: 2500 }
      );

      if (response.data && response.data.recommendations) {
        return res.status(200).json({ success: true, recommendations: response.data.recommendations });
      }
    } catch (aiErr) {
      console.log("AI Recommender service unavailable, using algorithm fallback.");
    }

    // Fallback: Calculate match scores dynamically using event & user data
    const recommendations = events.map((event) => {
      const match = computeUserEventMatch(user, event);
      return {
        ...event.toObject(),
        matchScore: match.matchScore,
        reason: match.reason,
      };
    });

    recommendations.sort((a, b) => b.matchScore - a.matchScore);

    return res.status(200).json({ success: true, recommendations: recommendations.slice(0, 6) });
  } catch (error) {
    console.error("getEventRecommendations Error:", error);
    return res.status(500).json({ message: "Recommendation failed", error: error.message });
  }
};

const { resolveLocationCoordinates } = require("../utils/aiSync");


// @desc Get AI Recommended Volunteers for NGO Event
// @route POST /api/recommendations/volunteers
exports.getVolunteerRecommendations = async (req, res) => {
  try {
    const { eventId } = req.body;
    if (!eventId) {
      return res.status(400).json({ message: "eventId is required" });
    }

    const event = await Campaign.findById(eventId);
    if (!event) {
      return res.status(404).json({ message: "Event not found" });
    }

    // Populate registeredEvents to determine active campaigns and completed history
    const volunteers = await User.find({
      $or: [{ role: "user" }, { role: "volunteer" }, { role: { $exists: false } }, { role: null }],
    })
      .select("-otp -otpExpires")
      .populate("registeredEvents");


    const eventCoords = resolveLocationCoordinates(event.location, event.coordinates);

    const formattedVolunteers = volunteers.map((v) => {
      const regEvents = v.registeredEvents || [];
      const activeCount = regEvents.filter(
        (e) => e && (e.status === "Active" || e.status === "Upcoming")
      ).length;
      const historyIds = regEvents
        .filter((e) => e && e.status === "Completed")
        .map((e) => e._id.toString());
      const vCoords = resolveLocationCoordinates(v.location, v.coordinates);

      return {
        id: v._id.toString(),
        _id: v._id.toString(),
        name: v.name,
        email: v.email,
        phone: v.phone,
        avatar: v.avatar,
        skills: v.skills,
        interests: v.interests,
        availability: v.availability,
        location: v.location,
        coordinates: vCoords,
        occupation: v.occupation,
        attendance_rate: 0.95,
        social_impact_score: v.impactScore || 85,
        impact_score: v.impactScore || 85,
        active_campaigns: activeCount,
        history_ids: historyIds,
      };
    });

    const eventPayload = {
      id: event._id.toString(),
      _id: event._id.toString(),
      title: event.title,
      description: event.description || "",
      category: event.category || "General",
      requiredSkills: event.requiredSkills || [],
      location: event.location,
      coordinates: eventCoords,
      date: event.date,
    };

    try {
      const response = await axios.post(
        `${AI_VOLUNTEER_SERVICE_URL}/recommend-volunteers`,
        {
          event: eventPayload,
          volunteers: formattedVolunteers,
        },
        { timeout: 15000 }
      );

      if (response.data && response.data.volunteers) {
        return res.status(200).json({ success: true, volunteers: response.data.volunteers });
      }
    } catch (aiErr) {
      console.log("AI Volunteer Matching service unavailable or timed out, using enhanced algorithm fallback:", aiErr.message);
    }

    // Fallback volunteer matching scoring
    const scoredVolunteers = formattedVolunteers.map((v) => computeVolunteerMatch(event, v, eventCoords));
    scoredVolunteers.sort((a, b) => b.matchPercent - a.matchPercent);

    return res.status(200).json({ success: true, volunteers: scoredVolunteers });

  } catch (error) {
    console.error("getVolunteerRecommendations Error:", error);
    return res.status(500).json({ message: "Volunteer matching failed", error: error.message });
  }
};

// @desc AI Tender Generator
// @route POST /api/tender/generate
exports.generateTender = async (req, res) => {
  try {
    const { eventType, requirements, budget, items, prompt } = req.body;
    if (!eventType || !budget) {
      return res.status(400).json({ message: "eventType and budget are required" });
    }

    const refNo = `RFQ-${new Date().getFullYear()}-${Math.floor(1000 + Math.random() * 9000)}`;
    const issueDate = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
    const formattedBudget = Number(budget).toLocaleString();

    let itemsListStr = "";
    if (items) {
      const lines = items.split("\n").filter(Boolean);
      itemsListStr = lines.map((it, idx) => `  ${idx + 1}. ${it.trim()}`).join("\n");
    } else {
      itemsListStr = `  1. Event Coordination & Facilitation Services\n  2. Logistics and On-Site Setup Equipment\n  3. Educational & Banner Printing Materials\n  4. Refreshments & Catering for Volunteers\n  5. Emergency Medical & First Aid Standby Kit`;
    }

    const documentContent = `PROCUREMENT / TENDER DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORGANIZATION:   CivicEngage NGO Network
DOCUMENT TYPE:  Request for Quotation (RFQ)
REFERENCE NO.:  ${refNo}
DATE ISSUED:    ${issueDate}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EVENT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Event Type:     ${eventType}
Total Budget:   PHP ${formattedBudget}
Requirements:   ${requirements || "Standard NGO event mobilization and community assistance."}

${prompt ? `Special Instructions:\n${prompt}\n` : ""}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. SCOPE OF PROCUREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Invitations are extended to qualified contractors and service providers to submit 
sealed quotations for the execution of goods and services needed for the ${eventType}.

Preferential evaluation will be accorded to vendor entities with demonstrated NGO or
community-level delivery experience and compliant business documentation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. ITEMS / SERVICES REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

${itemsListStr}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. QUOTATION REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interested suppliers must provide:
  a) Itemized cost proposal breakdown
  b) Valid Tax Identification Number (TIN) / BIR Certificate
  c) Business License / Mayor's Permit
  d) Corporate profile and verifiable past project references

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. BUDGET ALLOCATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Total Approved Budget:    PHP ${formattedBudget}
  Tax Inclusions:           VAT Inclusive
  Payment Terms:            30% Mobilization, 70% Final Delivery Certificate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. SUBMISSION DEADLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All tender proposals must be submitted within seven (7) business days of issuance.
Late submissions will not be evaluated.

Contact Email:  procurement@civicengage.org
Location:       CivicEngage NGO Operations Center, Manila, Philippines

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[AI-Generated via CivicEngage Procurement Module · Verified]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;

    return res.status(200).json({
      success: true,
      refNo,
      issueDate,
      document: documentContent,
    });
  } catch (error) {
    console.error("generateTender Error:", error);
    return res.status(500).json({ message: "Tender generation failed", error: error.message });
  }
};
