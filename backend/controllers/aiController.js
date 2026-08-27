const axios = require("axios");
const Campaign = require("../models/Campaign");
const User = require("../models/User");
const { resolveLocationCoordinates } = require("../utils/aiSync");

// ============================================================
// AI SERVICE URLS
// ============================================================

const AI_RECOMMENDER_URL =
  process.env.AI_RECOMMENDATION_SERVICE_URL ||
  "http://127.0.0.1:5001";

const AI_VOLUNTEER_SERVICE_URL =
  process.env.AI_VOLUNTEER_MATCHING_SERVICE_URL ||
  "http://127.0.0.1:8000";


// ============================================================
// HELPER: USER → EVENT FALLBACK MATCHING
// ============================================================

function computeUserEventMatch(user, event) {
  let score = 60;

  const userSkills = (user.skills || []).map((s) =>
    String(s).toLowerCase()
  );

  const userInterests = (user.interests || []).map((i) =>
    String(i).toLowerCase()
  );

  const requiredSkills = (event.requiredSkills || []).map((s) =>
    String(s).toLowerCase()
  );

  const category = String(event.category || "").toLowerCase();

  const matchedSkills = [];

  requiredSkills.forEach((reqSkill) => {
    if (
      userSkills.some(
        (userSkill) =>
          userSkill.includes(reqSkill) ||
          reqSkill.includes(userSkill)
      )
    ) {
      score += 15;
      matchedSkills.push(reqSkill);
    }
  });

  if (
    userInterests.some(
      (interest) =>
        category.includes(interest) ||
        interest.includes(category)
    )
  ) {
    score += 15;
  }

  score = Math.min(98, Math.max(65, score));

  const reason = `Matches your skills in ${
    matchedSkills.length > 0
      ? matchedSkills.join(", ")
      : "community service"
  } and location.`;

  return {
    matchScore: score,
    reason,
  };
}


// ============================================================
// HELPER: HAVERSINE DISTANCE
// ============================================================

function haversineDistance(coord1, coord2) {
  if (!coord1 || !coord2) {
    return 5.0;
  }

  const lat1 = coord1.lat;
  const lon1 = coord1.lng;

  const lat2 = coord2.lat;
  const lon2 = coord2.lng;

  if (
    lat1 === undefined ||
    lon1 === undefined ||
    lat2 === undefined ||
    lon2 === undefined
  ) {
    return 5.0;
  }

  const R = 6371.0;

  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);

  const c =
    2 * Math.atan2(
      Math.sqrt(a),
      Math.sqrt(1 - a)
    );

  return Math.round(R * c * 10) / 10;
}


// ============================================================
// HELPER: VOLUNTEER FALLBACK MATCHING
// ============================================================

function computeVolunteerMatch(event, volunteer, eventCoords) {
  let score = 65;

  const volSkills = (volunteer.skills || []).map((s) =>
    String(s).toLowerCase()
  );

  const volInterests = (volunteer.interests || []).map((i) =>
    String(i).toLowerCase()
  );

  const requiredSkills = (event.requiredSkills || []).map((s) =>
    String(s).toLowerCase()
  );

  const category = String(event.category || "").toLowerCase();

  let skillMatchCount = 0;
  const matchedSkillsList = [];

  requiredSkills.forEach((reqSkill) => {
    if (
      volSkills.some(
        (volSkill) =>
          volSkill.includes(reqSkill) ||
          reqSkill.includes(volSkill)
      )
    ) {
      skillMatchCount++;
      matchedSkillsList.push(reqSkill);
    }
  });

  score += skillMatchCount * 12;

  if (
    volInterests.some((interest) =>
      category.includes(interest)
    )
  ) {
    score += 10;
  }

  if (
    String(volunteer.availability || "").toLowerCase() ===
    "available"
  ) {
    score += 5;
  }

  const vCoords =
    volunteer.coordinates ||
    resolveLocationCoordinates(volunteer.location);

  const eCoords =
    eventCoords ||
    resolveLocationCoordinates(
      event.location,
      event.coordinates
    );

  const distKm = haversineDistance(eCoords, vCoords);

  const proximityScore = Math.min(
    99,
    Math.max(
      65,
      Math.round(100 / (1 + distKm / 15))
    )
  );

  score = Math.min(99, Math.max(68, score));

  const breakdown = {
    skills: Math.min(
      98,
      70 + skillMatchCount * 10
    ),

    interests: Math.min(
      95,
      75 + (volInterests.length > 0 ? 10 : 0)
    ),

    availability:
      String(volunteer.availability || "").toLowerCase() ===
      "available"
        ? 95
        : 80,

    proximity: proximityScore,
  };

  const whyRecommended = `Strong match for ${
    event.title || "this event"
  } with verified skills in ${
    matchedSkillsList.length > 0
      ? matchedSkillsList.join(", ")
      : "community outreach"
  }, located ${distKm} km away.`;

  return {
    id: volunteer._id || volunteer.id,

    userId: volunteer._id || volunteer.id,

    name: volunteer.name,

    email: volunteer.email,

    phone: volunteer.phone,

    avatar:
      volunteer.avatar ||
      "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",

    skills: volunteer.skills,

    interests: volunteer.interests,

    availability: volunteer.availability,

    location: volunteer.location || "Manila",

    coordinates: vCoords,

    distanceKm: distKm,

    impactScore:
      volunteer.impactScore ||
      volunteer.social_impact_score ||
      85,

    matchPercent: score,

    breakdown,

    whyRecommended,

    matchedSkills: matchedSkillsList,
  };
}


// ============================================================
// GET AI RECOMMENDED EVENTS FOR USER
// ============================================================
// POST /api/recommendations/events
// ============================================================

exports.getEventRecommendations = async (req, res) => {
  try {
    // --------------------------------------------------------
    // 1. Get user
    // --------------------------------------------------------

    let user = req.user;

    if (!user) {
      const { userId } = req.body;

      if (userId) {
        user = await User.findById(userId);
      }
    }

    // --------------------------------------------------------
    // 2. Fallback demo user
    // --------------------------------------------------------

    if (!user) {
      user = {
        skills: [
          "Teaching",
          "Environmental Science",
          "Communication",
        ],

        interests: [
          "Environment",
          "Education",
        ],

        location: "Manila",

        availability: "weekends",

        coordinates: null,
      };
    }

    // --------------------------------------------------------
    // 3. Get active campaigns
    // --------------------------------------------------------

    const events = await Campaign.find({
      status: "Active",
    }).populate(
      "ngoId",
      "ngoName logo"
    );

    // --------------------------------------------------------
    // 4. Resolve user's coordinates
    // --------------------------------------------------------

    const userCoordinates =
      resolveLocationCoordinates(
        user.location,
        user.coordinates
      );

    // --------------------------------------------------------
    // 5. Send data to Flask AI recommender
    // --------------------------------------------------------

    try {
      console.log(
        "🤖 Calling AI Event Recommendation Service..."
      );

      const response = await axios.post(
        `${AI_RECOMMENDER_URL}/recommend`,
        {
          user: {
            id:
              user._id?.toString() ||
              user.id,

            skills: user.skills || [],

            interests: user.interests || [],

            location: user.location || "",

            coordinates: userCoordinates,

            availability:
              user.availability || "",
          },

          events: events.map((event) => ({
            id: event._id.toString(),

            title: event.title,

            description:
              event.description || "",

            category:
              event.category || "",

            requiredSkills:
              event.requiredSkills || [],

            location:
              event.location || "",

            coordinates:
              resolveLocationCoordinates(
                event.location,
                event.coordinates
              ),

            date: event.date,
          })),
        },

        {
          // Transformer model + Groq can take a little time
          timeout: 15000,
        }
      );

      // ------------------------------------------------------
      // 6. Return AI recommendations
      // ------------------------------------------------------

      if (
        response.data &&
        response.data.recommendations
      ) {
        console.log(
          `✅ AI returned ${response.data.recommendations.length} recommendations`
        );

        return res.status(200).json({
          success: true,

          recommendations:
            response.data.recommendations,
        });
      }

      console.log(
        "⚠️ AI returned no recommendations. Using fallback."
      );
    } catch (aiErr) {
      console.log(
        "⚠️ AI Recommender service unavailable, using algorithm fallback."
      );

      console.log(
        "AI error:",
        aiErr.message
      );
    }

    // --------------------------------------------------------
    // 7. Fallback recommendation algorithm
    // --------------------------------------------------------

    const recommendations = events.map(
      (event) => {
        const match =
          computeUserEventMatch(
            user,
            event
          );

        return {
          ...event.toObject(),

          matchScore:
            match.matchScore,

          reason:
            match.reason,
        };
      }
    );

    // Highest score first
    recommendations.sort(
      (a, b) =>
        b.matchScore -
        a.matchScore
    );

    return res.status(200).json({
      success: true,

      recommendations:
        recommendations.slice(0, 6),
    });
  } catch (error) {
    console.error(
      "❌ getEventRecommendations Error:",
      error
    );

    return res.status(500).json({
      message:
        "Recommendation failed",

      error:
        error.message,
    });
  }
};


// ============================================================
// GET AI RECOMMENDED VOLUNTEERS FOR NGO EVENT
// ============================================================
// POST /api/recommendations/volunteers
// ============================================================

exports.getVolunteerRecommendations = async (
  req,
  res
) => {
  try {
    const { eventId } = req.body;

    // --------------------------------------------------------
    // 1. Validate event ID
    // --------------------------------------------------------

    if (!eventId) {
      return res.status(400).json({
        message: "eventId is required",
      });
    }

    // --------------------------------------------------------
    // 2. Find campaign/event
    // --------------------------------------------------------

    const event =
      await Campaign.findById(eventId);

    if (!event) {
      return res.status(404).json({
        message: "Event not found",
      });
    }

    // --------------------------------------------------------
    // 3. Get volunteers
    // --------------------------------------------------------

    const volunteers =
      await User.find({
        $or: [
          { role: "user" },
          { role: "volunteer" },
          { role: { $exists: false } },
          { role: null },
        ],
      })
        .select("-otp -otpExpires")
        .populate("registeredEvents");

    // --------------------------------------------------------
    // 4. Event coordinates
    // --------------------------------------------------------

    const eventCoords =
      resolveLocationCoordinates(
        event.location,
        event.coordinates
      );

    // --------------------------------------------------------
    // 5. Format volunteers for AI service
    // --------------------------------------------------------

    const formattedVolunteers =
      volunteers.map((volunteer) => {
        const registeredEvents =
          volunteer.registeredEvents || [];

        const activeCount =
          registeredEvents.filter(
            (registeredEvent) =>
              registeredEvent &&
              (
                registeredEvent.status ===
                  "Active" ||
                registeredEvent.status ===
                  "Upcoming"
              )
          ).length;

        const historyIds =
          registeredEvents
            .filter(
              (registeredEvent) =>
                registeredEvent &&
                registeredEvent.status ===
                  "Completed"
            )
            .map((registeredEvent) =>
              registeredEvent._id.toString()
            );

        const volunteerCoords =
          resolveLocationCoordinates(
            volunteer.location,
            volunteer.coordinates
          );

        return {
          id: volunteer._id.toString(),

          _id: volunteer._id.toString(),

          name: volunteer.name,

          email: volunteer.email,

          phone: volunteer.phone,

          avatar: volunteer.avatar,

          skills: volunteer.skills || [],

          interests:
            volunteer.interests || [],

          availability:
            volunteer.availability,

          location:
            volunteer.location,

          coordinates:
            volunteerCoords,

          occupation:
            volunteer.occupation,

          attendance_rate: 0.95,

          social_impact_score:
            volunteer.impactScore || 85,

          impact_score:
            volunteer.impactScore || 85,

          active_campaigns:
            activeCount,

          history_ids:
            historyIds,
        };
      });

    // --------------------------------------------------------
    // 6. Format event for volunteer AI service
    // --------------------------------------------------------

    const eventPayload = {
      id: event._id.toString(),

      _id: event._id.toString(),

      title: event.title,

      description:
        event.description || "",

      category:
        event.category || "General",

      requiredSkills:
        event.requiredSkills || [],

      location:
        event.location,

      coordinates:
        eventCoords,

      date:
        event.date,
    };

    // --------------------------------------------------------
    // 7. Call volunteer matching AI service
    // --------------------------------------------------------

    try {
      console.log(
        "🤖 Calling AI Volunteer Matching Service..."
      );

      const response =
        await axios.post(
          `${AI_VOLUNTEER_SERVICE_URL}/recommend-volunteers`,
          {
            event: eventPayload,

            volunteers:
              formattedVolunteers,
          },
          {
            timeout: 15000,
          }
        );

      if (
        response.data &&
        response.data.volunteers
      ) {
        console.log(
          `✅ AI returned ${response.data.volunteers.length} volunteers`
        );

        return res.status(200).json({
          success: true,

          volunteers:
            response.data.volunteers,
        });
      }
    } catch (aiErr) {
      console.log(
        "⚠️ AI Volunteer Matching service unavailable or timed out."
      );

      console.log(
        "AI volunteer error:",
        aiErr.message
      );

      console.log(
        "Using enhanced algorithm fallback..."
      );
    }

    // --------------------------------------------------------
    // 8. Fallback volunteer matching
    // --------------------------------------------------------

    const scoredVolunteers =
      formattedVolunteers.map(
        (volunteer) =>
          computeVolunteerMatch(
            event,
            volunteer,
            eventCoords
          )
      );

    scoredVolunteers.sort(
      (a, b) =>
        b.matchPercent -
        a.matchPercent
    );

    return res.status(200).json({
      success: true,

      volunteers:
        scoredVolunteers,
    });
  } catch (error) {
    console.error(
      "❌ getVolunteerRecommendations Error:",
      error
    );

    return res.status(500).json({
      message:
        "Volunteer matching failed",

      error:
        error.message,
    });
  }
};


// ============================================================
// AI TENDER GENERATOR
// ============================================================
// POST /api/tender/generate
// ============================================================

exports.generateTender = async (
  req,
  res
) => {
  try {
    const {
      eventType,
      requirements,
      budget,
      items,
      prompt,
    } = req.body;

    // --------------------------------------------------------
    // 1. Validate input
    // --------------------------------------------------------

    if (!eventType || !budget) {
      return res.status(400).json({
        message:
          "eventType and budget are required",
      });
    }

    // --------------------------------------------------------
    // 2. Generate reference number
    // --------------------------------------------------------

    const refNo =
      `RFQ-${new Date().getFullYear()}-${Math.floor(
        1000 + Math.random() * 9000
      )}`;

    const issueDate =
      new Date().toLocaleDateString(
        "en-US",
        {
          year: "numeric",
          month: "long",
          day: "numeric",
        }
      );

    const formattedBudget =
      Number(budget).toLocaleString();

    // --------------------------------------------------------
    // 3. Prepare items
    // --------------------------------------------------------

    let itemsListStr = "";

    if (items) {
      const lines =
        items
          .split("\n")
          .filter(Boolean);

      itemsListStr =
        lines
          .map(
            (item, index) =>
              `  ${index + 1}. ${item.trim()}`
          )
          .join("\n");
    } else {
      itemsListStr = `
  1. Event Coordination & Facilitation Services
  2. Logistics and On-Site Setup Equipment
  3. Educational & Banner Printing Materials
  4. Refreshments & Catering for Volunteers
  5. Emergency Medical & First Aid Standby Kit`;
    }

    // --------------------------------------------------------
    // 4. Generate tender document
    // --------------------------------------------------------

    const documentContent = `
PROCUREMENT / TENDER DOCUMENT
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
Requirements:   ${
      requirements ||
      "Standard NGO event mobilization and community assistance."
    }

${
  prompt
    ? `Special Instructions:
${prompt}
`
    : ""
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. SCOPE OF PROCUREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Invitations are extended to qualified contractors and
service providers to submit sealed quotations for the
execution of goods and services needed for the
${eventType}.

Preferential evaluation will be accorded to vendor
entities with demonstrated NGO or community-level
delivery experience and compliant business documentation.

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
  Payment Terms:            30% Mobilization,
                            70% Final Delivery Certificate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. SUBMISSION DEADLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All tender proposals must be submitted within seven
(7) business days of issuance.

Late submissions will not be evaluated.

Contact Email:  procurement@civicengage.org
Location:       CivicEngage NGO Operations Center,
                Manila, Philippines

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[AI-Generated via CivicEngage Procurement Module · Verified]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`;

    // --------------------------------------------------------
    // 5. Return tender
    // --------------------------------------------------------

    return res.status(200).json({
      success: true,

      refNo,

      issueDate,

      document:
        documentContent,
    });
  } catch (error) {
    console.error(
      "❌ generateTender Error:",
      error
    );

    return res.status(500).json({
      message:
        "Tender generation failed",

      error:
        error.message,
    });
  }
};