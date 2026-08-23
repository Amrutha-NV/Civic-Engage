const axios = require("axios");
const { resolveCoordinatesSync, resolveCoordinatesAsync, ALL_LOCATIONS } = require("./geocoderService");

const AI_VOLUNTEER_SERVICE_URL =
  process.env.AI_VOLUNTEER_MATCHING_SERVICE_URL || "http://127.0.0.1:8000";

const CITY_COORDINATES = ALL_LOCATIONS;

function resolveLocationCoordinates(locationStr, explicitCoords) {
  return resolveCoordinatesSync(locationStr, explicitCoords);
}

const resolveLocationCoordinatesAsync = async (locationStr, explicitCoords) => {
  return await resolveCoordinatesAsync(locationStr, explicitCoords);
};

/**
 * Syncs a volunteer's profile to the AI Volunteer Matching vector store
 */
const syncVolunteerToAIService = async (user) => {
  try {
    const coords = user.coordinates && user.coordinates.lat
      ? user.coordinates
      : resolveCoordinatesSync(user.location);

    const payload = {
      id: user._id.toString(),
      _id: user._id.toString(),
      name: user.name || "Volunteer",
      email: user.email || "",
      phone: user.phone || "",
      skills: user.skills || [],
      interests: user.interests || [],
      availability: user.availability || "Available",
      location: user.location || "Delhi, India",
      coordinates: coords,
      occupation: user.occupation || "Volunteer",
      impact_score: user.impactScore || 85,
    };

    const response = await axios.post(
      `${AI_VOLUNTEER_SERVICE_URL}/embedding/sync`,
      payload,
      { timeout: 4000 }
    );
    console.log(`[AI Sync] Successfully synced volunteer ${user._id} (${user.name}) to AI service:`, response.data?.detail || "OK");
    return true;
  } catch (err) {
    console.log(`[AI Sync Notice] Could not sync volunteer ${user._id} to AI service: ${err.message}`);
    return false;
  }
};

module.exports = {
  syncVolunteerToAIService,
  resolveLocationCoordinates,
  resolveLocationCoordinatesAsync,
  CITY_COORDINATES,
};

