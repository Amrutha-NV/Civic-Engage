const axios = require("axios");
const { INDIA_LOCATIONS } = require("./indiaLocations");

// In-Memory Geocoding Cache for instant response
const geocodingCache = new Map();

// Default Country & Global Fallbacks
const GLOBAL_LOCATIONS = {
  // Metro Manila & Philippines
  "manila": { lat: 14.5995, lng: 120.9842 },
  "quezon": { lat: 14.6760, lng: 121.0437 },
  "makati": { lat: 14.5547, lng: 121.0244 },
  "taguig": { lat: 14.5176, lng: 121.0509 },
  "bgc": { lat: 14.5507, lng: 121.0519 },
  "pasig": { lat: 14.5764, lng: 121.0851 },
  "cebu": { lat: 10.3157, lng: 123.8854 },
  "davao": { lat: 7.1907, lng: 125.4553 },
  
  // Global Hubs
  "singapore": { lat: 1.3521, lng: 103.8198 },
  "tokyo": { lat: 35.6762, lng: 139.6503 },
  "newyork": { lat: 40.7128, lng: -74.0060 },
  "sanfrancisco": { lat: 37.7749, lng: -122.4194 },
  "london": { lat: 51.5074, lng: -0.1278 },
};

// Combined offline dictionary
const ALL_LOCATIONS = {
  ...INDIA_LOCATIONS,
  ...GLOBAL_LOCATIONS,
};

const INDIAN_STATE_KEYS = new Set([
  "andhrapradesh", "arunachalpradesh", "assam", "bihar", "chhattisgarh", "gujarat", "haryana",
  "himachalpradesh", "jharkhand", "karnataka", "kerala", "madhyapradesh", "maharashtra", "manipur",
  "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab", "rajasthan", "sikkim", "tamilnadu",
  "telangana", "tripura", "uttarpradesh", "uttarakhand", "westbengal"
]);

// Sorted keys
const CITY_LOCALITY_KEYS = Object.keys(ALL_LOCATIONS)
  .filter((k) => !INDIAN_STATE_KEYS.has(k) && k !== "india" && k !== "manila")
  .sort((a, b) => b.length - a.length);

const STATE_KEYS = Array.from(INDIAN_STATE_KEYS).sort((a, b) => b.length - a.length);

/**
 * Fast synchronous lookup against our in-memory 150+ Indian & Global locations dictionary.
 */
function resolveCoordinatesSync(locationStr, explicitCoords) {
  // 1. If locationStr is specified, prioritize matching it against the city/state coordinate dictionary
  if (locationStr && typeof locationStr === "string" && locationStr.trim().length > 0) {
    const cleanStr = locationStr.toLowerCase().replace(/[^a-z0-9]/g, "");

    // Check in-memory dynamic cache first
    if (geocodingCache.has(cleanStr)) {
      return geocodingCache.get(cleanStr);
    }

    // Pass 1: Match specific cities and localities first
    for (const key of CITY_LOCALITY_KEYS) {
      const cleanKey = key.replace(/[^a-z0-9]/g, "");
      if (cleanStr.includes(cleanKey)) {
        const result = { lat: ALL_LOCATIONS[key].lat, lng: ALL_LOCATIONS[key].lng };
        geocodingCache.set(cleanStr, result);
        return result;
      }
    }

    // Pass 2: Match state centroids if no city matched
    for (const key of STATE_KEYS) {
      const cleanKey = key.replace(/[^a-z0-9]/g, "");
      if (cleanStr.includes(cleanKey)) {
        const result = { lat: ALL_LOCATIONS[key].lat, lng: ALL_LOCATIONS[key].lng };
        geocodingCache.set(cleanStr, result);
        return result;
      }
    }

    // Pass 3: Global / Country Fallback
    if (cleanStr.includes("manila") || cleanStr.includes("philippines")) {
      return { lat: 14.5995, lng: 120.9842 };
    }
  }

  // 2. If explicit coordinates are provided, use them
  if (
    explicitCoords &&
    typeof explicitCoords === "object" &&
    typeof explicitCoords.lat === "number" &&
    typeof explicitCoords.lng === "number" &&
    !isNaN(explicitCoords.lat) &&
    !isNaN(explicitCoords.lng)
  ) {
    return { lat: Number(explicitCoords.lat), lng: Number(explicitCoords.lng) };
  }

  // 3. Fallback default
  return { lat: 28.6139, lng: 77.2090 };
}


/**
 * Dynamic asynchronous geocoding using OpenStreetMap Nominatim with offline fallback.
 * Runs during signup and profile update to resolve any specific locality, street, or PIN code in India.
 */
async function resolveCoordinatesAsync(locationStr, explicitCoords) {
  if (
    explicitCoords &&
    typeof explicitCoords === "object" &&
    typeof explicitCoords.lat === "number" &&
    typeof explicitCoords.lng === "number" &&
    !isNaN(explicitCoords.lat) &&
    !isNaN(explicitCoords.lng)
  ) {
    return { lat: Number(explicitCoords.lat), lng: Number(explicitCoords.lng) };
  }

  if (!locationStr || typeof locationStr !== "string" || locationStr.trim().length === 0) {
    return { lat: 28.6139, lng: 77.2090 };
  }

  const cleanStr = locationStr.toLowerCase().replace(/[^a-z0-9]/g, "");

  // 1. Check Cache
  if (geocodingCache.has(cleanStr)) {
    return geocodingCache.get(cleanStr);
  }

  // 2. Check Offline Specific Cities / Localities first (0ms)
  for (const key of CITY_LOCALITY_KEYS) {
    const cleanKey = key.replace(/[^a-z0-9]/g, "");
    if (cleanStr.includes(cleanKey)) {
      const result = { lat: ALL_LOCATIONS[key].lat, lng: ALL_LOCATIONS[key].lng };
      geocodingCache.set(cleanStr, result);
      return result;
    }
  }

  // 3. Dynamic Online Geocoder via OpenStreetMap Nominatim for specific Indian addresses/PIN codes
  try {
    const query = locationStr.toLowerCase().includes("india") ? locationStr : `${locationStr}, India`;
    const response = await axios.get("https://nominatim.openstreetmap.org/search", {
      params: {
        q: query,
        format: "json",
        limit: 1,
        addressdetails: 1,
      },
      headers: {
        "User-Agent": "CivicEngage-Geocoding-Service/1.0 (info@civicengage.org)",
      },
      timeout: 2500,
    });

    if (response.data && response.data.length > 0) {
      const match = response.data[0];
      const result = {
        lat: parseFloat(match.lat),
        lng: parseFloat(match.lon),
      };
      console.log(`[Geocoder Service] Online lookup resolved "${locationStr}" -> Lat: ${result.lat}, Lng: ${result.lng} (${match.display_name?.substring(0, 50)}...)`);
      geocodingCache.set(cleanStr, result);
      return result;
    }
  } catch (geoErr) {
    console.log(`[Geocoder Service Notice] Online lookup skipped for "${locationStr}": ${geoErr.message}`);
  }

  // 4. Default Fallback to State or Delhi
  return resolveCoordinatesSync(locationStr, explicitCoords);
}


module.exports = {
  resolveCoordinatesSync,
  resolveCoordinatesAsync,
  ALL_LOCATIONS,
  INDIA_LOCATIONS,
};
