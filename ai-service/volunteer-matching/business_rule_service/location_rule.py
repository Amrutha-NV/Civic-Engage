from __future__ import annotations

import math
from typing import List, Optional, Tuple, Dict, Any

from models.candidate import Candidate
from models.campaign import Campaign
from models.volunteer import Volunteer
from business_rule_service.rule_engine import BusinessRule


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculates the great-circle distance between two points on Earth in kilometers using Haversine formula.
    coord = (lat, lng)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371.0  # Earth's radius in kilometers

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


CITY_COORDINATES: Dict[str, Tuple[float, float]] = {
    # --- India Metros & Tech Hubs ---
    "delhi": (28.6139, 77.2090),
    "newdelhi": (28.6139, 77.2090),
    "ncr": (28.6139, 77.2090),
    "noida": (28.5355, 77.3910),
    "greaternoida": (28.4744, 77.5040),
    "gurgaon": (28.4595, 77.0266),
    "gurugram": (28.4595, 77.0266),
    "faridabad": (28.4089, 77.3178),
    "ghaziabad": (28.6692, 77.4538),

    "mumbai": (19.0760, 72.8777),
    "navimumbai": (19.0330, 73.0297),
    "thane": (19.2183, 72.9781),
    "pune": (18.5204, 73.8567),
    "nagpur": (21.1458, 79.0882),
    "nashik": (19.9975, 73.7898),
    "aurangabad": (19.8762, 75.3433),
    "chhatrapatisambhajinagar": (19.8762, 75.3433),
    "solapur": (17.6599, 75.9064),
    "kolhapur": (16.7050, 74.2433),

    "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "mysore": (12.2958, 76.6394),
    "mysuru": (12.2958, 76.6394),
    "mangalore": (12.9141, 74.8560),
    "mangaluru": (12.9141, 74.8560),
    "hubli": (15.3647, 75.1240),
    "dharwad": (15.4589, 75.0078),
    "belgaum": (15.8497, 74.4977),
    "udupi": (13.3409, 74.7421),

    "hyderabad": (17.3850, 78.4867),
    "secunderabad": (17.4399, 78.4983),
    "warangal": (17.9689, 79.5941),
    "karimnagar": (18.4386, 79.1288),

    "chennai": (13.0827, 80.2707),
    "coimbatore": (11.0168, 76.9558),
    "madurai": (9.9252, 78.1198),
    "tiruchirappalli": (10.7905, 78.7047),
    "trichy": (10.7905, 78.7047),
    "salem": (11.6643, 78.1460),
    "tirunelveli": (8.7139, 77.7567),
    "vellore": (12.9165, 79.1325),

    "kolkata": (22.5726, 88.3639),
    "howrah": (22.5958, 88.2636),
    "siliguri": (26.7271, 88.3953),
    "durgapur": (23.5204, 87.3119),
    "asansol": (23.6739, 86.9524),

    "ahmedabad": (23.0225, 72.5714),
    "surat": (21.1702, 72.8311),
    "vadodara": (22.3072, 73.1812),
    "rajkot": (22.3039, 70.8022),
    "gandhinagar": (23.2156, 72.6369),

    "jaipur": (26.9124, 75.7873),
    "jodhpur": (26.2389, 73.0243),
    "udaipur": (24.5854, 73.7125),
    "kota": (25.2138, 75.8648),
    "bikaner": (28.0229, 73.3119),
    "ajmer": (26.4499, 74.6399),

    "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319),
    "varanasi": (25.3176, 82.9739),
    "banaras": (25.3176, 82.9739),
    "agra": (27.1767, 78.0081),
    "prayagraj": (25.4358, 81.8463),
    "allahabad": (25.4358, 81.8463),
    "meerut": (28.9845, 77.7064),
    "bareilly": (28.3670, 79.4304),
    "aligarh": (27.8974, 78.0880),
    "gorakhpur": (26.7606, 83.3732),

    "bhopal": (23.2599, 77.4126),
    "indore": (22.7196, 75.8577),
    "gwalior": (26.2183, 78.1828),
    "jabalpur": (23.1815, 79.9864),
    "ujjain": (23.1765, 75.7885),

    "patna": (25.5941, 85.1376),
    "gaya": (24.7914, 85.0002),
    "muzaffarpur": (26.1209, 85.3647),

    "visakhapatnam": (17.6868, 83.2185),
    "vizag": (17.6868, 83.2185),
    "vijayawada": (16.5062, 80.6480),
    "guntur": (16.3067, 80.4365),
    "tirupati": (13.6288, 79.4192),

    "thiruvananthapuram": (8.5241, 76.9366),
    "trivandrum": (8.5241, 76.9366),
    "kochi": (9.9312, 76.2673),
    "cochin": (9.9312, 76.2673),
    "kozhikode": (11.2588, 75.7804),
    "calicut": (11.2588, 75.7804),

    "bhubaneswar": (20.2961, 85.8245),
    "cuttack": (20.4625, 85.8828),
    "rourkela": (22.2604, 84.8536),

    "chandigarh": (30.7333, 76.7794),
    "ludhiana": (30.9010, 75.8573),
    "amritsar": (31.6340, 74.8723),
    "jalandhar": (31.3260, 75.5762),

    "ranchi": (23.3441, 85.3096),
    "jamshedpur": (22.8046, 86.2029),
    "dhanbad": (23.7957, 86.4304),

    "raipur": (21.2514, 81.6296),
    "guwahati": (26.1445, 91.7362),
    "dehradun": (30.3165, 78.0322),
    "haridwar": (29.9457, 78.1642),
    "rishikesh": (30.0869, 78.2676),
    "shimla": (31.1048, 77.1734),
    "dharamshala": (32.2190, 76.3234),
    "srinagar": (34.0837, 74.7973),
    "jammu": (32.7266, 74.8570),
    "panaji": (15.4909, 73.8278),
    "goa": (15.2993, 74.1240),
    "pondicherry": (11.9416, 79.8083),

    # Localities
    "koramangala": (12.9352, 77.6245),
    "indiranagar": (12.9784, 77.6408),
    "whitefield": (12.9698, 77.7500),
    "hsrlayout": (12.9121, 77.6446),
    "electroniccity": (12.8452, 77.6602),
    "bandra": (19.0596, 72.8295),
    "andheri": (19.1136, 72.8697),
    "powai": (19.1176, 72.9060),
    "hiteccity": (17.4474, 78.3762),
    "gachibowli": (17.4401, 78.3489),
    "saket": (28.5245, 77.2066),
    "connaughtplace": (28.6315, 77.2167),
    "saltlake": (22.5804, 88.4178),

    # States Centroids
    "karnataka": (15.3173, 75.7139),
    "maharashtra": (19.7515, 75.7139),
    "tamilnadu": (11.1271, 78.6569),
    "telangana": (18.1124, 79.0193),
    "uttarpradesh": (26.8467, 80.9462),
    "kerala": (10.8505, 76.2711),
    "gujarat": (22.2587, 71.1924),
    "rajasthan": (27.0238, 74.2179),
    "westbengal": (22.9868, 87.8550),
    "madhyapradesh": (22.9734, 78.6569),
    "bihar": (25.0961, 85.3131),
    "andhrapradesh": (15.9129, 79.7400),
    "punjab": (31.1471, 75.3412),
    "haryana": (29.0588, 76.0856),
    "odisha": (20.9517, 85.0985),

    # Global Hubs
    "manila": (14.5995, 120.9842),
    "quezon": (14.6760, 121.0437),
    "cebu": (10.3157, 123.8854),
    "singapore": (1.3521, 103.8198),
    "tokyo": (35.6762, 139.6503),
    "newyork": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
}




def extract_lat_lng(obj: Any) -> Optional[Tuple[float, float]]:
    if getattr(obj, "lat", None) is not None and getattr(obj, "lng", None) is not None:
        return (float(obj.lat), float(obj.lng))
    coords = getattr(obj, "coordinates", None)
    if isinstance(coords, dict) and "lat" in coords and "lng" in coords:
        try:
            return (float(coords["lat"]), float(coords["lng"]))
        except (ValueError, TypeError):
            pass
    
    # Fallback to city string matching
    loc = getattr(obj, "location", "")
    if isinstance(loc, str) and loc.strip():
        loc_lower = loc.lower()
        sorted_keys = sorted(CITY_COORDINATES.keys(), key=lambda k: len(k), reverse=True)
        # Pass 1: match specific cities first
        for city_name in sorted_keys:
            if city_name == "manila":
                continue
            if city_name in loc_lower:
                return CITY_COORDINATES[city_name]
        # Pass 2: match manila
        if "manila" in loc_lower:
            return CITY_COORDINATES["manila"]

    return CITY_COORDINATES["manila"]




class LocationRule(BusinessRule):
    def __init__(self, max_distance_km: float = 50.0):
        self.max_distance_km = max_distance_km

    def apply(self, campaign: Campaign, candidates: List[Candidate], volunteers: List[Volunteer]) -> List[Candidate]:
        camp_coords = extract_lat_lng(campaign)
        volunteer_map = {v.id: v for v in volunteers}

        if not camp_coords:
            for c in candidates:
                if c.metadata is None:
                    c.metadata = {}
                c.metadata["distance_km"] = 5.0
            return candidates

        filtered: List[Candidate] = []
        for candidate in candidates:
            vol = volunteer_map.get(candidate.volunteer_id)
            if vol is None:
                continue

            vol_coords = extract_lat_lng(vol)
            if not vol_coords:
                dist = 5.0
            else:
                dist = haversine_distance(camp_coords, vol_coords)

            if candidate.metadata is None:
                candidate.metadata = {}
            candidate.metadata["distance_km"] = round(dist, 2)

            if dist <= self.max_distance_km or len(candidates) <= 3:
                filtered.append(candidate)

        # Fallback if hard filtering removed all candidates
        return filtered if filtered else candidates

