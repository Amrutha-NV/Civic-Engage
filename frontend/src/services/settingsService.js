import api from "../lib/api";

export const updatePreferences = async (preferences) => {
  // preferences should be an object: { emailPreferences?: boolean, privacyMode?: "public" | "private" }
  const response = await api.put(`/settings/preferences`, preferences);
  return response.data;
};
