import api from "../lib/api";

export const getRecommendations = async (interest) => {
  try {
    const response = await api.post(`/ai/recommend`, { interest });
    console.log(response.data);
    return response.data;
  } catch (error) {
    console.log(error);
    return [];
  }
};
