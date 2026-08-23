import api from "../lib/api";

export const getScore = async () => {
  const response = await api.get(`/impact/score`);
  return response.data;
};

export const getBadges = async () => {
  const response = await api.get(`/impact/badges`);
  return response.data;
};
