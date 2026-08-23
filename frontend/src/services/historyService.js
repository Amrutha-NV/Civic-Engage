import api from "../lib/api";

export const getParticipations = async () => {
  const response = await api.get(`/history/participations`);
  return response.data;
};

export const getHostedEvents = async () => {
  const response = await api.get(`/history/events`);
  return response.data;
};
