import api from "../lib/api";

export const getNotifications = async (page = 1, limit = 10) => {
  const response = await api.get(`/notifications?page=${page}&limit=${limit}`);
  return response.data;
};

export const markAsRead = async (id) => {
  const response = await api.put(`/notifications/${id}/read`);
  return response.data;
};
