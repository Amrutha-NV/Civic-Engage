import api from "../lib/api";

export const sendMessage = async (receiverId, content) => {
  const response = await api.post(`/messages`, { receiverId, content });
  return response.data;
};

export const getConversation = async (userId) => {
  const response = await api.get(`/messages/${userId}`);
  return response.data;
};
