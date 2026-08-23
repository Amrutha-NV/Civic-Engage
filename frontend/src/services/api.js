import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to attach JWT token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("civic_token") || localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },

  (error) => Promise.reject(error)
);


// Response interceptor
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.message || error.message || 'An unexpected error occurred.';
    return Promise.reject(new Error(message));
  }
);

// Auth Services
export const authService = {
  sendUserOTP: (phone) => api.post('/auth/user/send-otp', { phone }),
  verifyUserOTP: (payload) => api.post('/auth/user/verify-otp', payload),
  registerNGO: (payload) => api.post('/auth/ngo/register', payload),
  loginNGO: (payload) => api.post('/auth/ngo/login', payload),
  getMe: () => api.get('/auth/me'),
};

// User Profile Services
export const userService = {
  getProfile: () => api.get('/users/profile'),
  updateProfile: (payload) => api.put('/users/profile', payload),
  getVolunteers: () => api.get('/users/volunteers'),
};

// Event Services
export const eventService = {
  getAllEvents: (params) => api.get('/events', { params }),
  getEventById: (id) => api.get(`/events/${id}`),
  createEvent: (payload) => api.post('/ngos/events', payload),
  updateEvent: (id, payload) => api.put(`/ngos/events/${id}`, payload),
  deleteEvent: (id) => api.delete(`/ngos/events/${id}`),
  registerForEvent: (id) => api.post(`/events/${id}/register`),
};

// AI & Tender Services
export const aiService = {
  getEventRecommendations: (payload) => api.post('/recommendations/events', payload || {}),
  getVolunteerRecommendations: (eventId) => api.post('/recommendations/volunteers', { eventId }),
  generateTender: (payload) => api.post('/tender/generate', payload),
};

// Image Upload Service
export const uploadService = {
  uploadImage: (file) => {
    const formData = new FormData();
    formData.append('image', file);
    return api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};

export default api;
