import React, { createContext, useContext, useState, useEffect } from "react";
import { authService, userService } from "../services/api";

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ngo, setNgo] = useState(null);
  const [role, setRole] = useState(localStorage.getItem("civic_role") || null);
  const [loading, setLoading] = useState(true);

  const fetchSession = async () => {
    const token = localStorage.getItem("civic_token") || localStorage.getItem("token");
    if (!token) {
      setLoading(false);
      return;
    }
    try {
      const data = await authService.getMe();
      
      // Safely check nested or direct profile structure
      const profileData = data.profile || data.ngo || data.user || data;
      const userRole = (data.role || profileData.role || localStorage.getItem("civic_role") || "").toLowerCase();

      if (userRole === "ngo") {
        setNgo(profileData);
        setUser(null);
        setRole("ngo");
        localStorage.setItem("civic_role", "ngo");
      } else if (userRole === "user" || userRole) {
        setUser(profileData);
        setNgo(null);
        setRole("user");
        localStorage.setItem("civic_role", "user");
      } else {
        console.error("User fetched, but no role specified in response:", data);
      }
    } catch (err) {
      console.warn("Session check failed:", err.message);
      const errStr = (err.message || "").toLowerCase();
      // Only log out if it's genuinely an unauthenticated / expired token error (401)
      if (errStr.includes("401") || errStr.includes("unauthorized") || errStr.includes("expired") || errStr.includes("token missing")) {
        logout();
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSession();
  }, []);

  const loginUserWithToken = (token, userData) => {
    localStorage.setItem("civic_token", token);
    localStorage.setItem("token", token);
    localStorage.setItem("civic_role", "user");
    setUser(userData);
    setNgo(null);
    setRole("user");
  };

  const loginNGOWithToken = (token, ngoData) => {
    localStorage.setItem("civic_token", token);
    localStorage.setItem("token", token);
    localStorage.setItem("civic_role", "ngo");
    setNgo(ngoData);
    setUser(null);
    setRole("ngo");
  };

  const logout = () => {
    localStorage.removeItem("civic_token");
    localStorage.removeItem("token");
    localStorage.removeItem("civic_role");
    setUser(null);
    setNgo(null);
    setRole(null);
  };


  const updateUserProfileState = (updatedUser) => {
    setUser(updatedUser);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        ngo,
        role,
        loading,
        loginUserWithToken,
        loginNGOWithToken,
        logout,
        updateUserProfileState,
        refreshSession: fetchSession,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
