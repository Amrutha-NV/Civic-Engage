import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";

// Pages & Layouts
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import NGOLogin from "./pages/NGOLogin";
import NGOSignup from "./pages/NGOSignup";

import DashboardLayout from "./layouts/DashboardLayout";
import Dashboard from "./pages/Dashboard";
import Profile from "./pages/Profile";
import History from "./pages/History";
import ImpactScore from "./pages/ImpactScore";
import Badges from "./pages/Badges";

import NGOLayout from "./layouts/NGOLayout";
import NGOOverview from "./pages/ngo/NGOOverview";
import NGOTender from "./pages/ngo/NGOTender";
import NGOOrganization from "./pages/ngo/NGOOrganization";
import NGOVolunteers from "./pages/ngo/NGOVolunteers";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Authentication Routes */}
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/ngo-login" element={<NGOLogin />} />
          <Route path="/ngo-signup" element={<NGOSignup />} />

          {/* User Dashboard Routes */}
          <Route path="/app" element={<DashboardLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="profile" element={<Profile />} />
            <Route path="history" element={<History />} />
            <Route path="impact" element={<ImpactScore />} />
            <Route path="badges" element={<Badges />} />
          </Route>

          {/* NGO Portal Routes */}
          <Route path="/ngo" element={<NGOLayout />}>
            <Route index element={<NGOOverview />} />
            <Route path="tender" element={<NGOTender />} />
            <Route path="organization" element={<NGOOrganization />} />
            <Route path="volunteers" element={<NGOVolunteers />} />
            <Route path="campaigns" element={<NGOOverview />} />
          </Route>

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
