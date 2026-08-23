import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Mail, Lock, Eye, EyeOff, Building2, Loader2 } from "lucide-react";
import { authService } from "../services/api";
import { useAuth } from "../context/AuthContext";

export default function NGOLogin() {
  const navigate = useNavigate();
  const { loginNGOWithToken } = useAuth();

  const [email, setEmail] = useState("contact@greenfutureph.org");
  const [password, setPassword] = useState("password123");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await authService.loginNGO({ email, password });
      const ngoData = res.ngo || res.profile || res.data || res;
      if (res.token) {
        loginNGOWithToken(res.token, ngoData);
        navigate("/ngo");
      } else {
        setError(res.message || "Failed to retrieve token from login.");
      }
    } catch (err) {
      setError(err.message || "Invalid NGO email or password.");
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6 text-foreground">
      <div className="w-full max-w-md rounded-2xl border border-border bg-card p-8 shadow-xl space-y-6">
        <div className="text-center space-y-2">
          <Link to="/" className="inline-flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-600 text-white font-bold text-lg shadow-md">
              <Building2 className="h-5 w-5" />
            </div>
            <span className="text-2xl font-bold text-primary">CivicEngage</span>
          </Link>
          <h2 className="text-xl font-bold">NGO Portal Login</h2>
          <p className="text-sm text-muted-foreground">
            Sign in to manage events, AI volunteer matching, and tenders
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="mb-2 block text-sm font-medium">
              Organization Email
            </label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setError("");
                }}
                placeholder="contact@ngo.org"
                className="w-full rounded-xl border border-border bg-background px-10 py-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20 transition-all"
                required
              />
            </div>
          </div>

          <div>
            <label htmlFor="password" className="mb-2 block text-sm font-medium">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
              <input
                id="password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError("");
                }}
                placeholder="Enter password"
                className="w-full rounded-xl border border-border bg-background px-10 py-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20 transition-all"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
              >
                {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
              </button>
            </div>
          </div>

          {error && <p className="text-xs font-semibold text-destructive">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-700 transition-all shadow-md disabled:opacity-60 flex items-center justify-center gap-2"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            Login to NGO Dashboard
          </button>
        </form>

        <div className="border-t border-border pt-4 text-center text-sm text-muted-foreground">
          Don't have an NGO account?{" "}
          <Link to="/ngo-signup" className="font-semibold text-blue-600 hover:underline">
            Register Organization
          </Link>
          <div className="mt-3">
            <Link to="/login" className="text-xs text-muted-foreground hover:text-foreground underline">
              Are you a volunteer? Login here
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
