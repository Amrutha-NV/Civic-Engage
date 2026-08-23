import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Building2, Mail, Phone, MapPin, Lock, Loader2 } from "lucide-react";
import { authService } from "../services/api";
import { useAuth } from "../context/AuthContext";

export default function NGOSignup() {
  const navigate = useNavigate();
  const { loginNGOWithToken } = useAuth();

  const [formData, setFormData] = useState({
    ngoName: "",
    email: "",
    phone: "",
    address: "",
    password: "",
    category: "Environmental Protection",
    description: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.ngoName || !formData.email || !formData.phone || !formData.address || !formData.password) {
      setError("Please fill out all required fields.");
      return;
    }

    setError("");
    setLoading(true);
    try {
      const res = await authService.registerNGO(formData);
      if (res.token && res.ngo) {
        loginNGOWithToken(res.token, res.ngo);
        navigate("/ngo");
      }
    } catch (err) {
      setError(err.message || "NGO registration failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6 text-foreground">
      <div className="w-full max-w-xl rounded-2xl border border-border bg-card p-8 shadow-xl space-y-6">
        <div className="text-center space-y-2">
          <Link to="/" className="inline-flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-600 text-white font-bold text-lg shadow-md">
              <Building2 className="h-5 w-5" />
            </div>
            <span className="text-2xl font-bold text-primary">CivicEngage</span>
          </Link>
          <h2 className="text-xl font-bold">Register Non-Governmental Organization</h2>
          <p className="text-sm text-muted-foreground">
            Create an official NGO account to manage events, volunteers, and tenders
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="mb-1.5 block text-sm font-medium">Organization Name *</label>
              <div className="relative">
                <Building2 className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  type="text"
                  value={formData.ngoName}
                  onChange={(e) => setFormData({ ...formData, ngoName: e.target.value })}
                  placeholder="GreenFuture Foundation"
                  className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20"
                  required
                />
              </div>
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium">Official Email *</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  placeholder="contact@greenfuture.org"
                  className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20"
                  required
                />
              </div>
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium">Contact Phone *</label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  type="tel"
                  value={formData.phone}
                  onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                  placeholder="+63 2 8123 4567"
                  className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20"
                  required
                />
              </div>
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium">Primary Domain Category</label>
              <select
                value={formData.category}
                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                className="w-full rounded-xl border border-border bg-background py-2.5 px-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20"
              >
                <option value="Environmental Protection">Environmental Protection</option>
                <option value="Education & Literacy">Education & Literacy</option>
                <option value="Healthcare & Wellness">Healthcare & Wellness</option>
                <option value="Community Development">Community Development</option>
                <option value="Disaster Relief">Disaster Relief</option>
              </select>
            </div>
          </div>

          <div>
            <label className="mb-1.5 block text-sm font-medium">Headquarters Address *</label>
            <div className="relative">
              <MapPin className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                value={formData.address}
                onChange={(e) => setFormData({ ...formData, address: e.target.value })}
                placeholder="12F Ayala Avenue, Makati City"
                className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20"
                required
              />
            </div>
          </div>

          <div>
            <label className="mb-1.5 block text-sm font-medium">Account Password *</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                placeholder="Minimum 6 characters"
                className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20"
                required
              />
            </div>
          </div>

          <div>
            <label className="mb-1.5 block text-sm font-medium">Organization Mission & Description</label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              rows={3}
              placeholder="Describe your NGO's goals and initiatives..."
              className="w-full resize-none rounded-xl border border-border bg-background p-3 text-sm outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20"
            />
          </div>

          {error && <p className="text-xs font-semibold text-destructive">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-700 transition-all shadow-md disabled:opacity-60 flex items-center justify-center gap-2"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            Register Organization & Access Portal
          </button>
        </form>

        <div className="border-t border-border pt-4 text-center text-sm text-muted-foreground">
          Already registered?{" "}
          <Link to="/ngo-login" className="font-semibold text-blue-600 hover:underline">
            Login here
          </Link>
        </div>
      </div>
    </div>
  );
}
