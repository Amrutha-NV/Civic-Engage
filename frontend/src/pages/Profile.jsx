import React, { useState } from "react";
import { User, Mail, Phone, MapPin, Briefcase, Award, Check, Sparkles, Pencil, Loader2, Save } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { userService } from "../services/api";

const availableSkillsList = [
  "Teaching",
  "Healthcare",
  "Technology",
  "Marketing",
  "Design",
  "Writing",
  "Event Planning",
  "Photography",
  "Environmental Science",
  "First Aid",
  "Community Outreach",
];

const availableInterestsList = [
  "Environmental Protection",
  "Education & Literacy",
  "Healthcare & Wellness",
  "Disaster Relief",
  "Youth Development",
  "Animal Welfare",
];

export default function Profile() {
  const { user, updateUserProfileState } = useAuth();
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [successMsg, setSuccessMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const [formData, setFormData] = useState({
    name: user?.name || "Volunteer User",
    email: user?.email || "",
    location: user?.location || "Manila, Philippines",
    occupation: user?.occupation || "Student / Volunteer",
    availability: user?.availability || "Available",
    skills: user?.skills || ["Community Outreach"],
    interests: user?.interests || ["Environmental Protection"],
  });

  const toggleSkill = (skill) => {
    setFormData((prev) => ({
      ...prev,
      skills: prev.skills.includes(skill)
        ? prev.skills.filter((s) => s !== skill)
        : [...prev.skills, skill],
    }));
  };

  const toggleInterest = (interest) => {
    setFormData((prev) => ({
      ...prev,
      interests: prev.interests.includes(interest)
        ? prev.interests.filter((i) => i !== interest)
        : [...prev.interests, interest],
    }));
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setErrorMsg("");
    setSuccessMsg("");
    setLoading(true);
    try {
      const res = await userService.updateProfile(formData);
      if (res.user) {
        updateUserProfileState(res.user);
        setSuccessMsg("Profile updated successfully and synced with AI Volunteer Matching!");
        setEditing(false);
      }
    } catch (err) {
      setErrorMsg(err.message || "Failed to update profile.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold">Volunteer Profile</h1>
          <p className="text-sm text-muted-foreground">
            Manage your personal details, skills, cause interests, and AI matching parameters
          </p>
        </div>

        <button
          onClick={() => setEditing((v) => !v)}
          className={`flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition-all ${
            editing
              ? "bg-muted text-foreground hover:bg-muted/80"
              : "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm"
          }`}
        >
          <Pencil className="h-4 w-4" />
          {editing ? "Cancel Editing" : "Edit Profile"}
        </button>
      </div>

      {successMsg && (
        <div className="rounded-xl border border-green-200 bg-green-50 p-4 text-xs font-semibold text-green-800">
          ✅ {successMsg}
        </div>
      )}
      {errorMsg && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-xs font-semibold text-red-800">
          ❌ {errorMsg}
        </div>
      )}

      {/* Main Profile Card */}
      <div className="rounded-2xl border border-border bg-card p-6 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row items-center sm:items-start gap-5 pb-6 border-b border-border">
          <img
            src={user?.avatar || "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150"}
            alt={user?.name}
            className="h-24 w-24 rounded-full object-cover border-2 border-primary shadow-md"
          />
          <div className="space-y-1 text-center sm:text-left flex-1">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h2 className="text-xl font-bold">{user?.name || "Volunteer"}</h2>
              <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-3 py-1 text-xs font-bold text-primary border border-primary/20">
                <Award className="h-3.5 w-3.5" />
                Impact Score: {user?.impactScore || 85}/100
              </span>
            </div>
            <p className="text-sm text-muted-foreground">{user?.occupation || "Volunteer"}</p>
            <div className="flex flex-wrap items-center justify-center sm:justify-start gap-4 pt-1 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <Phone className="h-3.5 w-3.5 text-primary" /> {user?.phone}
              </span>
              {user?.email && (
                <span className="flex items-center gap-1">
                  <Mail className="h-3.5 w-3.5 text-primary" /> {user?.email}
                </span>
              )}
              <span className="flex items-center gap-1">
                <MapPin className="h-3.5 w-3.5 text-primary" /> {user?.location || "Manila"}
              </span>
            </div>
          </div>
        </div>

        {/* View vs Edit */}
        {editing ? (
          <form onSubmit={handleSave} className="space-y-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="mb-1 block text-xs font-semibold">Full Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full rounded-xl border border-border bg-background py-2 px-3 text-sm outline-none focus:border-primary"
                  required
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold">Email Address</label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="w-full rounded-xl border border-border bg-background py-2 px-3 text-sm outline-none focus:border-primary"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold">Location / City</label>
                <input
                  type="text"
                  value={formData.location}
                  onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                  className="w-full rounded-xl border border-border bg-background py-2 px-3 text-sm outline-none focus:border-primary"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold">Occupation</label>
                <input
                  type="text"
                  value={formData.occupation}
                  onChange={(e) => setFormData({ ...formData, occupation: e.target.value })}
                  className="w-full rounded-xl border border-border bg-background py-2 px-3 text-sm outline-none focus:border-primary"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold">Availability</label>
                <select
                  value={formData.availability}
                  onChange={(e) => setFormData({ ...formData, availability: e.target.value })}
                  className="w-full rounded-xl border border-border bg-background py-2 px-3 text-sm outline-none focus:border-primary"
                >
                  <option value="Available">Available</option>
                  <option value="Weekends">Weekends Only</option>
                  <option value="Weekdays">Weekdays Only</option>
                  <option value="Flexible">Flexible</option>
                </select>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-xs font-semibold">Skills</label>
              <div className="flex flex-wrap gap-2">
                {availableSkillsList.map((skill) => (
                  <button
                    key={skill}
                    type="button"
                    onClick={() => toggleSkill(skill)}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition-all ${
                      formData.skills.includes(skill)
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "bg-muted text-muted-foreground hover:bg-muted/80"
                    }`}
                  >
                    {skill}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="mb-2 block text-xs font-semibold">Cause Interests</label>
              <div className="flex flex-wrap gap-2">
                {availableInterestsList.map((interest) => (
                  <button
                    key={interest}
                    type="button"
                    onClick={() => toggleInterest(interest)}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition-all ${
                      formData.interests.includes(interest)
                        ? "bg-blue-600 text-white shadow-sm"
                        : "bg-muted text-muted-foreground hover:bg-muted/80"
                    }`}
                  >
                    {interest}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setEditing(false)}
                className="rounded-xl border border-border px-5 py-2.5 text-sm font-semibold hover:bg-muted transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="flex items-center gap-2 rounded-xl bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md disabled:opacity-60"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                Save & Sync AI Profile
              </button>
            </div>
          </form>
        ) : (
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-2">Registered Skills</h3>
              <div className="flex flex-wrap gap-2">
                {(user?.skills || []).map((skill) => (
                  <span
                    key={skill}
                    className="rounded-full bg-primary/10 border border-primary/20 px-3 py-1 text-xs font-semibold text-primary"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-2">Cause Interests</h3>
              <div className="flex flex-wrap gap-2">
                {(user?.interests || []).map((interest) => (
                  <span
                    key={interest}
                    className="rounded-full bg-blue-50 border border-blue-200 px-3 py-1 text-xs font-semibold text-blue-700"
                  >
                    {interest}
                  </span>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border">
              <div className="rounded-xl border border-border p-3">
                <span className="text-xs text-muted-foreground block">Availability Status</span>
                <span className="font-bold text-sm">{user?.availability || "Available"}</span>
              </div>
              <div className="rounded-xl border border-border p-3">
                <span className="text-xs text-muted-foreground block">Verified Account</span>
                <span className="font-bold text-sm text-green-600 flex items-center gap-1">
                  <Check className="h-4 w-4" /> Mobile Verified
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
