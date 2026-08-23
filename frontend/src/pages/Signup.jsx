import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Phone, User, Mail, MapPin, Briefcase, ArrowRight, Loader2, RefreshCw } from "lucide-react";
import { authService } from "../services/api";
import { useAuth } from "../context/AuthContext";

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

export default function Signup() {
  const navigate = useNavigate();
  const { loginUserWithToken } = useAuth();

  const [step, setStep] = useState("form"); // "form" | "otp"
  const [formData, setFormData] = useState({
    phone: "+919876543210",
    name: "",
    email: "",
    location: "Bengaluru, Karnataka",
    occupation: "Student / Volunteer",
    availability: "Available",
    skills: ["Teaching", "Community Outreach"],
    interests: ["Education & Literacy", "Environmental Protection"],
  });


  const [otp, setOtp] = useState(["", "", "", "", "", ""]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

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

  const handleSendOTP = async (e) => {
    e.preventDefault();
    if (!formData.name || !formData.phone) {
      setError("Name and mobile number are required.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      await authService.sendUserOTP(formData.phone);
      setStep("otp");
    } catch (err) {
      setError(err.message || "Failed to send OTP.");
    } finally {
      setLoading(false);
    }
  };

  const handleOtpChange = (index, value) => {
    if (!/^\d?$/.test(value)) return;
    const next = [...otp];
    next[index] = value;
    setOtp(next);
    if (value && index < 5) {
      document.getElementById(`otp-${index + 1}`)?.focus();
    }
  };

  const handleOtpKeyDown = (index, e) => {
    if (e.key === "Backspace" && !otp[index] && index > 0) {
      document.getElementById(`otp-${index - 1}`)?.focus();
    }
  };

  const handleVerifyAndRegister = async (e) => {
    e.preventDefault();
    const entered = otp.join("");
    if (entered.length < 6) {
      setError("Please enter 6-digit OTP.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await authService.verifyUserOTP({
        ...formData,
        otp: entered,
      });
      if (res.token && res.user) {
        loginUserWithToken(res.token, res.user);
        navigate("/app");
      }
    } catch (err) {
      setError(err.message || "Registration verification failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6 text-foreground">
      <div className="w-full max-w-xl rounded-2xl border border-border bg-card p-8 shadow-xl space-y-6">
        <div className="text-center space-y-2">
          <Link to="/" className="inline-flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-primary-foreground font-bold text-lg">
              C
            </div>
            <span className="text-2xl font-bold text-primary">CivicEngage</span>
          </Link>
          <h2 className="text-xl font-bold">Create Volunteer Account</h2>
          <p className="text-sm text-muted-foreground">
            {step === "form"
              ? "Join CivicEngage to discover events & match with NGOs"
              : `Enter 6-digit OTP sent to ${formData.phone}`}
          </p>
        </div>

        {step === "form" ? (
          <form onSubmit={handleSendOTP} className="space-y-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="mb-1.5 block text-sm font-medium">Full Name *</label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="Jane Doe"
                    className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium">Mobile Number *</label>
                <div className="relative">
                  <Phone className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="tel"
                    value={formData.phone}
                    onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                    placeholder="+63 9XX XXX XXXX"
                    className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium">Email Address</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    placeholder="jane@example.com"
                    className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  />
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium">City / Location</label>
                <div className="relative">
                  <MapPin className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="text"
                    value={formData.location}
                    onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                    placeholder="e.g. Bengaluru, Mumbai, Delhi, or PIN code"
                    className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  />
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium">Occupation</label>
                <div className="relative">
                  <Briefcase className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="text"
                    value={formData.occupation}
                    onChange={(e) => setFormData({ ...formData, occupation: e.target.value })}
                    placeholder="Student / Engineer"
                    className="w-full rounded-xl border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  />
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium">Availability</label>
                <select
                  value={formData.availability}
                  onChange={(e) => setFormData({ ...formData, availability: e.target.value })}
                  className="w-full rounded-xl border border-border bg-background py-2.5 px-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                >
                  <option value="Available">Available Anytime</option>
                  <option value="Weekends">Weekends Only</option>
                  <option value="Weekdays">Weekdays Only</option>
                  <option value="Flexible">Flexible Schedule</option>
                </select>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm font-semibold">Select Your Skills</label>
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
              <label className="mb-2 block text-sm font-semibold">Select Your Cause Interests</label>
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

            {error && <p className="text-xs font-semibold text-destructive">{error}</p>}

            <button
              type="submit"
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md disabled:opacity-60"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Continue & Verify Mobile OTP"}
              <ArrowRight className="h-4 w-4" />
            </button>
          </form>
        ) : (
          <form onSubmit={handleVerifyAndRegister} className="space-y-6">
            <div className="space-y-3">
              <label className="block text-center text-sm font-medium">Enter 6-digit OTP Code</label>
              <div className="flex justify-center gap-2">
                {otp.map((digit, i) => (
                  <input
                    key={i}
                    id={`otp-${i}`}
                    type="text"
                    inputMode="numeric"
                    maxLength={1}
                    value={digit}
                    onChange={(e) => handleOtpChange(i, e.target.value)}
                    onKeyDown={(e) => handleOtpKeyDown(i, e)}
                    className="h-12 w-12 rounded-xl border border-border bg-background text-center text-lg font-bold outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all"
                  />
                ))}
              </div>

              {error && <p className="text-center text-xs font-semibold text-destructive">{error}</p>}

              <div className="rounded-xl border border-blue-200 bg-blue-50/50 p-3 text-center space-y-1">
                <p className="text-xs text-blue-800">
                  Demo OTP Code: <span className="font-mono font-bold text-blue-900">123456</span>
                </p>
                <button
                  type="button"
                  onClick={() => setOtp(["1", "2", "3", "4", "5", "6"])}
                  className="text-xs text-blue-600 hover:underline font-medium"
                >
                  Auto-fill demo OTP (123456)
                </button>
              </div>
            </div>

            <div className="space-y-3">
              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md disabled:opacity-60 flex items-center justify-center gap-2"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                Complete Registration & Access Dashboard
              </button>
              <button
                type="button"
                onClick={() => setStep("form")}
                className="flex w-full items-center justify-center gap-2 rounded-xl border border-border px-4 py-2.5 text-xs text-muted-foreground hover:bg-muted transition-all"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Back to Edit Information
              </button>
            </div>
          </form>
        )}

        <div className="border-t border-border pt-4 text-center text-sm text-muted-foreground">
          Already registered?{" "}
          <Link to="/login" className="font-semibold text-primary hover:underline">
            Login here
          </Link>
        </div>
      </div>
    </div>
  );
}
