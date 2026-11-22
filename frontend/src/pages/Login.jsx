import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Mail, ArrowRight, RefreshCw, Loader2 } from "lucide-react";
import { authService } from "../services/api";
import { useAuth } from "../context/AuthContext";

export default function Login() {
  const navigate = useNavigate();
  const { loginUserWithToken } = useAuth();

  const [step, setStep] = useState("email"); // "email" | "otp"
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState(["", "", "", "", "", ""]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [otpNotice, setOtpNotice] = useState("");

  const handleSendOTP = async (e) => {
    e.preventDefault();
    if (!email.trim()) {
      setError("Please enter your email address.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await authService.sendLoginOtp(email.trim());
      setOtpNotice(res.message || "OTP sent successfully.");
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
      const nextInput = document.getElementById(`otp-${index + 1}`);
      nextInput?.focus();
    }
  };

  const handleOtpKeyDown = (index, e) => {
    if (e.key === "Backspace" && !otp[index] && index > 0) {
      const prevInput = document.getElementById(`otp-${index - 1}`);
      prevInput?.focus();
    }
  };

  const handleVerify = async (e) => {
    e.preventDefault();
    const entered = otp.join("");
    if (entered.length < 6) {
      setError("Please enter all 6 digits.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await authService.verifyLoginOtp({ email: email.trim(), otp: entered });
      const userData = res.user || res.profile || res.data || res;
      if (res.token) {
        loginUserWithToken(res.token, userData);
        navigate("/app");
      } else {
        setError(res.message || "Failed to retrieve token from verification.");
      }
    } catch (err) {
      setError(err.message || "OTP verification failed.");
    } finally {
      setLoading(false);
    }
  };


  const maskedEmail = email.replace(/^(.).+(@.+)$/, "$1••••$2");

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6 text-foreground">
      <div className="w-full max-w-md rounded-2xl border border-border bg-card p-8 shadow-xl space-y-6">
        <div className="text-center space-y-2">
          <Link to="/" className="inline-flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-primary-foreground font-bold text-lg">
              C
            </div>
            <span className="text-2xl font-bold text-primary">CivicEngage</span>
          </Link>
          <h2 className="text-xl font-bold">
            {step === "email" ? "Volunteer Email Login" : "Verify Email OTP"}
          </h2>
          <p className="text-sm text-muted-foreground">
            {step === "email"
              ? "Enter your registered email address to continue"
              : `OTP code sent to ${maskedEmail}`}
          </p>
        </div>

        {step === "email" ? (
          <form onSubmit={handleSendOTP} className="space-y-4">
            <div>
              <label htmlFor="email" className="mb-2 block text-sm font-medium">
                Email Address
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
                  placeholder="you@example.com"
                  className="w-full rounded-xl border border-border bg-background px-10 py-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all"
                  required
                />
              </div>
              {error && <p className="mt-2 text-xs font-semibold text-destructive">{error}</p>}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md disabled:opacity-60"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Send Email OTP"}
              <ArrowRight className="h-4 w-4" />
            </button>
          </form>
        ) : (
          <form onSubmit={handleVerify} className="space-y-6">
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

              {otpNotice && <p className="text-center text-xs text-muted-foreground">{otpNotice}</p>}
            </div>

            <div className="space-y-3">
              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md disabled:opacity-60 flex items-center justify-center gap-2"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                Verify & Access Dashboard
              </button>
              <button
                type="button"
                onClick={() => {
                  setStep("email");
                  setOtp(["", "", "", "", "", ""]);
                  setError("");
                  setOtpNotice("");
                }}
                className="flex w-full items-center justify-center gap-2 rounded-xl border border-border px-4 py-2.5 text-xs text-muted-foreground hover:bg-muted transition-all"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Change email / Request another OTP
              </button>
            </div>
          </form>
        )}

        <div className="border-t border-border pt-4 text-center text-sm text-muted-foreground">
          New volunteer?{" "}
          <Link to="/signup" className="font-semibold text-primary hover:underline">
            Create Account
          </Link>
          <div className="mt-3">
            <Link to="/ngo-login" className="text-xs text-muted-foreground hover:text-foreground underline">
              Are you an NGO? Login here
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
