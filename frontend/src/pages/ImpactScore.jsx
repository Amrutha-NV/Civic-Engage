import React from "react";
import { Heart, Award, TrendingUp, Sparkles, CheckCircle2 } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function ImpactScore() {
  const { user } = useAuth();
  const score = user?.impactScore || 85;

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-extrabold">Civic Impact Score</h1>
        <p className="text-sm text-muted-foreground">
          Quantifying your overall contribution, attendance, and skill deployment in NGO events
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="rounded-2xl border border-primary/20 bg-primary/5 p-6 shadow-sm flex flex-col items-center text-center space-y-2">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary text-primary-foreground font-bold text-2xl shadow-md">
            {score}
          </div>
          <h3 className="font-bold text-lg">Overall Impact Score</h3>
          <p className="text-xs text-muted-foreground">Top 10% of active volunteers in Metro Manila</p>
        </div>

        <div className="rounded-2xl border border-border bg-card p-6 shadow-sm flex flex-col items-center text-center space-y-2">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-green-100 text-green-600 font-bold text-xl">
            96%
          </div>
          <h3 className="font-bold text-lg">Attendance Rate</h3>
          <p className="text-xs text-muted-foreground">Punctuality and reliability across events</p>
        </div>

        <div className="rounded-2xl border border-border bg-card p-6 shadow-sm flex flex-col items-center text-center space-y-2">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-100 text-blue-600 font-bold text-xl">
            {(user?.registeredEvents?.length || 1) * 6} hrs
          </div>
          <h3 className="font-bold text-lg">Volunteer Hours</h3>
          <p className="text-xs text-muted-foreground">Total verified hours served with NGOs</p>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card p-6 shadow-sm space-y-4">
        <h3 className="text-base font-bold flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary" />
          Impact Score Calculation Factors
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-muted-foreground">
          <div className="flex items-start gap-2 p-3 rounded-xl bg-muted/30">
            <CheckCircle2 className="h-4 w-4 text-green-600 shrink-0 mt-0.5" />
            <div>
              <strong className="text-foreground block">Event Participation & Attendance (+40 pts)</strong>
              Verified attendance at registered NGO events.
            </div>
          </div>
          <div className="flex items-start gap-2 p-3 rounded-xl bg-muted/30">
            <CheckCircle2 className="h-4 w-4 text-green-600 shrink-0 mt-0.5" />
            <div>
              <strong className="text-foreground block">Skill Match Deployment (+30 pts)</strong>
              Utilizing high-demand skills like healthcare or teaching.
            </div>
          </div>
          <div className="flex items-start gap-2 p-3 rounded-xl bg-muted/30">
            <CheckCircle2 className="h-4 w-4 text-green-600 shrink-0 mt-0.5" />
            <div>
              <strong className="text-foreground block">NGO Ratings & Feedback (+20 pts)</strong>
              Positive ratings given by NGO event organizers.
            </div>
          </div>
          <div className="flex items-start gap-2 p-3 rounded-xl bg-muted/30">
            <CheckCircle2 className="h-4 w-4 text-green-600 shrink-0 mt-0.5" />
            <div>
              <strong className="text-foreground block">Continuous Consistency (+10 pts)</strong>
              Active participation over consecutive months.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
