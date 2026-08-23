import React from "react";
import { Award, ShieldCheck, TreePine, BookOpen, HeartPulse, Sparkles } from "lucide-react";

export default function Badges() {
  const badgesList = [
    { title: "Eco Guardian", desc: "Participated in watershed & reforestation campaigns", icon: TreePine, color: "bg-green-100 text-green-700 border-green-200" },
    { title: "Knowledge Builder", desc: "Taught digital literacy or tutored children", icon: BookOpen, color: "bg-blue-100 text-blue-700 border-blue-200" },
    { title: "Community Lifesaver", desc: "Assisted in medical missions & first aid triage", icon: HeartPulse, color: "bg-purple-100 text-purple-700 border-purple-200" },
    { title: "Verified Volunteer", desc: "Mobile OTP authenticated active user", icon: ShieldCheck, color: "bg-amber-100 text-amber-700 border-amber-200" },
  ];

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-extrabold">Achievements & Badges</h1>
        <p className="text-sm text-muted-foreground">
          Earn recognition badges as you participate in diverse NGO campaigns
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {badgesList.map((badge, idx) => {
          const Icon = badge.icon;
          return (
            <div
              key={idx}
              className="rounded-2xl border border-border bg-card p-5 shadow-sm flex items-start gap-4 hover:shadow-md transition-all"
            >
              <div className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border ${badge.color}`}>
                <Icon className="h-7 w-7" />
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h3 className="font-bold text-base">{badge.title}</h3>
                  <span className="rounded-full bg-green-100 text-green-800 text-[10px] font-bold px-2 py-0.5">
                    Unlocked
                  </span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{badge.desc}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
