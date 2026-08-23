import React, { useState, useEffect } from "react";
import { Calendar, MapPin, CheckCircle2, Clock } from "lucide-react";
import { userService } from "../services/api";

export default function History() {
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    userService
      .getProfile()
      .then((res) => {
        if (res.user) setUserProfile(res.user);
      })
      .catch((err) => console.log("History fetch error:", err))
      .finally(() => setLoading(false));
  }, []);

  const registered = userProfile?.registeredEvents || [];

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-extrabold">Activity & Participation History</h1>
        <p className="text-sm text-muted-foreground">
          Track events you have registered for and completed across partner NGOs
        </p>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading participation history...</p>
      ) : registered.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center space-y-3">
          <CheckCircle2 className="h-12 w-12 text-muted-foreground mx-auto" />
          <h3 className="text-lg font-bold">No Events Joined Yet</h3>
          <p className="text-sm text-muted-foreground">
            Explore available events in your dashboard and register to start building your civic impact history!
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {registered.map((event) => (
            <div
              key={event._id || event.id}
              className="rounded-2xl border border-border bg-card p-5 shadow-sm hover:shadow-md transition-all flex flex-col md:flex-row md:items-center justify-between gap-4"
            >
              <div className="space-y-1.5">
                <span className="rounded-full bg-green-100 text-green-700 border border-green-200 px-2.5 py-0.5 text-xs font-semibold">
                  Registered / Confirmed
                </span>
                <h3 className="text-lg font-bold">{event.title}</h3>
                <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Calendar className="h-3.5 w-3.5 text-primary" /> {event.date}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-3.5 w-3.5 text-primary" /> {event.startTime} - {event.endTime}
                  </span>
                  <span className="flex items-center gap-1">
                    <MapPin className="h-3.5 w-3.5 text-primary" /> {event.location}
                  </span>
                </div>
              </div>

              <div className="shrink-0 text-right">
                <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-bold text-primary">
                  {event.category}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
