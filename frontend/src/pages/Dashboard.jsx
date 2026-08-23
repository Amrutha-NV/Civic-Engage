import React, { useState, useEffect } from "react";
import { MapPin, Calendar, Clock, Sparkles, X, User, Briefcase, CheckCircle2, Search, Loader2 } from "lucide-react";
import { eventService, aiService } from "../services/api";
import { useAuth } from "../context/AuthContext";

function StatusBadge({ status }) {
  const styles = {
    Active: "bg-green-100 text-green-700 border-green-200",
    Upcoming: "bg-blue-100 text-blue-700 border-blue-200",
    Completed: "bg-muted text-muted-foreground border-border",
  };
  return (
    <span
      className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${
        styles[status] || "bg-muted text-muted-foreground"
      }`}
    >
      {status || "Active"}
    </span>
  );
}

function EventCard({ event, onParticipate, recommended, reason }) {
  const volunteersJoined = event.volunteersJoined || 0;
  const capacity = event.capacity || 50;
  const pct = Math.round((volunteersJoined / capacity) * 100);

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all flex flex-col justify-between">
      <div>
        <div className="relative h-40 bg-muted">
          <img
            src={event.banner || "https://images.unsplash.com/photo-1559027615-cd4628902d4a?w=800"}
            alt={event.title}
            className="h-full w-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
          <div className="absolute bottom-3 left-3">
            <StatusBadge status={event.status} />
          </div>
          {recommended && (
            <div className="absolute right-3 top-3 flex items-center gap-1 rounded-full bg-primary px-3 py-1 text-[11px] font-bold text-primary-foreground shadow-md">
              <Sparkles className="h-3 w-3" />
              AI Pick ({event.matchScore || 92}%)
            </div>
          )}
        </div>

        <div className="p-4 space-y-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="font-bold text-base leading-tight">{event.title}</h3>
            <span className="shrink-0 rounded-full bg-muted px-2.5 py-0.5 text-[10px] font-medium text-muted-foreground">
              {event.category}
            </span>
          </div>

          {recommended && reason && (
            <p className="text-xs font-medium text-primary bg-primary/5 p-2 rounded-lg border border-primary/10">
              💡 {reason}
            </p>
          )}

          <p className="text-xs text-muted-foreground line-clamp-2">
            {event.shortDesc || event.description}
          </p>

          <div className="space-y-1.5 text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <Calendar className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span>{event.date}</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span>
                {event.startTime} – {event.endTime}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <MapPin className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span className="truncate">{event.location}</span>
            </div>
          </div>

          <div>
            <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
              <span>Volunteers Registered</span>
              <span className="font-semibold text-foreground">
                {volunteersJoined} / {capacity}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${Math.min(100, pct)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="p-4 pt-0">
        <button
          onClick={() => onParticipate(event)}
          className="w-full rounded-xl bg-primary py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors shadow-sm"
        >
          Register / Participate
        </button>
      </div>
    </div>
  );
}

function ParticipationModal({ event, onClose, onSuccess }) {
  const { user } = useAuth();
  const [formData, setFormData] = useState({
    name: user?.name || "",
    email: user?.email || "",
    phone: user?.phone || "",
    occupation: user?.occupation || "Volunteer",
    skills: user?.skills ? user.skills.join(", ") : "Community Service",
  });
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await eventService.registerForEvent(event._id || event.id);
      setSubmitted(true);
      if (onSuccess) onSuccess();
    } catch (err) {
      setError(err.message || "Failed to register for event.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-2xl border border-border bg-card shadow-2xl overflow-hidden">
        <div className="flex items-start justify-between border-b border-border p-5 bg-muted/30">
          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-primary">Event Registration</p>
            <h3 className="mt-0.5 text-lg font-bold">{event.title}</h3>
            <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <Calendar className="h-3 w-3 text-primary" /> {event.date}
              </span>
              <span className="flex items-center gap-1">
                <MapPin className="h-3 w-3 text-primary" /> {event.location}
              </span>
            </div>
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 hover:bg-muted transition-colors">
            <X className="h-5 w-5 text-muted-foreground" />
          </button>
        </div>

        {submitted ? (
          <div className="flex flex-col items-center justify-center gap-3 p-10 text-center space-y-2">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-100 text-green-600">
              <CheckCircle2 className="h-9 w-9" />
            </div>
            <h4 className="text-xl font-bold">Successfully Registered!</h4>
            <p className="max-w-xs text-sm text-muted-foreground">
              Your registration for <strong>{event.title}</strong> has been recorded in MongoDB. The NGO team looks forward to having you!
            </p>
            <button
              onClick={onClose}
              className="mt-4 rounded-xl bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
            >
              Done
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="mb-1 block text-xs font-semibold">Full Name</label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full rounded-xl border border-border bg-background py-2 pl-9 pr-3 text-sm outline-none focus:border-primary"
                    required
                  />
                </div>
              </div>
              <div>
                <label className="mb-1 block text-xs font-semibold">Occupation</label>
                <div className="relative">
                  <Briefcase className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="text"
                    value={formData.occupation}
                    onChange={(e) => setFormData({ ...formData, occupation: e.target.value })}
                    className="w-full rounded-xl border border-border bg-background py-2 pl-9 pr-3 text-sm outline-none focus:border-primary"
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="mb-1 block text-xs font-semibold">Skills Provided</label>
              <input
                type="text"
                value={formData.skills}
                onChange={(e) => setFormData({ ...formData, skills: e.target.value })}
                className="w-full rounded-xl border border-border bg-background py-2 px-3 text-sm outline-none focus:border-primary"
              />
            </div>

            {error && <p className="text-xs font-semibold text-destructive">{error}</p>}

            <div className="flex gap-3 pt-2">
              <button
                type="button"
                onClick={onClose}
                className="flex-1 rounded-xl border border-border py-2.5 text-sm font-semibold hover:bg-muted transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting}
                className="flex-1 rounded-xl bg-primary py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md flex items-center justify-center gap-2"
              >
                {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
                Confirm Registration
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [events, setEvents] = useState([]);
  const [aiEvents, setAiEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("All");
  const [participatingEvent, setParticipatingEvent] = useState(null);

  const fetchEventsData = async () => {
    setLoading(true);
    try {
      const res = await eventService.getAllEvents({ search, category: categoryFilter });
      if (res.events) setEvents(res.events);

      const aiRes = await aiService.getEventRecommendations({});
      if (aiRes.recommendations) setAiEvents(aiRes.recommendations);
    } catch (err) {
      console.error("Dashboard fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEventsData();
  }, [categoryFilter]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    fetchEventsData();
  };

  return (
    <div className="space-y-10">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight">NGO Activities & Programs</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Discover community initiatives organized by NGOs and get AI-matched event recommendations.
          </p>
        </div>

        {/* Search & Filter */}
        <form onSubmit={handleSearchSubmit} className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search events or location..."
              className="w-full md:w-64 rounded-xl border border-border bg-background py-2 pl-9 pr-3 text-xs outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
            />
          </div>
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="rounded-xl border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary"
          >
            <option value="All">All Categories</option>
            <option value="Environmental Protection">Environmental</option>
            <option value="Education & Literacy">Education</option>
            <option value="Healthcare & Wellness">Healthcare</option>
          </select>
        </form>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20">
          <Loader2 className="h-8 w-8 text-primary animate-spin mb-2" />
          <p className="text-sm text-muted-foreground">Fetching events & AI recommendations...</p>
        </div>
      ) : (
        <>
          {/* AI Recommended Section */}
          <section className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-md">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-bold">AI Recommended Events For You</h2>
                <p className="text-xs text-muted-foreground">
                  Matched based on your registered skills, cause interests, and location
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {aiEvents.slice(0, 3).map((event) => (
                <EventCard
                  key={`ai-${event._id || event.id}`}
                  event={event}
                  recommended
                  reason={event.reason}
                  onParticipate={setParticipatingEvent}
                />
              ))}
            </div>
          </section>

          {/* All Available Events Section */}
          <section className="space-y-4">
            <h2 className="text-lg font-bold">All Available Events ({events.length})</h2>
            {events.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center text-muted-foreground">
                No events found matching your search.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {events.map((event) => (
                  <EventCard
                    key={event._id || event.id}
                    event={event}
                    onParticipate={setParticipatingEvent}
                  />
                ))}
              </div>
            )}
          </section>
        </>
      )}

      {/* Participation Modal */}
      {participatingEvent && (
        <ParticipationModal
          event={participatingEvent}
          onClose={() => setParticipatingEvent(null)}
          onSuccess={fetchEventsData}
        />
      )}
    </div>
  );
}
