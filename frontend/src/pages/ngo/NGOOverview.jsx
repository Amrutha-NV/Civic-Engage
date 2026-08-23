import React, { useState, useEffect } from "react";
import {
  Plus,
  Search,
  MapPin,
  Calendar,
  Clock,
  Pencil,
  Trash2,
  Eye,
  X,
  Sparkles,
  Filter,
  Megaphone,
  Mail,
  Phone,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  UserCheck,
  Users,
  BadgeCheck,
  Loader2,
  Upload,
} from "lucide-react";
import { eventService, aiService, uploadService } from "../../services/api";
import { useAuth } from "../../context/AuthContext";

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

function BreakdownBar({ label, value }) {
  const color = value >= 90 ? "bg-green-500" : value >= 75 ? "bg-blue-500" : "bg-yellow-500";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 shrink-0 text-muted-foreground">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${value}%` }} />
      </div>
      <span className="w-8 shrink-0 text-right font-semibold tabular-nums">{value}%</span>
    </div>
  );
}

function VolunteerCard({ rec, assigned, onAssign }) {
  const [expanded, setExpanded] = useState(false);

  const matchPercent = rec.matchPercent || 85;
  const matchColor =
    matchPercent >= 90
      ? "bg-green-100 text-green-700 border-green-200"
      : matchPercent >= 80
      ? "bg-blue-100 text-blue-700 border-blue-200"
      : "bg-yellow-100 text-yellow-700 border-yellow-200";

  return (
    <div
      className={`rounded-2xl border bg-card p-4 shadow-sm transition-all hover:shadow-md space-y-3 ${
        assigned ? "border-green-300 bg-green-50/30" : "border-border"
      }`}
    >
      <div className="flex items-start gap-3">
        <div className="relative shrink-0">
          <img
            src={rec.avatar || "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150"}
            alt={rec.name}
            className="h-12 w-12 rounded-full object-cover border border-border"
          />
          <div className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-blue-600">
            <Sparkles className="h-2.5 w-2.5 text-white" />
          </div>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 flex-wrap">
            <div>
              <p className="font-bold text-sm">{rec.name}</p>
              <p className="text-xs text-muted-foreground">{rec.occupation || "Volunteer"}</p>
            </div>
            <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-extrabold ${matchColor}`}>
              {matchPercent}% AI Match
            </span>
          </div>

          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
            {rec.email && (
              <span className="flex items-center gap-1">
                <Mail className="h-3 w-3" /> {rec.email}
              </span>
            )}
            {rec.phone && (
              <span className="flex items-center gap-1">
                <Phone className="h-3 w-3" /> {rec.phone}
              </span>
            )}
            {rec.location && (
              <span className="flex items-center gap-1">
                <MapPin className="h-3 w-3" /> {rec.location}
                {rec.distanceKm !== undefined && (
                  <span className="text-[10px] text-muted-foreground">({rec.distanceKm}km)</span>
                )}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="space-y-1">
        <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Skills</p>
        <div className="flex flex-wrap gap-1">
          {(rec.skills || []).map((s) => (
            <span key={s} className="rounded-full bg-blue-50 px-2 py-0.5 text-[11px] font-semibold text-blue-700 border border-blue-100">
              {s}
            </span>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-border bg-muted/20">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="flex w-full items-center justify-between px-3 py-2 text-xs font-semibold text-foreground hover:bg-muted/50 rounded-xl transition-colors"
        >
          <span className="flex items-center gap-1.5">
            <Sparkles className="h-3 w-3 text-blue-600" />
            Why this volunteer?
          </span>
          {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
        </button>

        {expanded && (
          <div className="px-3 pb-3 space-y-2 text-xs border-t border-border/50 pt-2">
            <p className="text-muted-foreground italic">"{rec.whyRecommended || "Strong match for event requirements."}"</p>
            {rec.breakdown && (
              <div className="space-y-1.5 pt-1">
                <BreakdownBar label="Skills Match" value={rec.breakdown.skills ?? 85} />
                <BreakdownBar label="Interests" value={rec.breakdown.interests ?? 80} />
                <BreakdownBar label="Availability" value={rec.breakdown.availability ?? 90} />
                <BreakdownBar label="Proximity" value={rec.breakdown.proximity ?? (rec.distanceKm ? Math.max(65, Math.min(99, Math.round(100 / (1 + rec.distanceKm / 15)))) : 85)} />
              </div>
            )}

          </div>
        )}
      </div>

      <div className="flex gap-2 pt-1">
        {assigned ? (
          <button
            disabled
            className="w-full flex items-center justify-center gap-1.5 rounded-xl bg-green-600 py-2 text-xs font-semibold text-white shadow-sm cursor-default"
          >
            <BadgeCheck className="h-3.5 w-3.5" />
            Volunteer Assigned
          </button>
        ) : (
          <button
            onClick={onAssign}
            className="w-full flex items-center justify-center gap-1.5 rounded-xl bg-blue-600 py-2 text-xs font-semibold text-white hover:bg-blue-700 transition-colors shadow-sm"
          >
            <UserCheck className="h-3.5 w-3.5" />
            Assign / Invite Volunteer
          </button>
        )}
      </div>
    </div>
  );
}

function AIRecommendationsSection({ event }) {
  const [volunteers, setVolunteers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [assignedIds, setAssignedIds] = useState([]);

  const fetchVolunteers = async () => {
    setLoading(true);
    try {
      const res = await aiService.getVolunteerRecommendations(event._id || event.id);
      if (res.volunteers) {
        setVolunteers(res.volunteers);
      }
    } catch (err) {
      console.log("Volunteer rec fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (event) fetchVolunteers();
  }, [event]);

  return (
    <div className="p-6 space-y-5 border-t border-border bg-muted/10">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold">AI Recommended Volunteers</h3>
            <p className="text-xs text-muted-foreground">Ranked by skills, interests, and availability for this event</p>
          </div>
        </div>

        <button
          onClick={fetchVolunteers}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-xl border border-border px-3 py-1.5 text-xs font-semibold hover:bg-muted transition-colors disabled:opacity-60"
        >
          <RotateCcw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh Matching
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-10">
          <Loader2 className="h-6 w-6 text-blue-600 animate-spin mr-2" />
          <span className="text-xs text-muted-foreground">Running AI Volunteer Matching Pipeline...</span>
        </div>
      ) : volunteers.length === 0 ? (
        <div className="text-center py-8 text-xs text-muted-foreground">
          No volunteer matches found.
        </div>
      ) : (
        <div className="space-y-4">
          {volunteers.map((rec) => {
            const vId = rec.userId || rec.id;
            return (
              <VolunteerCard
                key={vId}
                rec={rec}
                assigned={assignedIds.includes(vId)}
                onAssign={() => setAssignedIds((prev) => [...prev, vId])}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

function EventDetailPanel({ event, onClose, onEdit }) {
  const volunteersJoined = event.volunteersJoined || 0;
  const capacity = event.capacity || 50;
  const pct = Math.round((volunteersJoined / capacity) * 100);

  return (
    <div className="fixed inset-0 z-50 flex items-stretch justify-end bg-black/40 backdrop-blur-sm">
      <div className="flex h-full w-full max-w-2xl flex-col bg-card shadow-2xl overflow-hidden text-foreground">
        <div className="relative h-48 bg-muted shrink-0">
          <img src={event.banner} alt={event.title} className="h-full w-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
          <button
            onClick={onClose}
            className="absolute right-4 top-4 flex h-8 w-8 items-center justify-center rounded-full bg-black/50 text-white hover:bg-black/70 transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
          <div className="absolute bottom-3 left-4">
            <StatusBadge status={event.status} />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-4">
            <div>
              <span className="rounded-full bg-blue-100 px-3 py-1 text-xs font-bold text-blue-700">
                {event.category}
              </span>
              <h2 className="mt-2 text-xl font-bold">{event.title}</h2>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{event.description}</p>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-xl border border-border p-3">
                <span className="text-muted-foreground block flex items-center gap-1">
                  <Calendar className="h-3.5 w-3.5 text-blue-600" /> Date
                </span>
                <span className="font-bold text-sm text-foreground">{event.date}</span>
              </div>
              <div className="rounded-xl border border-border p-3">
                <span className="text-muted-foreground block flex items-center gap-1">
                  <Clock className="h-3.5 w-3.5 text-blue-600" /> Time
                </span>
                <span className="font-bold text-sm text-foreground">
                  {event.startTime} - {event.endTime}
                </span>
              </div>
              <div className="col-span-2 rounded-xl border border-border p-3">
                <span className="text-muted-foreground block flex items-center gap-1">
                  <MapPin className="h-3.5 w-3.5 text-blue-600" /> Location
                </span>
                <span className="font-bold text-sm text-foreground">{event.location}</span>
              </div>
            </div>

            <div>
              <p className="mb-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">Required Skills</p>
              <div className="flex flex-wrap gap-1.5">
                {(event.requiredSkills || []).map((s) => (
                  <span key={s} className="rounded-full bg-muted px-3 py-1 text-xs font-semibold text-muted-foreground">
                    {s}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                <span>Volunteer Registrations</span>
                <span className="font-bold text-foreground">
                  {volunteersJoined} / {capacity}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${pct}%` }} />
              </div>
            </div>
          </div>

          <AIRecommendationsSection event={event} />
        </div>

        <div className="shrink-0 border-t border-border p-4 bg-card flex gap-3">
          <button
            onClick={onEdit}
            className="flex-1 rounded-xl bg-blue-600 py-3 text-sm font-semibold text-white hover:bg-blue-700 transition-colors shadow-md flex items-center justify-center gap-2"
          >
            <Pencil className="h-4 w-4" /> Edit Event
          </button>
        </div>
      </div>
    </div>
  );
}

function CreateEventModal({ eventToEdit, onClose, onSuccess }) {
  const [formData, setFormData] = useState({
    title: eventToEdit?.title || "",
    description: eventToEdit?.description || "",
    category: eventToEdit?.category || "Environmental Protection",
    requiredSkills: eventToEdit?.requiredSkills ? eventToEdit.requiredSkills.join(", ") : "Environmental Science, Event Planning",
    location: eventToEdit?.location || "Manila",
    date: eventToEdit?.date || "Saturday, Oct 15, 2026",
    startTime: eventToEdit?.startTime || "08:00 AM",
    endTime: eventToEdit?.endTime || "04:00 PM",
    banner: eventToEdit?.banner || "https://images.unsplash.com/photo-1559027615-cd4628902d4a?w=800",
    capacity: eventToEdit?.capacity || 50,
  });

  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const res = await uploadService.uploadImage(file);
      const imageUrl = typeof res === "string" ? res : (res?.url || res?.data?.url || res?.secure_url);
      if (imageUrl && typeof imageUrl === "string") {
        setFormData((prev) => ({ ...prev, banner: imageUrl }));
      }
    } catch (err) {
      console.error("Image upload error:", err);
    } finally {
      setUploading(false);
    }
  };


  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.title || !formData.description || !formData.location || !formData.date) {
      setError("Please fill out required fields.");
      return;
    }
    setError("");
    setSubmitting(true);
    try {
      if (eventToEdit) {
        await eventService.updateEvent(eventToEdit._id || eventToEdit.id, formData);
      } else {
        await eventService.createEvent(formData);
      }
      onSuccess();
    } catch (err) {
      setError(err.message || "Failed to save event.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-xl rounded-2xl border border-border bg-card shadow-2xl overflow-hidden text-foreground">
        <div className="flex items-center justify-between border-b border-border p-5 bg-muted/30">
          <h3 className="text-lg font-bold">{eventToEdit ? "Edit Event" : "Create New NGO Event"}</h3>
          <button onClick={onClose} className="rounded-lg p-1.5 hover:bg-muted transition-colors">
            <X className="h-5 w-5 text-muted-foreground" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4 max-h-[80vh] overflow-y-auto">
          <div>
            <label className="mb-1 block text-xs font-semibold">Event Title *</label>
            <input
              type="text"
              value={formData.title}
              onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              placeholder="e.g. Marikina Watershed Reforestation Drive"
              className="w-full rounded-xl border border-border bg-background py-2.5 px-3 text-sm outline-none focus:border-blue-600"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-1 block text-xs font-semibold">Category</label>
              <select
                value={formData.category}
                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                className="w-full rounded-xl border border-border bg-background py-2.5 px-3 text-sm outline-none focus:border-blue-600"
              >
                <option value="Environmental Protection">Environmental Protection</option>
                <option value="Education & Literacy">Education & Literacy</option>
                <option value="Healthcare & Wellness">Healthcare & Wellness</option>
                <option value="Community Development">Community Development</option>
              </select>
            </div>

            <div>
              <label className="mb-1 block text-xs font-semibold">Volunteer Capacity</label>
              <input
                type="number"
                value={formData.capacity}
                onChange={(e) => setFormData({ ...formData, capacity: e.target.value })}
                className="w-full rounded-xl border border-border bg-background py-2.5 px-3 text-sm outline-none focus:border-blue-600"
              />
            </div>
          </div>

          <div>
            <label className="mb-1 block text-xs font-semibold">Required Skills (Comma separated)</label>
            <input
              type="text"
              value={formData.requiredSkills}
              onChange={(e) => setFormData({ ...formData, requiredSkills: e.target.value })}
              placeholder="Teaching, Technology, First Aid"
              className="w-full rounded-xl border border-border bg-background py-2.5 px-3 text-sm outline-none focus:border-blue-600"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-1 block text-xs font-semibold">Date *</label>
              <input
                type="text"
                value={formData.date}
                onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                placeholder="Saturday, Oct 15, 2026"
                className="w-full rounded-xl border border-border bg-background py-2.5 px-3 text-sm outline-none focus:border-blue-600"
                required
              />
            </div>

            <div>
              <label className="mb-1 block text-xs font-semibold">Location *</label>
              <input
                type="text"
                value={formData.location}
                onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                placeholder="Marikina City"
                className="w-full rounded-xl border border-border bg-background py-2.5 px-3 text-sm outline-none focus:border-blue-600"
                required
              />
            </div>
          </div>

          <div>
            <label className="mb-1 block text-xs font-semibold">Event Banner Image (Cloudinary Upload or URL)</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={formData.banner}
                onChange={(e) => setFormData({ ...formData, banner: e.target.value })}
                placeholder="https://res.cloudinary.com/..."
                className="flex-1 rounded-xl border border-border bg-background py-2 px-3 text-xs outline-none focus:border-blue-600"
              />
              <label className="flex items-center gap-1.5 rounded-xl border border-border bg-muted px-3 py-2 text-xs font-semibold cursor-pointer hover:bg-muted/80 transition-colors">
                {uploading ? <Loader2 className="h-4 w-4 animate-spin text-blue-600" /> : <Upload className="h-4 w-4 text-blue-600" />}
                {uploading ? "Uploading..." : "Upload Image"}
                <input type="file" accept="image/*" onChange={handleImageUpload} disabled={uploading} className="hidden" />
              </label>
            </div>
            {formData.banner && (
              <div className="mt-2 relative rounded-xl overflow-hidden border border-border h-24 bg-muted/20">
                <img src={formData.banner} alt="Event Banner Preview" className="w-full h-full object-cover" />
                <span className="absolute bottom-1.5 right-1.5 bg-black/60 backdrop-blur-sm text-white text-[10px] px-2 py-0.5 rounded-md">
                  Cloudinary Banner Preview
                </span>
              </div>
            )}
          </div>


          <div>
            <label className="mb-1 block text-xs font-semibold">Full Description *</label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              rows={3}
              className="w-full resize-none rounded-xl border border-border bg-background p-3 text-sm outline-none focus:border-blue-600"
              required
            />
          </div>

          {error && <p className="text-xs font-semibold text-destructive">{error}</p>}

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl border border-border px-5 py-2.5 text-sm font-semibold hover:bg-muted transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="rounded-xl bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 transition-all shadow-md disabled:opacity-60 flex items-center gap-2"
            >
              {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
              {eventToEdit ? "Update Event" : "Create Event"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default function NGOOverview() {
  const { ngo } = useAuth();
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [eventToEdit, setEventToEdit] = useState(null);

  const fetchNGOEvents = async () => {
    setLoading(true);
    try {
      const res = await eventService.getAllEvents({
        search,
        status: statusFilter,
        ngoId: ngo?._id,
      });
      if (res.events) setEvents(res.events);
    } catch (err) {
      console.log("Fetch NGO events error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNGOEvents();
  }, [statusFilter]);

  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this event?")) return;
    try {
      await eventService.deleteEvent(id);
      fetchNGOEvents();
    } catch (err) {
      alert(err.message || "Failed to delete event.");
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold">NGO Events & AI Volunteer Matching</h1>
          <p className="text-sm text-muted-foreground">
            Create events, view registrations, and inspect ranked AI volunteer recommendations
          </p>
        </div>

        <button
          onClick={() => {
            setEventToEdit(null);
            setShowCreateModal(true);
          }}
          className="flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 transition-colors shadow-md shrink-0"
        >
          <Plus className="h-4 w-4" /> Create New Event
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative flex-1 min-w-52 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search events..."
            className="w-full rounded-xl border border-border bg-background py-2 pl-9 pr-3 text-xs outline-none focus:border-blue-600"
          />
        </div>

        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="rounded-xl border border-border bg-background px-3 py-2 text-xs outline-none focus:border-blue-600"
        >
          <option value="All">All Statuses</option>
          <option value="Active">Active</option>
          <option value="Upcoming">Upcoming</option>
          <option value="Completed">Completed</option>
        </select>
      </div>

      {/* Events Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 text-blue-600 animate-spin mr-2" />
          <span className="text-sm text-muted-foreground">Loading NGO events...</span>
        </div>
      ) : events.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center space-y-3">
          <Megaphone className="h-10 w-10 text-muted-foreground mx-auto" />
          <h3 className="text-lg font-bold">No Events Found</h3>
          <p className="text-sm text-muted-foreground">Create your first NGO event to start matching volunteers.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {events.map((event) => {
            const volunteersJoined = event.volunteersJoined || 0;
            const capacity = event.capacity || 50;
            const pct = Math.round((volunteersJoined / capacity) * 100);
            const id = event._id || event.id;

            return (
              <div
                key={id}
                className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm hover:shadow-md transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="relative h-40 bg-muted">
                    <img src={event.banner} alt={event.title} className="h-full w-full object-cover" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
                    <div className="absolute bottom-2 left-3">
                      <StatusBadge status={event.status} />
                    </div>
                  </div>

                  <div className="p-4 space-y-3">
                    <h3 className="font-bold text-base leading-tight">{event.title}</h3>
                    <p className="text-xs text-muted-foreground line-clamp-2">{event.shortDesc || event.description}</p>

                    <div className="space-y-1 text-xs text-muted-foreground">
                      <div className="flex items-center gap-1.5">
                        <Calendar className="h-3.5 w-3.5 text-blue-600" />
                        <span>{event.date}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <MapPin className="h-3.5 w-3.5 text-blue-600" />
                        <span className="truncate">{event.location}</span>
                      </div>
                    </div>

                    <div>
                      <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                        <span>Volunteers</span>
                        <span className="font-bold text-foreground">
                          {volunteersJoined} / {capacity}
                        </span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                        <div className="h-full rounded-full bg-blue-600" style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex gap-2 p-4 pt-0">
                  <button
                    onClick={() => setSelectedEvent(event)}
                    className="flex flex-1 items-center justify-center gap-1.5 rounded-xl border border-border py-2 text-xs font-semibold hover:bg-muted transition-colors"
                  >
                    <Eye className="h-3.5 w-3.5" /> View & AI Matches
                  </button>
                  <button
                    onClick={() => {
                      setEventToEdit(event);
                      setShowCreateModal(true);
                    }}
                    className="flex items-center justify-center rounded-xl border border-blue-200 px-3 py-2 text-xs font-semibold text-blue-600 hover:bg-blue-50 transition-colors"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </button>
                  <button
                    onClick={() => handleDelete(id)}
                    className="flex items-center justify-center rounded-xl border border-red-200 px-3 py-2 text-xs font-semibold text-red-600 hover:bg-red-50 transition-colors"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Event Detail & AI Volunteer Recommendations Panel */}
      {selectedEvent && (
        <EventDetailPanel
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
          onEdit={() => {
            setEventToEdit(selectedEvent);
            setSelectedEvent(null);
            setShowCreateModal(true);
          }}
        />
      )}

      {/* Create / Edit Event Modal */}
      {showCreateModal && (
        <CreateEventModal
          eventToEdit={eventToEdit}
          onClose={() => setShowCreateModal(false)}
          onSuccess={() => {
            setShowCreateModal(false);
            fetchNGOEvents();
          }}
        />
      )}
    </div>
  );
}
