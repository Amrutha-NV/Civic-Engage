import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Loader2, ArrowLeft, MapPin, Calendar, Users, Building2 } from "lucide-react";
import { toast } from "react-toastify";
import api from "../lib/api";
import Badge from "../components/badge";
import Tag from "../components/tag";

export default function EventDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [event, setEvent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [joining, setJoining] = useState(false);

  useEffect(() => {
    const fetchEvent = async () => {
      try {
        const res = await api.get(`/events/${id}`);
        setEvent(res.data);
      } catch (error) {
        toast.error("Failed to load event details");
      } finally {
        setLoading(false);
      }
    };
    fetchEvent();
  }, [id]);

  const handleJoin = async () => {
    setJoining(true);
    try {
      await api.post(`/events/${id}/join`);
      toast.success("Successfully joined the event!");
      navigate("/tasks");
    } catch (error) {
      toast.error(error.response?.data?.message || "Failed to join event");
    } finally {
      setJoining(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="animate-spin text-primary w-10 h-10" />
      </div>
    );
  }

  if (!event) {
    return (
      <div className="p-8 text-center">
        <p className="mb-4 text-muted-foreground">Event not found.</p>
        <button onClick={() => navigate("/opportunities")} className="text-primary hover:underline">
          Go back to Opportunities
        </button>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" /> Back
      </button>

      <div className="rounded-xl border border-border bg-card p-8 shadow-sm">
        <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold mb-2">{event.title}</h1>
            <div className="flex items-center gap-4 text-muted-foreground">
              <span className="flex items-center gap-1"><Building2 className="h-4 w-4"/> {event.ngo || event.organizer || "Community"}</span>
              <span className="flex items-center gap-1"><MapPin className="h-4 w-4"/> {event.location || "Global/Online"}</span>
              {event.date && <span className="flex items-center gap-1"><Calendar className="h-4 w-4"/> {new Date(event.date).toLocaleDateString()}</span>}
            </div>
          </div>
          <Badge variant="success">Open for Volunteers</Badge>
        </div>

        <div className="mb-8">
          <h2 className="mb-3 text-xl font-semibold">About the Event</h2>
          <p className="text-foreground leading-relaxed whitespace-pre-wrap">{event.description || "No description provided."}</p>
        </div>

        {event.skills && event.skills.length > 0 && (
          <div className="mb-8">
            <h2 className="mb-3 text-xl font-semibold">Skills Required</h2>
            <div className="flex flex-wrap gap-2">
              {event.skills.map(skill => (
                <Tag key={skill}>{skill}</Tag>
              ))}
            </div>
          </div>
        )}

        <div className="flex justify-end pt-4 border-t border-border">
          <button
            onClick={handleJoin}
            disabled={joining}
            className="rounded-lg bg-primary px-8 py-3 text-lg font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center justify-center min-w-[150px]"
          >
            {joining ? <Loader2 className="animate-spin w-5 h-5" /> : "Join Now"}
          </button>
        </div>
      </div>
    </div>
  );
}
