import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Search, MapPin, Loader2, Filter } from "lucide-react";
import api from "../lib/api";
import Badge from "../components/badge";
import Tag from "../components/tag";

export default function Opportunities() {
  const [opportunities, setOpportunities] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchOpportunities = async () => {
      try {
        const res = await api.get("/events");
        console.log(res);
        setOpportunities(res.data.events || []);
      } catch (error) {
        console.error("Failed to fetch events", error);
      } finally {
        setLoading(false);
      }
    };
    fetchOpportunities();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold">Opportunities</h1>
          <p className="mt-2 text-muted-foreground">
            Find volunteer campaigns and events to join.
          </p>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center p-20">
          <Loader2 className="animate-spin text-primary w-10 h-10" />
        </div>
      ) : opportunities.length === 0 ? (
        <div className="rounded-xl border border-border bg-card p-10 text-center">
          <p className="text-muted-foreground">
            No opportunities available at this time.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {opportunities.map((opp) => (
            <div
              key={opp._id}
              className="group flex flex-col rounded-xl border border-border bg-card p-6 shadow-sm transition-all hover:shadow-lg hover:border-primary"
            >
              <div className="mb-4 flex-1">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-lg line-clamp-2">
                    {opp.title}
                  </h3>
                  <Badge variant="success">Open</Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  {opp.ngo || opp.organizer || "Community Event"}
                </p>
              </div>
              <div className="mb-4">
                <p className="text-sm text-foreground line-clamp-3 mb-4">
                  {opp.description}
                </p>
              </div>
              <div className="mb-4 flex flex-wrap gap-2">
                {(opp.skills || []).map((skill, i) => (
                  <Tag key={i}>{skill}</Tag>
                ))}
              </div>
              <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
                <MapPin className="h-4 w-4" />
                <span>{opp.location || "Online"}</span>
              </div>
              <Link
                to={`/opportunities/${opp._id}`}
                className="w-full text-center block rounded-lg bg-primary px-4 py-2.5 text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                View Details
              </Link>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
