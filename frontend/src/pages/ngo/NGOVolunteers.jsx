import React, { useState, useEffect } from "react";
import { Users, Mail, Phone, MapPin, Search, Loader2 } from "lucide-react";
import { userService } from "../../services/api";

export default function NGOVolunteers() {
  const [volunteers, setVolunteers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");

  useEffect(() => {
    userService
      .getVolunteers()
      .then((res) => {
        if (res.volunteers) setVolunteers(res.volunteers);
      })
      .catch((err) => console.log("Volunteers fetch error:", err))
      .finally(() => setLoading(false));
  }, []);

  const filtered = volunteers.filter(
    (v) =>
      v.name?.toLowerCase().includes(search.toLowerCase()) ||
      v.skills?.some((s) => s.toLowerCase().includes(search.toLowerCase())) ||
      v.location?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6 max-w-5xl mx-auto text-foreground">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold">Registered Volunteers Directory</h1>
          <p className="text-sm text-muted-foreground">
            Browse all verified volunteers registered in the CivicEngage network
          </p>
        </div>

        <div className="relative flex-1 min-w-52 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search volunteers by name, skill, location..."
            className="w-full rounded-xl border border-border bg-background py-2 pl-9 pr-3 text-xs outline-none focus:border-blue-600"
          />
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 text-blue-600 animate-spin mr-2" />
          <span className="text-sm text-muted-foreground">Loading volunteer directory...</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center text-muted-foreground">
          No volunteers found.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {filtered.map((vol) => (
            <div key={vol._id || vol.id} className="rounded-2xl border border-border bg-card p-5 shadow-sm space-y-3">
              <div className="flex items-center gap-3">
                <img
                  src={vol.avatar || "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150"}
                  alt={vol.name}
                  className="h-12 w-12 rounded-full object-cover border border-border"
                />
                <div>
                  <h3 className="font-bold text-base">{vol.name}</h3>
                  <p className="text-xs text-muted-foreground">{vol.occupation || "Volunteer"}</p>
                </div>
              </div>

              <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                {vol.email && (
                  <span className="flex items-center gap-1">
                    <Mail className="h-3.5 w-3.5 text-blue-600" /> {vol.email}
                  </span>
                )}
                {vol.phone && (
                  <span className="flex items-center gap-1">
                    <Phone className="h-3.5 w-3.5 text-blue-600" /> {vol.phone}
                  </span>
                )}
                <span className="flex items-center gap-1">
                  <MapPin className="h-3.5 w-3.5 text-blue-600" /> {vol.location || "Manila"}
                </span>
              </div>

              <div>
                <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-1">Skills</p>
                <div className="flex flex-wrap gap-1">
                  {(vol.skills || []).map((s) => (
                    <span key={s} className="rounded-full bg-blue-50 text-blue-700 border border-blue-100 px-2.5 py-0.5 text-xs font-medium">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
