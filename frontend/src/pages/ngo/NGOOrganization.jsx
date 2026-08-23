import React from "react";
import { Building2, Mail, Phone, MapPin, ShieldCheck, Check } from "lucide-react";
import { useAuth } from "../../context/AuthContext";

export default function NGOOrganization() {
  const { ngo } = useAuth();

  return (
    <div className="space-y-6 max-w-4xl mx-auto text-foreground">
      <div>
        <h1 className="text-2xl font-extrabold">Organization Profile</h1>
        <p className="text-sm text-muted-foreground">
          Official details and contact information for your non-governmental organization
        </p>
      </div>

      <div className="rounded-2xl border border-border bg-card p-6 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row items-center sm:items-start gap-5 pb-6 border-b border-border">
          <img
            src={ngo?.logo || "https://images.unsplash.com/photo-1599305445671-ac291c95aaa9?w=150"}
            alt={ngo?.ngoName}
            className="h-24 w-24 rounded-2xl object-cover border-2 border-blue-600 shadow-md"
          />
          <div className="space-y-1.5 text-center sm:text-left flex-1">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h2 className="text-xl font-bold">{ngo?.ngoName || "GreenFuture Philippines Foundation"}</h2>
              <span className="inline-flex items-center gap-1 rounded-full bg-blue-50 border border-blue-200 px-3 py-1 text-xs font-bold text-blue-700">
                <ShieldCheck className="h-3.5 w-3.5" /> SEC Verified NGO
              </span>
            </div>
            <p className="text-xs font-semibold text-blue-600">{ngo?.category || "Environmental Protection"}</p>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {ngo?.description || "Dedicated NGO working towards community improvement and youth empowerment."}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="rounded-xl border border-border p-4 space-y-1">
            <span className="text-muted-foreground block flex items-center gap-1">
              <Mail className="h-3.5 w-3.5 text-blue-600" /> Official Email
            </span>
            <span className="font-bold text-sm">{ngo?.email}</span>
          </div>

          <div className="rounded-xl border border-border p-4 space-y-1">
            <span className="text-muted-foreground block flex items-center gap-1">
              <Phone className="h-3.5 w-3.5 text-blue-600" /> Contact Phone
            </span>
            <span className="font-bold text-sm">{ngo?.phone}</span>
          </div>

          <div className="col-span-1 md:col-span-2 rounded-xl border border-border p-4 space-y-1">
            <span className="text-muted-foreground block flex items-center gap-1">
              <MapPin className="h-3.5 w-3.5 text-blue-600" /> Headquarters Address
            </span>
            <span className="font-bold text-sm">{ngo?.address}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
