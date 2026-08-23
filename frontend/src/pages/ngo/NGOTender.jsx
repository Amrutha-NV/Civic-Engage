import React, { useState } from "react";
import { Sparkles, Download, Pencil, Check, RotateCcw, FileText, DollarSign, Package, MessageSquare, ChevronRight, Loader2 } from "lucide-react";
import { aiService } from "../../services/api";

const eventTypes = [
  "Environmental Protection Campaign",
  "Education Program & School Outreach",
  "Community Health & Medical Fair",
  "Disaster Relief & Rehabilitation",
  "Youth Development Workshop",
  "Food Distribution Drive",
  "Plantation & Reforestation Drive",
];

export default function NGOTender() {
  const [form, setForm] = useState({
    eventType: "Environmental Protection Campaign",
    requirements: "Need 500 trees saplings, shovel tools, gloves, and volunteer catering for 100 people.",
    budget: "150000",
    items: "Garden gloves (100 pairs)\nTree saplings (500 units)\nShovels and Digging tools (30 pcs)\nVolunteer Lunch Catering (100 meals)\nFirst Aid Standby Kits (5 sets)",
    prompt: "Include standard Philippines procurement terms, SEC compliance preference, and Net 15 payment terms.",
  });

  const [step, setStep] = useState("form"); // "form" | "generating" | "result"
  const [document, setDocument] = useState("");
  const [editing, setEditing] = useState(false);
  const [editedDoc, setEditedDoc] = useState("");
  const [downloaded, setDownloaded] = useState(false);
  const [error, setError] = useState("");

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!form.eventType || !form.budget) {
      setError("Event Type and Budget are required.");
      return;
    }

    setError("");
    setStep("generating");

    try {
      const res = await aiService.generateTender(form);
      if (res.document) {
        setDocument(res.document);
        setEditedDoc(res.document);
        setStep("result");
      }
    } catch (err) {
      setError(err.message || "Failed to generate tender document.");
      setStep("form");
    }
  };

  const handleDownload = () => {
    const textToDownload = editing ? editedDoc : document;
    const blob = new Blob([textToDownload], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = window.document.createElement("a");
    link.href = url;
    link.download = `CivicEngage_Tender_RFQ_${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
    setDownloaded(true);
    setTimeout(() => setDownloaded(false), 2000);
  };

  const handleReset = () => {
    setStep("form");
    setDocument("");
    setEditedDoc("");
    setEditing(false);
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto text-foreground">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-blue-600 shadow-md">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          <h1 className="text-2xl font-extrabold">AI Tender & RFQ Generator</h1>
        </div>
        <p className="text-sm text-muted-foreground ml-13">
          Automatically generate structured Request for Quotation (RFQ) procurement documentation for NGO projects
        </p>
      </div>

      {/* Progress Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground bg-muted/40 p-3 rounded-xl border border-border">
        <span className={step === "form" ? "font-bold text-foreground" : ""}>1. Fill Project Details</span>
        <ChevronRight className="h-4 w-4" />
        <span className={step === "generating" ? "font-bold text-foreground" : ""}>2. AI Processing</span>
        <ChevronRight className="h-4 w-4" />
        <span className={step === "result" ? "font-bold text-foreground" : ""}>3. Review, Edit & Export</span>
      </div>

      {/* Step 1: Form */}
      {step === "form" && (
        <form onSubmit={handleGenerate} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-5">
              <div className="rounded-2xl border border-border bg-card p-5 shadow-sm space-y-4">
                <h3 className="flex items-center gap-2 text-sm font-bold">
                  <FileText className="h-4 w-4 text-blue-600" /> Event Information
                </h3>

                <div>
                  <label className="mb-1 block text-xs font-semibold">Event / Campaign Type *</label>
                  <select
                    value={form.eventType}
                    onChange={(e) => setForm({ ...form, eventType: e.target.value })}
                    className="w-full rounded-xl border border-border bg-background p-2.5 text-sm outline-none focus:border-blue-600"
                    required
                  >
                    {eventTypes.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="mb-1 block text-xs font-semibold">Scope & Requirements</label>
                  <textarea
                    value={form.requirements}
                    onChange={(e) => setForm({ ...form, requirements: e.target.value })}
                    rows={3}
                    className="w-full resize-none rounded-xl border border-border bg-background p-3 text-sm outline-none focus:border-blue-600"
                    placeholder="Describe main goals..."
                  />
                </div>
              </div>

              <div className="rounded-2xl border border-border bg-card p-5 shadow-sm space-y-4">
                <h3 className="flex items-center gap-2 text-sm font-bold">
                  <DollarSign className="h-4 w-4 text-green-600" /> Budget Allocation
                </h3>

                <div>
                  <label className="mb-1 block text-xs font-semibold">Total Approved Budget (PHP) *</label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm font-bold text-muted-foreground">₱</span>
                    <input
                      type="number"
                      value={form.budget}
                      onChange={(e) => setForm({ ...form, budget: e.target.value })}
                      placeholder="150000"
                      className="w-full rounded-xl border border-border bg-background py-2.5 pl-8 pr-3 text-sm outline-none focus:border-blue-600"
                      required
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-5">
              <div className="rounded-2xl border border-border bg-card p-5 shadow-sm space-y-4">
                <h3 className="flex items-center gap-2 text-sm font-bold">
                  <Package className="h-4 w-4 text-purple-600" /> Items & Equipment List
                </h3>

                <div>
                  <label className="mb-1 block text-xs font-semibold">Required Goods / Services (One per line)</label>
                  <textarea
                    value={form.items}
                    onChange={(e) => setForm({ ...form, items: e.target.value })}
                    rows={5}
                    className="w-full resize-none rounded-xl border border-border bg-background p-3 font-mono text-xs leading-relaxed outline-none focus:border-blue-600"
                  />
                </div>
              </div>

              <div className="rounded-2xl border border-border bg-card p-5 shadow-sm space-y-4">
                <h3 className="flex items-center gap-2 text-sm font-bold">
                  <MessageSquare className="h-4 w-4 text-amber-500" /> Custom AI Parameters
                </h3>

                <div>
                  <label className="mb-1 block text-xs font-semibold">Additional Legal / Vendor Instructions</label>
                  <textarea
                    value={form.prompt}
                    onChange={(e) => setForm({ ...form, prompt: e.target.value })}
                    rows={2}
                    className="w-full resize-none rounded-xl border border-border bg-background p-3 text-xs outline-none focus:border-blue-600"
                  />
                </div>
              </div>
            </div>
          </div>

          {error && <p className="text-xs font-semibold text-destructive">{error}</p>}

          <div className="flex justify-end">
            <button
              type="submit"
              className="flex items-center gap-2 rounded-xl bg-blue-600 px-8 py-3.5 text-sm font-semibold text-white hover:bg-blue-700 transition-all shadow-md"
            >
              <Sparkles className="h-4 w-4" />
              Generate Tender RFQ Document
            </button>
          </div>
        </form>
      )}

      {/* Step 2: Generating */}
      {step === "generating" && (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-border bg-card py-20 text-center shadow-sm space-y-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-100 text-blue-600 shadow-md">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
          <h3 className="text-lg font-bold">Generating RFQ Procurement Document</h3>
          <p className="max-w-sm text-xs text-muted-foreground">
            Formatting procurement framework, calculating budget tables, and compiling terms...
          </p>
        </div>
      )}

      {/* Step 3: Result */}
      {step === "result" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between rounded-2xl border border-border bg-card p-4 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-green-100 text-green-700">
                <Check className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm font-bold">RFQ Document Ready</p>
                <p className="text-xs text-muted-foreground">Review document below, edit inline, or export as text file.</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleReset}
                className="flex items-center gap-1.5 rounded-xl border border-border px-3 py-2 text-xs font-semibold hover:bg-muted transition-colors"
              >
                <RotateCcw className="h-4 w-4" /> New Document
              </button>

              <button
                onClick={() => {
                  setEditing((v) => !v);
                  if (editing) setDocument(editedDoc);
                }}
                className={`flex items-center gap-1.5 rounded-xl px-4 py-2 text-xs font-semibold transition-all ${
                  editing ? "bg-green-600 text-white" : "border border-border hover:bg-muted"
                }`}
              >
                <Pencil className="h-4 w-4" />
                {editing ? "Save Edits" : "Edit Text"}
              </button>

              <button
                onClick={handleDownload}
                className="flex items-center gap-1.5 rounded-xl bg-blue-600 px-5 py-2 text-xs font-semibold text-white hover:bg-blue-700 transition-colors shadow-md"
              >
                <Download className="h-4 w-4" />
                {downloaded ? "Downloaded!" : "Download (.txt)"}
              </button>
            </div>
          </div>

          {editing ? (
            <textarea
              value={editedDoc}
              onChange={(e) => setEditedDoc(e.target.value)}
              className="w-full rounded-2xl border border-border bg-card p-6 font-mono text-xs leading-relaxed outline-none focus:border-blue-600 shadow-sm"
              rows={35}
            />
          ) : (
            <div className="rounded-2xl border border-border bg-card p-6 shadow-sm overflow-x-auto">
              <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-foreground">
                {document}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
