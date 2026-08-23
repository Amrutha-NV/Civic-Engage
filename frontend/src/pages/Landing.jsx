import React from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Users, HeartHandshake, Sparkles, Building2, ShieldCheck } from "lucide-react";

export default function Landing() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col justify-between">
      {/* Header / Navbar */}
      <header className="border-b border-border bg-card/80 backdrop-blur-md sticky top-0 z-50 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground font-bold text-xl shadow-md">
            C
          </div>
          <span className="text-2xl font-bold tracking-tight text-primary">CivicEngage</span>
        </div>
        <div className="flex items-center gap-4">
          <Link
            to="/login"
            className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors px-3 py-2"
          >
            User Login
          </Link>
          <Link
            to="/ngo-login"
            className="flex items-center gap-1.5 rounded-lg border border-border px-4 py-2 text-sm font-semibold hover:bg-muted transition-colors"
          >
            <Building2 className="h-4 w-4 text-primary" />
            NGO Portal
          </Link>
          <Link
            to="/signup"
            className="flex items-center gap-2 rounded-lg bg-primary px-5 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors shadow-sm"
          >
            Get Started
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main className="flex-1">
        <section className="px-6 py-20 md:py-28 max-w-6xl mx-auto text-center space-y-8">
          <div className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-1.5 text-xs font-semibold text-primary">
            <Sparkles className="h-3.5 w-3.5" />
            AI-Powered Civic Engagement & NGO Matchmaking
          </div>

          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight leading-tight">
            Connect Passionate Volunteers with Impactful <span className="text-primary">NGO Campaigns</span>
          </h1>

          <p className="max-w-2xl mx-auto text-lg text-muted-foreground leading-relaxed">
            CivicEngage bridges non-profit organizations and eager volunteers through AI recommendations,
            smart volunteer matching, streamlined mobile authentication, and automated tender generation.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Link
              to="/signup"
              className="w-full sm:w-auto flex items-center justify-center gap-2 rounded-xl bg-primary px-8 py-4 text-base font-semibold text-primary-foreground hover:bg-primary/90 shadow-lg transition-all"
            >
              Join as Volunteer
              <ArrowRight className="h-5 w-5" />
            </Link>
            <Link
              to="/ngo-signup"
              className="w-full sm:w-auto flex items-center justify-center gap-2 rounded-xl border border-border bg-card px-8 py-4 text-base font-semibold text-card-foreground hover:bg-muted transition-all"
            >
              <Building2 className="h-5 w-5 text-primary" />
              Register Organization
            </Link>
          </div>
        </section>

        {/* Key Features */}
        <section className="px-6 py-16 bg-muted/30 border-y border-border">
          <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="rounded-2xl border border-border bg-card p-6 shadow-sm space-y-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-100 text-blue-600">
                <Sparkles className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-bold">AI Event Recommendations</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Discover NGO events tailored directly to your specific skills, location, availability, and social causes.
              </p>
            </div>

            <div className="rounded-2xl border border-border bg-card p-6 shadow-sm space-y-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-green-100 text-green-600">
                <Users className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-bold">Smart Volunteer Matching</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                NGOs receive ranked volunteer recommendations based on required campaign skills, match scores, and impact history.
              </p>
            </div>

            <div className="rounded-2xl border border-border bg-card p-6 shadow-sm space-y-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-purple-100 text-purple-600">
                <HeartHandshake className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-bold">AI Tender Generator</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Effortlessly generate formal Request for Quotation (RFQ) procurement documents for NGO event budgets and supplies.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border px-8 py-6 text-center text-sm text-muted-foreground bg-card">
        <p>© {new Date().getFullYear()} CivicEngage. All rights reserved. Connecting communities for social impact.</p>
      </footer>
    </div>
  );
}
