import React from "react";
import { Outlet, Link, useLocation, useNavigate } from "react-router-dom";
import { LayoutDashboard, User, History, Award, Sparkles, LogOut, Heart } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function DashboardLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const navItems = [
    { label: "Activities & Events", path: "/app", icon: LayoutDashboard },
    { label: "My Profile", path: "/app/profile", icon: User },
    { label: "Activity History", path: "/app/history", icon: History },
    { label: "Impact Score", path: "/app/impact", icon: Heart },
    { label: "Badges & Badges", path: "/app/badges", icon: Award },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Top Navbar */}
      <header className="border-b border-border bg-card px-6 py-3.5 flex items-center justify-between sticky top-0 z-40 shadow-sm">
        <div className="flex items-center gap-3">
          <Link to="/app" className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-primary-foreground font-bold text-lg">
              C
            </div>
            <span className="text-xl font-bold text-primary">CivicEngage</span>
          </Link>
          <span className="hidden sm:inline-block rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-semibold text-primary border border-primary/20">
            Volunteer Dashboard
          </span>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm font-medium">
            <img
              src={user?.avatar || "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150"}
              alt={user?.name || "User"}
              className="h-8 w-8 rounded-full object-cover border border-border"
            />
            <span className="hidden md:inline font-semibold">{user?.name || "Volunteer"}</span>
          </div>

          <button
            onClick={handleLogout}
            className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs font-medium hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
          >
            <LogOut className="h-3.5 w-3.5" />
            Logout
          </button>
        </div>
      </header>

      {/* Main Container with Sidebar */}
      <div className="flex-1 flex max-w-7xl w-full mx-auto p-6 gap-6">
        {/* Sidebar */}
        <aside className="w-64 shrink-0 hidden md:block space-y-2">
          <div className="rounded-2xl border border-border bg-card p-3 space-y-1 shadow-sm sticky top-24">
            <div className="px-3 py-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">
              Volunteer Menu
            </div>
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    active
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/70"
                  }`}
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  {item.label}
                </Link>
              );
            })}
          </div>
        </aside>

        {/* Dynamic Page Content */}
        <main className="flex-1 min-w-0">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
