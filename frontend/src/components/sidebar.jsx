import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  Search,
  ListTodo,
  History,
  TrendingUp,
  Award,
  Bell,
  User,
  LogOut,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";

const menuItems = [
  { path: "/", label: "Dashboard", icon: LayoutDashboard },
  { path: "/opportunities", label: "Find Opportunities", icon: Search },
  { path: "/tasks", label: "My Tasks", icon: ListTodo },
  { path: "/history", label: "History", icon: History },
  { path: "/impact-score", label: "Impact Score", icon: TrendingUp },
  { path: "/badges", label: "Badges", icon: Award },
  { path: "/notifications", label: "Notifications", icon: Bell },
  { path: "/profile", label: "Profile", icon: User },
];

export default function Sidebar() {
  const location = useLocation();
  const { logout } = useAuth();

  return (
    <aside className="fixed left-0 top-0 h-screen w-60 border-r border-border bg-card p-4">
      <div className="mb-8">
        <h2 className="px-4 text-xl font-semibold text-primary">CivicEngage</h2>
      </div>
      <nav className="space-y-1">
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 rounded-lg px-4 py-3 transition-all ${
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-foreground hover:bg-muted"
              }`}
            >
              <Icon className="h-5 w-5" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="absolute bottom-4 left-4 right-4">
        <button
          onClick={logout}
          className="flex w-full items-center gap-3 rounded-lg px-4 py-3 transition-all text-destructive hover:bg-destructive/10"
        >
          <LogOut className="h-5 w-5" />
          <span>Logout</span>
        </button>
      </div>
    </aside>
  );
}
