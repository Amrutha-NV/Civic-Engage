import { Search, Bell } from "lucide-react";
import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-10 flex h-[70px] items-center justify-between border-b border-border bg-card px-6">
      <div className="flex-1 max-w-xl">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search campaigns, NGOs..."
            className="w-full rounded-lg border border-border bg-background px-10 py-2.5 outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
          />
        </div>
      </div>
      <div className="flex items-center gap-4">
        <Link
          to="/notifications"
          className="relative rounded-lg p-2 hover:bg-muted"
        >
          <Bell className="h-5 w-5" />
          <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-destructive"></span>
        </Link>
        <Link to="/profile" className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-full bg-primary flex items-center justify-center text-primary-foreground hover:opacity-90 transition-opacity">
            A
          </div>
        </Link>
      </div>
    </header>
  );
}
