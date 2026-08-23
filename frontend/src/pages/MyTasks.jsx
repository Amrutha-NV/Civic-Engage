import { useState, useEffect } from "react";
import { Calendar, Building2, Loader2, CheckCircle } from "lucide-react";
import { toast } from "react-toastify";
import api from "../lib/api";
import Badge from "../components/badge";

export default function MyTasks() {
  const [activeTab, setActiveTab] = useState("ongoing");
  const [loading, setLoading] = useState(true);
  const [tasks, setTasks] = useState({
    pending: [],
    ongoing: [],
    completed: [],
  });

  const fetchTasks = async () => {
    try {
      setLoading(true);
      const res = await api.get("/events/tasks");
      setTasks({
        pending: res.data.pending || [],
        ongoing: res.data.ongoing || [],
        completed: res.data.completed || [],
      });
    } catch (error) {
      toast.error("Failed to fetch tasks");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTasks();
  }, []);

  const handleMarkCompleted = async (taskId) => {
    try {
      await api.put(`/events/task/${taskId}`, { status: "completed" });
      toast.success("Task marked as completed!");
      fetchTasks();
    } catch (error) {
      toast.error("Failed to update task status");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold">My Tasks</h1>
        <p className="mt-2 text-muted-foreground">
          Manage and track your volunteer commitments
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-4 border-b border-border">
        {["pending", "ongoing", "completed"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`pb-3 px-2 capitalize transition-colors ${
              activeTab === tab
                ? "border-b-2 border-primary text-primary font-medium"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab} ({tasks[tab].length})
          </button>
        ))}
      </div>

      {/* Task List */}
      {loading ? (
        <div className="flex justify-center p-20">
          <Loader2 className="animate-spin text-primary w-10 h-10" />
        </div>
      ) : tasks[activeTab].length === 0 ? (
        <div className="rounded-xl border border-border bg-card p-10 text-center">
          <p className="text-muted-foreground">No tasks found in this section.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {tasks[activeTab].map((task) => (
            <div
              key={task._id}
              className="flex flex-col rounded-xl border border-border bg-card p-6 shadow-sm hover:shadow-lg transition-all"
            >
              <div className="mb-4 flex-1">
                <h3 className="text-lg font-semibold mb-2">{task.title || task.event?.title || "Untitled Task"}</h3>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Building2 className="h-4 w-4" />
                  <span>{task.ngo || task.event?.ngo || task.event?.organizer || "Community"}</span>
                </div>
              </div>
              <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
                <Calendar className="h-4 w-4" />
                <span>Deadline: {task.deadline ? new Date(task.deadline).toLocaleDateString() : "Flexible"}</span>
              </div>
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <Badge
                    variant={
                      task.status === "completed"
                        ? "success"
                        : task.status === "ongoing"
                        ? "warning"
                        : "default"
                    }
                  >
                    {task.status || activeTab}
                  </Badge>
                  {task.status !== "completed" && activeTab !== "completed" && (
                    <button className="text-sm text-primary hover:underline">
                      View Details
                    </button>
                  )}
                </div>
                {activeTab === "ongoing" && (
                  <button
                    onClick={() => handleMarkCompleted(task._id)}
                    className="mt-2 w-full flex items-center justify-center gap-2 rounded-lg bg-success/10 px-4 py-2 text-sm font-medium text-success hover:bg-success/20 transition-colors"
                  >
                    <CheckCircle className="h-4 w-4" /> Mark Completed
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
