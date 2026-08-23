export default function Badge({ variant = "default", children }) {
  const variants = {
    success: "bg-success/10 text-success border-success/20",
    warning: "bg-accent/10 text-accent border-accent/20",
    error: "bg-destructive/10 text-destructive border-destructive/20",
    default: "bg-muted text-muted-foreground border-border",
  };

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${variants[variant]}`}
    >
      {children}
    </span>
  );
}
