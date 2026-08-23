export default function Tag({ children }) {
  return (
    <span className="inline-flex items-center rounded-full bg-secondary/10 px-3 py-1 text-xs text-secondary border border-secondary/20">
      {children}
    </span>
  );
}
