import { cn } from "@/lib/utils";

interface PanelProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
  titleClassName?: string;
}

export default function Panel({ title, subtitle, children, className, titleClassName }: PanelProps) {
  return (
    <div className={cn("flex flex-col bg-[#111111] border border-[#1e1e1e]", className)}>
      {/* Title bar */}
      <div className="flex items-center justify-between h-6 px-2 border-b border-[#1e1e1e] shrink-0">
        <span
          className={cn(
            "text-[0.65rem] font-bold uppercase tracking-widest text-[#ff8c00]",
            titleClassName,
          )}
        >
          {title}
        </span>
        {subtitle && (
          <span className="text-[0.6rem] text-[#666666] font-normal">{subtitle}</span>
        )}
      </div>
      <div className="flex-1 overflow-hidden">{children}</div>
    </div>
  );
}
