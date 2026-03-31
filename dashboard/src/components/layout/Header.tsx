"use client";

import { usePathname } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { ChevronRight } from "lucide-react";

const PAGE_LABELS: Record<string, string> = {
  "/live":      "Live Trading",
  "/arena":     "Strategy Arena",
  "/simulator": "Simulator Lab",
  "/metrics":   "System Metrics",
};

interface HeaderProps {
  isRunning?: boolean;
}

export default function Header({ isRunning = false }: HeaderProps) {
  const pathname = usePathname();
  const label = PAGE_LABELS[pathname] ?? pathname;

  return (
    <header className="flex items-center justify-between h-14 px-6 border-b border-zinc-800 bg-zinc-950 shrink-0">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5 text-sm text-zinc-400">
        <span className="text-zinc-500">QuantFlow</span>
        <ChevronRight className="w-3.5 h-3.5 text-zinc-600" />
        <span className="text-zinc-200">{label}</span>
      </div>

      {/* Right side controls */}
      <div className="flex items-center gap-3">
        <Badge
          variant="outline"
          className={
            isRunning
              ? "border-green-700 text-green-400 bg-green-950/40"
              : "border-zinc-700 text-zinc-400 bg-zinc-800/40"
          }
        >
          {isRunning ? "Running" : "Stopped"}
        </Badge>
      </div>
    </header>
  );
}
