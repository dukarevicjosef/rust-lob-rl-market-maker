"use client";

import { cn } from "@/lib/utils";

interface ConnectionBadgeProps {
  connected?: boolean;
  collapsed?: boolean;
}

export default function ConnectionBadge({
  connected = false,
  collapsed = false,
}: ConnectionBadgeProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="relative flex h-2 w-2 shrink-0">
        {connected && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
        )}
        <span
          className={cn(
            "relative inline-flex rounded-full h-2 w-2",
            connected ? "bg-green-500" : "bg-zinc-500",
          )}
        />
      </span>
      {!collapsed && (
        <span className={cn("text-xs", connected ? "text-green-400" : "text-zinc-500")}>
          {connected ? "Connected" : "Disconnected"}
        </span>
      )}
    </div>
  );
}
