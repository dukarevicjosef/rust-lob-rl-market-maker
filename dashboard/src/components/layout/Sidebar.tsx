"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, BarChart3, FlaskConical, Cpu, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";
import ConnectionBadge from "@/components/ui/ConnectionBadge";

const NAV_ITEMS = [
  { href: "/live",      label: "Live Trading",    icon: Activity },
  { href: "/arena",     label: "Strategy Arena",  icon: BarChart3 },
  { href: "/simulator", label: "Simulator Lab",   icon: FlaskConical },
  { href: "/metrics",   label: "System Metrics",  icon: Cpu },
];

interface SidebarProps {
  collapsed?: boolean;
}

export default function Sidebar({ collapsed = false }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "flex flex-col h-screen bg-zinc-900 border-r border-zinc-800 transition-all duration-200",
        collapsed ? "w-14" : "w-60",
      )}
    >
      {/* Branding */}
      <div className="flex items-center gap-2.5 px-4 h-14 border-b border-zinc-800 shrink-0">
        <TrendingUp className="w-5 h-5 text-blue-500 shrink-0" />
        {!collapsed && (
          <span className="font-semibold tracking-tight text-zinc-100">QuantFlow</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 space-y-0.5 px-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 px-2 py-2 rounded-md text-sm transition-colors",
                active
                  ? "bg-zinc-800 text-zinc-100 border-l-2 border-blue-500 pl-[6px]"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60",
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {!collapsed && <span>{label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Connection status */}
      <div className="px-4 py-3 border-t border-zinc-800 shrink-0">
        <ConnectionBadge collapsed={collapsed} />
      </div>
    </aside>
  );
}
