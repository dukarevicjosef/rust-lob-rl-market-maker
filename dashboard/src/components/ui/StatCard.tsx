import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import type { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string;
  change?: string;
  icon?: LucideIcon;
  className?: string;
}

export default function StatCard({ label, value, change, icon: Icon, className }: StatCardProps) {
  const isPositive = change?.startsWith("+");
  const isNegative = change?.startsWith("-");

  return (
    <Card className={cn("bg-zinc-900 border-zinc-800", className)}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <p className="text-xs text-zinc-500 uppercase tracking-wider">{label}</p>
            <p className="text-xl font-mono font-semibold text-zinc-100">{value}</p>
            {change && (
              <p
                className={cn(
                  "text-xs font-mono",
                  isPositive && "text-green-400",
                  isNegative && "text-red-400",
                  !isPositive && !isNegative && "text-zinc-400",
                )}
              >
                {change}
              </p>
            )}
          </div>
          {Icon && <Icon className="w-4 h-4 text-zinc-600 mt-0.5" />}
        </div>
      </CardContent>
    </Card>
  );
}
