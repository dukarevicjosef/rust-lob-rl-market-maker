import { cn } from "@/lib/utils";

interface DataCellProps {
  label: string;
  value: string | number;
  unit?: string;
  precision?: number;
  colorize?: boolean;   // green if positive, red if negative
  className?: string;
}

export default function DataCell({
  label,
  value,
  unit,
  precision = 2,
  colorize = false,
  className,
}: DataCellProps) {
  const num    = typeof value === "number" ? value : parseFloat(String(value));
  const isNum  = !isNaN(num);
  const pos    = colorize && isNum && num > 0;
  const neg    = colorize && isNum && num < 0;
  const display =
    isNum && typeof value === "number"
      ? (num >= 0 && colorize ? "+" : "") + num.toFixed(precision)
      : String(value);

  return (
    <div className={cn("flex flex-col gap-0.5 p-1.5", className)}>
      <span className="text-[0.6rem] uppercase tracking-widest text-[#666666]">{label}</span>
      <span
        className={cn(
          "text-base font-bold font-mono leading-none",
          pos && "text-[#00d26a] glow-green",
          neg && "text-[#ff3b3b] glow-red",
          !pos && !neg && "text-[#ffffff]",
        )}
      >
        {display}
        {unit && <span className="text-[0.65rem] font-normal text-[#666666] ml-0.5">{unit}</span>}
      </span>
    </div>
  );
}
