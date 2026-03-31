import { cn } from "@/lib/utils";

export interface Column<T> {
  key: keyof T;
  header: string;
  align?: "left" | "right";
  render?: (val: T[keyof T], row: T) => React.ReactNode;
  colorize?: boolean;
}

interface BloombergTableProps<T extends Record<string, unknown>> {
  columns: Column<T>[];
  rows: T[];
  className?: string;
  rowKey?: keyof T;
}

export default function BloombergTable<T extends Record<string, unknown>>({
  columns,
  rows,
  className,
  rowKey,
}: BloombergTableProps<T>) {
  return (
    <div className={cn("overflow-auto w-full", className)}>
      <table className="w-full bb-table border-collapse">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                style={{ textAlign: col.align ?? "right" }}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={rowKey ? String(row[rowKey]) : ri}>
              {columns.map((col) => {
                const raw = row[col.key];
                const num = typeof raw === "number" ? raw : parseFloat(String(raw));
                const isNum = !isNaN(num);
                const color =
                  col.colorize && isNum
                    ? num > 0 ? "#00d26a" : num < 0 ? "#ff3b3b" : "#cccccc"
                    : undefined;

                return (
                  <td key={String(col.key)} style={{ textAlign: col.align ?? "right", color }}>
                    {col.render ? col.render(raw, row) : String(raw ?? "")}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
