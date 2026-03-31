import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function LivePage() {
  return (
    <div className="grid grid-cols-3 grid-rows-2 gap-4 h-[calc(100vh-theme(spacing.28))]">
      {/* LOB Depth — left column, full height */}
      <Card className="col-span-1 row-span-2 bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-zinc-400">LOB Depth</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
          Order book depth chart
        </CardContent>
      </Card>

      {/* Price Chart — top right (2 cols) */}
      <Card className="col-span-2 row-span-1 bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-zinc-400">Price Chart</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
          Mid-price + agent quotes
        </CardContent>
      </Card>

      {/* Trade Feed — bottom middle */}
      <Card className="col-span-1 row-span-1 bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-zinc-400">Trade Feed</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
          Recent fills
        </CardContent>
      </Card>

      {/* Stats Panel — bottom right */}
      <Card className="col-span-1 row-span-1 bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-zinc-400">Stats Panel</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
          PnL, inventory, Sharpe
        </CardContent>
      </Card>
    </div>
  );
}
