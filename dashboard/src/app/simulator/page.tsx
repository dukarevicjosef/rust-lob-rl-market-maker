import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function SimulatorPage() {
  return (
    <div className="flex gap-4 h-[calc(100vh-theme(spacing.28))]">
      {/* Parameter controls */}
      <Card className="w-72 shrink-0 bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-zinc-400">Parameters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6 pt-2 text-zinc-600 text-sm">
          Hawkes intensity, tick size, episode length sliders
        </CardContent>
      </Card>

      {/* Stylized facts 2×2 */}
      <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-4">
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-400">Return Distribution</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
            Histogram + normal overlay
          </CardContent>
        </Card>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-400">Autocorrelation</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
            ACF of abs returns
          </CardContent>
        </Card>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-400">Spread Distribution</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
            Bid-ask spread histogram
          </CardContent>
        </Card>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-400">Order Arrivals</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
            Hawkes event intensity
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
