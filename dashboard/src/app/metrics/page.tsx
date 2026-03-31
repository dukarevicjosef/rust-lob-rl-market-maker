import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function MetricsPage() {
  return (
    <div className="flex flex-col gap-4 h-[calc(100vh-theme(spacing.28))]">
      {/* Benchmark table */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-zinc-400">Latency Benchmarks</CardTitle>
        </CardHeader>
        <CardContent className="text-zinc-600 text-sm">
          Criterion benchmark results — insert / cancel / match throughput
        </CardContent>
      </Card>

      {/* Training curves */}
      <Card className="flex-1 bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-zinc-400">Training Curves</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
          Reward, PnL, Sharpe vs. timesteps
        </CardContent>
      </Card>
    </div>
  );
}
