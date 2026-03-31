import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function ArenaPage() {
  return (
    <div className="flex flex-col gap-4 h-[calc(100vh-theme(spacing.28))]">
      {/* Strategy selector */}
      <Tabs defaultValue="sac">
        <TabsList className="bg-zinc-900 border border-zinc-800">
          <TabsTrigger value="sac">SAC</TabsTrigger>
          <TabsTrigger value="optimized_as">Optimized AS</TabsTrigger>
          <TabsTrigger value="static_as">Static AS</TabsTrigger>
          <TabsTrigger value="naive">Naive</TabsTrigger>
        </TabsList>
      </Tabs>

      {/* PnL Chart */}
      <Card className="flex-1 bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-zinc-400">Cumulative PnL</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
          Cumulative PnL chart
        </CardContent>
      </Card>

      {/* Bottom metrics */}
      <div className="grid grid-cols-3 gap-4 shrink-0" style={{ minHeight: "180px" }}>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-400">Sharpe Ratio</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
            Sharpe comparison
          </CardContent>
        </Card>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-400">PnL Decomposition</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
            Spread vs. inventory PnL
          </CardContent>
        </Card>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-400">Distribution</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center h-[calc(100%-3.5rem)] text-zinc-600 text-sm">
            Episode PnL histogram
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
