import type { Metadata } from "next";
import "./globals.css";
import TopBar from "@/components/terminal/TopBar";
import FunctionKeyBar from "@/components/terminal/FunctionKeyBar";
import StatusBar from "@/components/terminal/StatusBar";
import CommandLine from "@/components/terminal/CommandLine";
import { TooltipProvider } from "@/components/ui/tooltip";

export const metadata: Metadata = {
  title: "QUANTFLOW",
  description: "High-frequency LOB simulation & RL market-making terminal",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full bg-[#0a0a0a] text-[#cccccc] overflow-hidden">
        <TooltipProvider>
          <div className="flex flex-col h-full">
            <TopBar />
            <FunctionKeyBar />
            <main className="flex-1 overflow-hidden">{children}</main>
            <CommandLine />
            <StatusBar />
          </div>
        </TooltipProvider>
      </body>
    </html>
  );
}
