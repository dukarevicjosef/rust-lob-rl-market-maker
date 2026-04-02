"use client";

import { useState, useEffect } from "react";
import { Play, Square, Pause, RotateCcw, Shuffle } from "lucide-react";
import type { SimConfig } from "@/hooks/useSimulation";

const SPEEDS     = [0.5, 1, 2, 5, 10];
const STRATEGIES = [
  { key: "as",  label: "Avellaneda-Stoikov" },
  { key: "sac", label: "SAC Agent"           },
];

interface ReplayFileInfo {
  date:    string;
  path:    string;
  size_mb: number;
  events:  number;
}

interface ControlBarProps {
  isConnected:     boolean;
  isRunning:       boolean;
  isPaused:        boolean;
  eventsProcessed: number;
  elapsedTime:     number;
  replayProgress:  number;
  onStart:         (cfg: SimConfig) => void;
  onStop:          () => void;
  onPause:         () => void;
  onResume:        () => void;
  onSpeedChange:   (s: number) => void;
  onReset:         (seed: number) => void;
}

export default function ControlBar({
  isConnected, isRunning, isPaused,
  eventsProcessed, elapsedTime, replayProgress,
  onStart, onStop, onPause, onResume, onSpeedChange, onReset,
}: ControlBarProps) {
  const [seed,        setSeed]        = useState(42);
  const [speed,       setSpeed]       = useState(1);
  const [strategy,    setStrategy]    = useState("as");
  const [mode,        setMode]        = useState<"simulate" | "replay">("simulate");
  const [replayPath,  setReplayPath]  = useState("");
  const [replayFiles, setReplayFiles] = useState<ReplayFileInfo[]>([]);

  // Fetch available replay files when switching to replay mode
  useEffect(() => {
    if (mode !== "replay") return;
    fetch("http://localhost:8000/api/replay/files")
      .then((r) => r.json())
      .then((files: ReplayFileInfo[]) => {
        setReplayFiles(files);
        if (files.length > 0 && !replayPath) {
          setReplayPath(files[0].path);
        }
      })
      .catch(() => setReplayFiles([]));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const handleStart = () => {
    if (mode === "replay") {
      onStart({ seed, speed, strategy, mode: "replay", replayPath });
    } else {
      onStart({ seed, speed, strategy, mode: "simulate" });
    }
  };

  const handleSpeed = (s: number) => {
    setSpeed(s);
    onSpeedChange(s);
  };

  const handleReset = () => onReset(seed);

  const randomSeed = () => setSeed(Math.floor(Math.random() * 9999));

  const fmtTime = (t: number) => {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  const isReplayRunning = isRunning && replayProgress > 0;

  return (
    <div className="relative flex flex-col shrink-0">
      <div className="flex items-center gap-3 px-3 h-12 bg-[#0a0a0a] border-b border-[#1e1e1e] select-none overflow-x-auto">

        {/* Connection / mode badge */}
        <div className="flex items-center gap-1.5 shrink-0">
          <span
            className={[
              "w-1.5 h-1.5 rounded-full shrink-0",
              isConnected ? "bg-[#00d26a]" : "bg-[#555]",
              isConnected && isRunning ? "animate-pulse" : "",
            ].join(" ")}
          />
          {isReplayRunning && (
            <span className="px-1 py-0 text-[0.55rem] font-bold tracking-widest bg-[#0055cc] text-white uppercase">
              REPLAY
            </span>
          )}
        </div>

        {/* Play / Pause / Stop */}
        <div className="flex gap-1 shrink-0">
          {!isRunning ? (
            <button
              onClick={isPaused ? onResume : handleStart}
              disabled={!isConnected || (mode === "replay" && !replayPath)}
              className="flex items-center gap-1 px-2 h-7 bg-[#ff8c00] text-black text-[0.65rem] font-bold uppercase tracking-wider hover:bg-[#ffb700] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Play className="w-3 h-3" />
              {isPaused ? "RESUME" : "START"}
            </button>
          ) : (
            <button
              onClick={onPause}
              className="flex items-center gap-1 px-2 h-7 border border-[#ff8c00] text-[#ff8c00] text-[0.65rem] font-bold uppercase tracking-wider hover:bg-[#ff8c00]/10 transition-colors"
            >
              <Pause className="w-3 h-3" />
              PAUSE
            </button>
          )}

          <button
            onClick={onStop}
            disabled={!isRunning && !isPaused}
            className="flex items-center gap-1 px-2 h-7 border border-[#333] text-[#666] text-[0.65rem] uppercase tracking-wider hover:border-[#ff3b3b] hover:text-[#ff3b3b] disabled:opacity-30 transition-colors"
          >
            <Square className="w-3 h-3" />
            STOP
          </button>

          <button
            onClick={handleReset}
            disabled={isRunning}
            className="flex items-center gap-1 px-2 h-7 border border-[#333] text-[#555] text-[0.65rem] uppercase tracking-wider hover:border-[#444] hover:text-[#888] disabled:opacity-30 transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            RESET
          </button>
        </div>

        <div className="w-px h-5 bg-[#1e1e1e] shrink-0" />

        {/* Speed selector */}
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="text-[0.6rem] text-[#444] uppercase">SPEED</span>
          <div className="flex">
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => handleSpeed(s)}
                className={[
                  "px-1.5 h-6 text-[0.6rem] font-mono border-r border-[#1e1e1e] last:border-0 transition-colors",
                  speed === s
                    ? "bg-[#ff8c00] text-black font-bold"
                    : "bg-[#111] text-[#555] hover:text-[#aaa]",
                ].join(" ")}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>

        <div className="w-px h-5 bg-[#1e1e1e] shrink-0" />

        {/* Mode toggle */}
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="text-[0.6rem] text-[#444] uppercase">MODE</span>
          <div className="flex">
            {(["simulate", "replay"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                disabled={isRunning}
                className={[
                  "px-2 h-6 text-[0.6rem] font-mono border-r border-[#1e1e1e] last:border-0 uppercase tracking-wider transition-colors disabled:opacity-40",
                  mode === m
                    ? m === "replay"
                      ? "bg-[#0055cc] text-white font-bold"
                      : "bg-[#ff8c00] text-black font-bold"
                    : "bg-[#111] text-[#555] hover:text-[#aaa]",
                ].join(" ")}
              >
                {m === "simulate" ? "SIM" : "REPLAY"}
              </button>
            ))}
          </div>
        </div>

        <div className="w-px h-5 bg-[#1e1e1e] shrink-0" />

        {mode === "replay" ? (
          /* ── Replay file selector ─────────────────────────────────────── */
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="text-[0.6rem] text-[#444] uppercase">FILE</span>
            {replayFiles.length === 0 ? (
              <span className="text-[0.6rem] text-[#555] font-mono">No files found</span>
            ) : (
              <select
                value={replayPath}
                onChange={(e) => setReplayPath(e.target.value)}
                disabled={isRunning}
                className="bg-[#111] border border-[#1e1e1e] text-[#cccccc] text-[0.6rem] px-2 h-6 font-mono appearance-none cursor-pointer hover:border-[#0055cc] outline-none disabled:opacity-40 max-w-[220px]"
              >
                {replayFiles.map((f) => (
                  <option key={f.path} value={f.path}>
                    {f.date} ({f.size_mb} MB, {(f.events / 1000).toFixed(0)}K events)
                  </option>
                ))}
              </select>
            )}
          </div>
        ) : (
          /* ── Strategy selector + seed ─────────────────────────────────── */
          <>
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="text-[0.6rem] text-[#444] uppercase">STRAT</span>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                disabled={isRunning}
                className="bg-[#111] border border-[#1e1e1e] text-[#cccccc] text-[0.6rem] px-2 h-6 font-mono appearance-none cursor-pointer hover:border-[#ff8c00] outline-none disabled:opacity-40"
              >
                {STRATEGIES.map((s) => (
                  <option key={s.key} value={s.key}>{s.label}</option>
                ))}
              </select>
            </div>

            <div className="w-px h-5 bg-[#1e1e1e] shrink-0" />

            <div className="flex items-center gap-1 shrink-0">
              <span className="text-[0.6rem] text-[#444] uppercase">SEED</span>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                disabled={isRunning}
                className="w-16 bg-[#111] border border-[#1e1e1e] text-[#cccccc] text-[0.65rem] px-2 h-6 font-mono outline-none hover:border-[#ff8c00] focus:border-[#ff8c00] disabled:opacity-40"
              />
              <button
                onClick={randomSeed}
                disabled={isRunning}
                className="flex items-center h-6 px-1.5 border border-[#1e1e1e] text-[#444] hover:text-[#ff8c00] hover:border-[#ff8c00] disabled:opacity-30 transition-colors"
                title="Random seed"
              >
                <Shuffle className="w-3 h-3" />
              </button>
            </div>
          </>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Live counters */}
        <div className="flex items-center gap-4 shrink-0 font-mono text-[0.6rem]">
          {mode === "replay" && replayProgress > 0 && (
            <span className="text-[#444]">
              PROG <span className="text-[#0088ff]">{(replayProgress * 100).toFixed(1)}%</span>
            </span>
          )}
          <span className="text-[#444]">
            EVT <span className="text-[#cccccc]">{eventsProcessed.toLocaleString()}</span>
          </span>
          <span className="text-[#444]">
            SIM <span className="text-[#cccccc]">{fmtTime(elapsedTime)}</span>
          </span>
        </div>
      </div>

      {/* Replay progress bar — 2px strip at the bottom of the control bar */}
      {mode === "replay" && replayProgress > 0 && (
        <div className="h-[2px] w-full bg-[#111]">
          <div
            className="h-full bg-[#0055cc] transition-all duration-300"
            style={{ width: `${replayProgress * 100}%` }}
          />
        </div>
      )}
    </div>
  );
}
