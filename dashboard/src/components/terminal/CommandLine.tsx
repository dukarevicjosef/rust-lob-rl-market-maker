"use client";

import { useState, useRef, useEffect } from "react";

const COMMANDS: Record<string, string> = {
  "SIM START":  "Starting simulation...",
  "SIM STOP":   "Stopping simulation...",
  "EVAL RUN":   "Launching evaluation suite...",
  "BENCH LOB":  "Running LOB benchmarks...",
  "HELP":       "Commands: SIM START, SIM STOP, EVAL RUN, BENCH LOB, CLEAR",
  "CLEAR":      "",
};

const SUGGESTIONS = Object.keys(COMMANDS);

export default function CommandLine() {
  const [input,    setInput]    = useState("");
  const [history,  setHistory]  = useState<string[]>([]);
  const [output,   setOutput]   = useState<string[]>([]);
  const [histIdx,  setHistIdx]  = useState(-1);
  const [suggest,  setSuggest]  = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Partial-match suggestion
    if (input.trim() === "") { setSuggest(null); return; }
    const up = input.toUpperCase();
    const match = SUGGESTIONS.find((s) => s.startsWith(up) && s !== up);
    setSuggest(match ? match.slice(input.length) : null);
  }, [input]);

  const execute = (raw: string) => {
    const cmd = raw.trim().toUpperCase();
    const result = COMMANDS[cmd] ?? `Unknown command: ${raw}`;
    const newHistory = [...history, raw];
    setHistory(newHistory);
    setHistIdx(-1);
    setInput("");
    if (cmd === "CLEAR") { setOutput([]); return; }
    setOutput((prev) => [...prev.slice(-20), `> ${raw}`, result]);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") { execute(input); }
    else if (e.key === "Tab" && suggest) {
      e.preventDefault();
      setInput(input + suggest);
    }
    else if (e.key === "ArrowUp") {
      const idx = Math.min(histIdx + 1, history.length - 1);
      setHistIdx(idx);
      setInput(history[history.length - 1 - idx] ?? "");
    }
    else if (e.key === "ArrowDown") {
      const idx = Math.max(histIdx - 1, -1);
      setHistIdx(idx);
      setInput(idx === -1 ? "" : history[history.length - 1 - idx] ?? "");
    }
  };

  return (
    <div
      className="border-t border-[#1e1e1e] bg-[#0a0a0a] px-3 py-1 cursor-text"
      onClick={() => inputRef.current?.focus()}
    >
      {output.length > 0 && (
        <div className="mb-1 max-h-16 overflow-y-auto">
          {output.map((line, i) => (
            <div
              key={i}
              className={[
                "text-[0.65rem] font-mono leading-4",
                line.startsWith(">") ? "text-[#cccccc]" : "text-[#666666]",
              ].join(" ")}
            >
              {line}
            </div>
          ))}
        </div>
      )}
      <div className="flex items-center gap-1.5">
        <span className="text-[#ff8c00] text-[0.7rem] font-bold select-none">{">"}</span>
        <div className="relative flex-1 flex items-center">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            className="w-full bg-transparent text-[0.75rem] font-mono text-[#ffffff] outline-none caret-[#ff8c00]"
            spellCheck={false}
            autoComplete="off"
            placeholder="TYPE COMMAND..."
          />
          {suggest && (
            <span className="absolute left-0 top-0 text-[0.75rem] font-mono pointer-events-none">
              <span className="invisible">{input}</span>
              <span className="text-[#444444]">{suggest}</span>
            </span>
          )}
        </div>
        <span className="blink text-[#ff8c00] text-[0.75rem] font-bold select-none">█</span>
      </div>
    </div>
  );
}
