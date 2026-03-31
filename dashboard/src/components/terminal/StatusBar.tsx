export default function StatusBar() {
  return (
    <div className="flex items-center h-5 bg-[#111111] border-t border-[#1e1e1e] px-3 shrink-0 select-none">
      {/* Left */}
      <span className="text-[0.6rem] text-[#444444] tracking-wide">
        RUST ENGINE{" "}
        <span className="text-[#666666]">v0.1.0</span>
        {"  "}|{"  "}
        PyO3 BRIDGE{" "}
        <span className="text-[#00d26a]">OK</span>
        {"  "}|{"  "}
        ARROW IPC
      </span>

      {/* Middle */}
      <span className="flex-1 text-center text-[0.6rem] text-[#444444]">
        LOB:{" "}
        <span className="text-[#666666]">778ns</span>
        {"  "}
        SNAP:{" "}
        <span className="text-[#666666]">595ns</span>
        {"  "}
        HAWKES:{" "}
        <span className="text-[#666666]">12D</span>
      </span>

      {/* Right */}
      <span className="text-[0.6rem] text-[#ff8c00] font-bold tracking-wide glow-orange">
        20.3M OPS/SEC
      </span>
    </div>
  );
}
