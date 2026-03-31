"use client";

import { useEffect, useRef } from "react";

interface TickerItem {
  label: string;
  value: string;
  color?: string;
}

interface TickerProps {
  items: TickerItem[];
  speed?: number; // px/sec, default 40
}

export default function Ticker({ items, speed = 40 }: TickerProps) {
  const trackRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const track = trackRef.current;
    if (!track) return;
    const width = track.scrollWidth / 2;
    let pos = 0;
    let last = performance.now();
    let raf: number;

    const step = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      pos += speed * dt;
      if (pos >= width) pos = 0;
      track.style.transform = `translateX(-${pos}px)`;
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [speed]);

  const repeated = [...items, ...items]; // seamless loop

  return (
    <div className="overflow-hidden whitespace-nowrap w-full">
      <div ref={trackRef} className="inline-flex gap-6 will-change-transform">
        {repeated.map((item, i) => (
          <span key={i} className="text-[0.65rem] font-mono">
            <span className="text-[#666666]">{item.label} </span>
            <span style={{ color: item.color ?? "#cccccc" }}>{item.value}</span>
            <span className="text-[#333333] mx-3">|</span>
          </span>
        ))}
      </div>
    </div>
  );
}
