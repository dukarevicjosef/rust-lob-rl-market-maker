"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";

const KEYS = [
  { f: "F1", label: "LIVE",   href: "/live" },
  { f: "F2", label: "ARENA",  href: "/arena" },
  { f: "F3", label: "SIMLAB", href: "/simulator" },
  { f: "F4", label: "SYSTEM", href: "/metrics" },
];

export default function FunctionKeyBar() {
  const router   = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const match = KEYS.find((k) => e.key === k.f);
      if (match) { e.preventDefault(); router.push(match.href); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [router]);

  return (
    <div className="flex h-6 bg-[#111111] border-b border-[#1e1e1e] select-none">
      {KEYS.map(({ f, label, href }) => {
        const active = pathname === href || pathname.startsWith(href + "/");
        return (
          <button
            key={f}
            onClick={() => router.push(href)}
            className={[
              "flex items-center gap-1 px-3 h-full text-[0.65rem] tracking-widest cursor-pointer",
              "border-r border-[#1e1e1e] transition-colors",
              active
                ? "bg-[#ff8c00] text-black font-bold"
                : "text-[#666666] hover:bg-[#1e1e1e] hover:text-[#cccccc]",
            ].join(" ")}
          >
            <span className={active ? "text-black/70" : "text-[#444]"}>{`<${f}>`}</span>
            <span>{label}</span>
          </button>
        );
      })}
      <div className="flex-1" />
    </div>
  );
}
