import Panel from "@/components/terminal/Panel";
import DataCell from "@/components/terminal/DataCell";

export default function LivePage() {
  // Static placeholders — wired up in Prompt 2
  const asks = [
    { price: 99.58, qty: 340 },
    { price: 99.55, qty: 120 },
    { price: 99.53, qty: 80 },
    { price: 99.51, qty: 210 },
    { price: 99.50, qty: 60 },
  ];
  const bids = [
    { price: 99.47, qty: 150 },
    { price: 99.45, qty: 90 },
    { price: 99.43, qty: 280 },
    { price: 99.40, qty: 200 },
    { price: 99.38, qty: 320 },
  ];

  return (
    <div className="grid grid-cols-12 h-full" style={{ gridTemplateRows: "1fr 1fr 1fr 1fr 1fr 1fr" }}>

      {/* ── ORDER BOOK — col 1-3, full height ─────────────────────────────── */}
      <div className="col-span-3 row-span-6 border-r border-[#1e1e1e] flex flex-col">
        <Panel title="ORDER BOOK" subtitle="L5" className="flex-1 border-0">
          <div className="flex flex-col h-full font-mono text-[0.7rem]">

            {/* Ask side */}
            <div className="flex-1 flex flex-col-reverse px-2 pb-1 overflow-hidden">
              {asks.map((l, i) => (
                <div key={i} className="flex justify-between py-0.5 relative">
                  <div
                    className="absolute right-0 top-0 h-full opacity-20 bg-[#ff3b3b]"
                    style={{ width: `${(l.qty / 400) * 100}%` }}
                  />
                  <span className="text-[#ff3b3b] relative z-10">{l.price.toFixed(2)}</span>
                  <span className="text-[#999999] relative z-10">{l.qty}</span>
                </div>
              ))}
            </div>

            {/* Spread indicator */}
            <div className="border-y border-[#1e1e1e] px-2 py-1 flex justify-between">
              <span className="text-[0.6rem] text-[#666666]">SPREAD</span>
              <span className="text-[#ffb700] font-bold">0.06</span>
            </div>

            {/* Bid side */}
            <div className="flex-1 px-2 pt-1 overflow-hidden">
              {bids.map((l, i) => (
                <div key={i} className="flex justify-between py-0.5 relative">
                  <div
                    className="absolute right-0 top-0 h-full opacity-20 bg-[#00d26a]"
                    style={{ width: `${(l.qty / 400) * 100}%` }}
                  />
                  <span className="text-[#00d26a] relative z-10">{l.price.toFixed(2)}</span>
                  <span className="text-[#999999] relative z-10">{l.qty}</span>
                </div>
              ))}
            </div>
          </div>
        </Panel>
      </div>

      {/* ── PRICE CHART — col 4-12, rows 1-4 ────────────────────────────────── */}
      <div className="col-span-9 row-span-4 border-b border-[#1e1e1e] flex flex-col">
        <Panel
          title="MID PRICE"
          subtitle="BID 99.47 / ASK 99.53 | SPREAD 0.06"
          className="flex-1 border-0"
        >
          <div className="flex items-center justify-center h-full text-[#333333] text-[0.65rem] tracking-widest">
            [ LIGHTWEIGHT-CHARTS CANDLESTICK — WIRED IN PROMPT 2 ]
          </div>
        </Panel>
      </div>

      {/* ── AGENT STATE ───────────────────────────────────────────────────────── */}
      <div className="col-span-3 row-span-2 border-r border-[#1e1e1e] border-t border-[#1e1e1e]">
        <Panel title="AGENT STATE" className="border-0 h-full">
          <div className="grid grid-cols-2 p-1">
            <DataCell label="PNL"       value={124.5}  colorize precision={1} />
            <DataCell label="INVENTORY" value={3}      colorize precision={0} />
            <DataCell label="SHARPE"    value={1.91}   colorize precision={2} />
            <DataCell label="FILL RT"   value="30.1"   unit="%" />
          </div>
        </Panel>
      </div>

      {/* ── TRADE FEED ───────────────────────────────────────────────────────── */}
      <div className="col-span-3 row-span-2 border-r border-[#1e1e1e] border-t border-[#1e1e1e]">
        <Panel title="TRADE FEED" className="border-0 h-full">
          <div className="overflow-y-auto h-full px-2 py-1 space-y-0.5">
            {[
              { side: "BUY",  price: 99.47, qty: 10,  ts: "14:23:01.442" },
              { side: "SELL", price: 99.53, qty: 5,   ts: "14:23:01.318" },
              { side: "BUY",  price: 99.47, qty: 20,  ts: "14:23:00.991" },
              { side: "SELL", price: 99.55, qty: 15,  ts: "14:22:59.872" },
              { side: "BUY",  price: 99.45, qty: 30,  ts: "14:22:59.501" },
            ].map((t, i) => (
              <div key={i} className="flex justify-between text-[0.65rem] font-mono">
                <span className="text-[#444444]">{t.ts}</span>
                <span className={t.side === "BUY" ? "text-[#00d26a]" : "text-[#ff3b3b]"}>
                  {t.side}
                </span>
                <span className="text-[#cccccc]">{t.price.toFixed(2)}</span>
                <span className="text-[#666666]">{t.qty}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {/* ── PERFORMANCE ──────────────────────────────────────────────────────── */}
      <div className="col-span-3 row-span-2 border-t border-[#1e1e1e]">
        <Panel title="PERFORMANCE" className="border-0 h-full">
          <div className="grid grid-cols-2 p-1">
            <DataCell label="MAX DD"    value={-221}   colorize precision={0} />
            <DataCell label="WIN RATE"  value="58.2"   unit="%" />
            <DataCell label="AVG PNL"   value={2.48}   colorize precision={2} />
            <DataCell label="VOL EST"   value="0.061" />
          </div>
        </Panel>
      </div>

    </div>
  );
}
