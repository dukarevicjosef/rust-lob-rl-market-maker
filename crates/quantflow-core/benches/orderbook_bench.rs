use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput,
};
use std::time::{Duration, Instant};

use quantflow_core::orderbook::{
    Order, OrderBook, OrderId, OrderType, Price, Quantity, Side, Timestamp,
};

// ── Setup helpers ─────────────────────────────────────────────────────────────

#[inline]
fn limit(id: u64, side: Side, price_f64: f64, qty: u64) -> Order {
    Order::new(
        OrderId(id),
        side,
        Price::from_f64(price_f64),
        Quantity(qty),
        Timestamp(id),
        OrderType::Limit,
    )
}

#[inline]
fn market(id: u64, side: Side, qty: u64) -> Order {
    Order::new(
        OrderId(id),
        side,
        Price::from_ticks(0), // price irrelevant for market orders
        Quantity(qty),
        Timestamp(id),
        OrderType::Market,
    )
}

/// Build a book with `n` resting orders split across 10 price levels per side.
/// Bids at [90, 89, ..., 81], asks at [110, 111, ..., 119]. No crossing.
///
/// ID layout per level (bp = bid price, ap = ask price, k = per-level index):
///   bid(id), ask(id+1), bid(id+2), ask(id+3), ...
///
/// Returns the book and the ordered list of all inserted IDs (needed to
/// select valid cancel targets deterministically).
fn seeded_book(n: usize) -> (OrderBook, Vec<OrderId>) {
    let mut book = OrderBook::new();
    let levels = 10usize;
    let per_level = ((n / 2) / levels).max(1);
    let mut ids = Vec::with_capacity(n);
    let mut id = 1u64;

    for i in 0..levels {
        let bp = 90.0 - i as f64;
        let ap = 110.0 + i as f64;
        for _ in 0..per_level {
            book.add_limit_order(limit(id, Side::Bid, bp, 10));
            ids.push(OrderId(id));
            id += 1;
            book.add_limit_order(limit(id, Side::Ask, ap, 10));
            ids.push(OrderId(id));
            id += 1;
        }
    }
    (book, ids)
}

/// Build a book with `levels` price levels per side, one order of `qty` lots
/// each. Best bid = 99.0, best ask = 101.0. Returns the book and the next
/// available order ID (for the caller to assign to incoming orders).
fn levelled_book(levels: usize, qty: u64) -> (OrderBook, u64) {
    let mut book = OrderBook::new();
    let mut id = 1u64;
    for i in 1..=levels {
        book.add_limit_order(limit(id, Side::Bid, 100.0 - i as f64, qty));
        id += 1;
        book.add_limit_order(limit(id, Side::Ask, 100.0 + i as f64, qty));
        id += 1;
    }
    (book, id)
}

// ── bench 1: insert limit order (no match) ───────────────────────────────────
//
// Target: < 500 ns
// Models the hot-path for a passive quote that does not cross the spread.
// Each iteration gets a fresh book clone so prior insertions do not accumulate.

fn bench_insert_limit_order(c: &mut Criterion) {
    c.bench_function("insert_limit_order_no_match", |b| {
        b.iter_batched(
            || {
                let (book, _) = seeded_book(10_000);
                // Bid well below the spread (best ask = 110); no match possible.
                let incoming = limit(99_999_999, Side::Bid, 50.0, 10);
                (book, incoming)
            },
            |(mut book, order)| {
                black_box(book.add_limit_order(order));
            },
            BatchSize::LargeInput,
        )
    });
}

// ── bench 2: cancel order ─────────────────────────────────────────────────────
//
// Target: < 200 ns
// Exercises: O(1) hash-map lookup → O(log L) BTreeMap access →
//            O(Q/2) VecDeque scan to locate the target order.
// We cancel the order at position ~125 in a 500-order price-level queue
// (ids[250] falls in the middle of the first bid level).

fn bench_cancel_order(c: &mut Criterion) {
    c.bench_function("cancel_order_mid_queue", |b| {
        b.iter_batched(
            || {
                let (book, ids) = seeded_book(10_000);
                // ids[250] = OrderId(251): 126th bid in level-0 queue (500 bids deep).
                // Chosen to exercise a realistic mid-queue linear scan.
                let target = ids[250];
                (book, target)
            },
            |(mut book, id)| {
                black_box(book.cancel_order(id));
            },
            BatchSize::LargeInput,
        )
    });
}

// ── bench 3: market order sweeping 5 ask levels ──────────────────────────────
//
// Target: < 1 µs
// 100 ask levels × 10 lots each. Market buy for 50 lots exhausts exactly
// ask levels at 101, 102, 103, 104, 105 and triggers 5 Trade events.

fn bench_market_order_match(c: &mut Criterion) {
    c.bench_function("market_order_5_level_sweep", |b| {
        b.iter_batched(
            || {
                let (book, next_id) = levelled_book(100, 10);
                let mkt = market(next_id, Side::Bid, 50); // 5 levels × 10 lots
                (book, mkt)
            },
            |(mut book, mkt)| {
                black_box(book.add_limit_order(mkt));
            },
            BatchSize::LargeInput,
        )
    });
}

// ── bench 4: depth snapshot ───────────────────────────────────────────────────
//
// Target: < 10 µs
// Read-only operation; book is shared across all iterations.
// Exercises BTreeMap iteration and per-level quantity aggregation.

fn bench_snapshot(c: &mut Criterion) {
    let (book, _) = levelled_book(100, 10);

    c.bench_function("snapshot_10_levels", |b| {
        b.iter(|| black_box(book.snapshot(10)));
    });
}

// ── bench 5: 1 M mixed operations ─────────────────────────────────────────────
//
// 70 % limit insert (non-crossing) | 20 % cancel | 10 % market (1 lot)
// Reports throughput in ops/sec via Criterion's Throughput API.

enum Op {
    Insert(Order),
    Cancel(OrderId),
    Market(Order),
}

/// Deterministically generate `n` operations in a 7:2:1 insert/cancel/market
/// ratio without any external RNG dependency.
///
/// - Insert IDs start at 20_000 (beyond the seeded_book(10_000) range of 1..=10_000).
/// - Cancel IDs cycle through 1..=10_000 (the initial book's order range).
/// - Market IDs start at 10_000_000 to avoid all collisions.
fn generate_ops(n: usize) -> Vec<Op> {
    let mut ops = Vec::with_capacity(n);
    let mut ins_id = 20_000u64;
    let mut mkt_id = 10_000_000u64;

    for i in 0..n {
        match i % 10 {
            0 => {
                // 10 % market buy — 1 lot, matches cheapest available ask.
                ops.push(Op::Market(market(mkt_id, Side::Bid, 1)));
                mkt_id += 1;
            }
            1 | 2 => {
                // 20 % cancel — cycle through the initial book's ID space.
                // Some may be stale (already cancelled); those are O(1) no-ops.
                let cancel_id = (i as u64 % 10_000) + 1;
                ops.push(Op::Cancel(OrderId(cancel_id)));
            }
            _ => {
                // 70 % insert — non-crossing passive quotes.
                // Bids at [50, 51, ..., 69], asks at [130, 131, ..., 149].
                let side = if ins_id % 2 == 0 { Side::Bid } else { Side::Ask };
                let price = if side == Side::Bid {
                    Price::from_f64(70.0 - (ins_id % 20) as f64)
                } else {
                    Price::from_f64(130.0 + (ins_id % 20) as f64)
                };
                ops.push(Op::Insert(Order::new(
                    OrderId(ins_id),
                    side,
                    price,
                    Quantity(10),
                    Timestamp(ins_id),
                    OrderType::Limit,
                )));
                ins_id += 1;
            }
        }
    }
    ops
}

fn bench_throughput(c: &mut Criterion) {
    const N_OPS: usize = 1_000_000;
    let ops = generate_ops(N_OPS);

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Elements(N_OPS as u64));
    // Each iteration processes 1 M ops; keep sample count low to limit wall time.
    group.sample_size(10);

    group.bench_function("1M_mixed_70ins_20can_10mkt", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                // Start each iteration from the same initial state.
                let (mut book, _) = seeded_book(10_000);
                let start = Instant::now();
                for op in &ops {
                    match op {
                        Op::Insert(order) => {
                            // Order is stack-only; clone is a plain struct copy.
                            black_box(book.add_limit_order(order.clone()));
                        }
                        Op::Cancel(id) => {
                            black_box(book.cancel_order(*id));
                        }
                        Op::Market(order) => {
                            black_box(book.add_limit_order(order.clone()));
                        }
                    }
                }
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

// ── Registration ──────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_insert_limit_order,
    bench_cancel_order,
    bench_market_order_match,
    bench_snapshot,
    bench_throughput,
);
criterion_main!(benches);
