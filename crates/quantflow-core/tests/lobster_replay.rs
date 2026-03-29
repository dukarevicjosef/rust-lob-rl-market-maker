//! Integration tests: load LOBSTER sample fixtures, replay through the LOB,
//! and compare the final book snapshot against the expected state.

use std::path::Path;

use quantflow_core::data::lobster::{load_snapshots, replay_messages, LobsterReplayer};
use quantflow_core::orderbook::{Price, Quantity};

// ── Message replay ────────────────────────────────────────────────────────────

/// Trace through sample_message.csv and verify the exact book state after each
/// event type has been exercised at least once.
///
/// Fixture event sequence (prices in PRICE_SCALE = 10^8 ticks):
///   1. NewLimit bid  id=1, 100@99.90
///   2. NewLimit ask  id=2,  50@100.10
///   3. NewLimit bid  id=3, 200@99.80
///   4. NewLimit ask  id=4,  80@100.20
///   5. Execution ask id=2,  50 → fully consumed
///   6. PartialCancel bid id=3, −100 → 100 remain
///   7. FullCancel    bid id=1
///   8. NewLimit ask  id=5, 150@100.05
///   9. Execution ask id=5,  50 → 100 remain
///  10. NewLimit bid  id=6,  75@99.75
///
/// Expected final state:
///   bids: [(99.80, 100), (99.75, 75)]
///   asks: [(100.05, 100), (100.20, 80)]
#[test]
fn replay_sample_messages_final_state() {
    let path = Path::new("tests/fixtures/sample_message.csv");
    // Fixture prices are already in PRICE_SCALE = 10^8 ticks, so src_scale = 10^8.
    let mut replayer = LobsterReplayer::new(100_000_000);

    for result in replay_messages(path).unwrap() {
        replayer.apply(&result.unwrap());
    }

    let snap = replayer.book.snapshot(5);

    assert_eq!(snap.bids.len(), 2, "expected two resting bid levels");
    assert_eq!(snap.asks.len(), 2, "expected two resting ask levels");

    // Best bid: id=3 partial-cancelled from 200 → 100 remaining at 99.80
    assert_eq!(snap.bids[0].0, Price::from_ticks(9_980_000_000));
    assert_eq!(snap.bids[0].1, Quantity(100));

    // Second bid: id=6 newly inserted at 99.75
    assert_eq!(snap.bids[1].0, Price::from_ticks(9_975_000_000));
    assert_eq!(snap.bids[1].1, Quantity(75));

    // Best ask: id=5 partially executed from 150 → 100 at 100.05
    assert_eq!(snap.asks[0].0, Price::from_ticks(10_005_000_000));
    assert_eq!(snap.asks[0].1, Quantity(100));

    // Second ask: id=4 untouched at 100.20
    assert_eq!(snap.asks[1].0, Price::from_ticks(10_020_000_000));
    assert_eq!(snap.asks[1].1, Quantity(80));
}

#[test]
fn replay_message_count_matches_fixture_lines() {
    let path = Path::new("tests/fixtures/sample_message.csv");
    let count = replay_messages(path).unwrap().count();
    assert_eq!(count, 10);
}

#[test]
fn replay_all_messages_parse_without_error() {
    let path = Path::new("tests/fixtures/sample_message.csv");
    for result in replay_messages(path).unwrap() {
        result.expect("every fixture message must parse cleanly");
    }
}

// ── Best bid / ask after partial events ──────────────────────────────────────

#[test]
fn replay_spread_correct_after_execution_events() {
    let path = Path::new("tests/fixtures/sample_message.csv");
    let mut replayer = LobsterReplayer::new(100_000_000);
    for result in replay_messages(path).unwrap() {
        replayer.apply(&result.unwrap());
    }

    let best_bid = replayer.book.best_bid().expect("book must have a bid");
    let best_ask = replayer.book.best_ask().expect("book must have an ask");

    // spread = 100.05 − 99.80 = 0.25
    let spread = replayer.book.spread().unwrap();
    assert_eq!(spread, best_ask.0 - best_bid.0);
    assert_eq!(spread, Price::from_f64(0.25));
}

// ── Snapshot file parsing ─────────────────────────────────────────────────────

#[test]
fn load_snapshots_row_count() {
    let path = Path::new("tests/fixtures/sample_orderbook.csv");
    let count = load_snapshots(path).unwrap().count();
    assert_eq!(count, 2);
}

#[test]
fn load_snapshots_level_count() {
    let path = Path::new("tests/fixtures/sample_orderbook.csv");
    for result in load_snapshots(path).unwrap() {
        let snap = result.unwrap();
        assert_eq!(snap.depth(), 3, "fixture has L=3 levels");
    }
}

#[test]
fn load_snapshots_first_row_values() {
    let path = Path::new("tests/fixtures/sample_orderbook.csv");
    let snap = load_snapshots(path).unwrap().next().unwrap().unwrap();

    let l0 = &snap.levels[0];
    assert_eq!(l0.ask_price, 10_010_000_000);
    assert_eq!(l0.ask_size, 50);
    assert_eq!(l0.bid_price, 9_990_000_000);
    assert_eq!(l0.bid_size, 100);

    let l2 = &snap.levels[2];
    assert_eq!(l2.ask_price, 10_030_000_000);
    assert_eq!(l2.bid_price, 9_970_000_000);
}

#[test]
fn load_snapshots_parses_without_error() {
    let path = Path::new("tests/fixtures/sample_orderbook.csv");
    for result in load_snapshots(path).unwrap() {
        result.expect("all fixture snapshot rows must parse cleanly");
    }
}
