use quantflow_core::exchange::{
    risk_manager::{RiskConfig, RiskManager, RiskViolation},
    OrderSide,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// A minimal config so tests run fast and bounds are easy to hit.
fn cfg() -> RiskConfig {
    RiskConfig {
        max_position:          0.1,
        max_daily_loss:        100.0,
        max_drawdown:          200.0,
        max_orders_per_second: 5,
        max_order_size:        0.05,
        max_open_orders:       10,
        max_notional:          50_000.0,
    }
}

fn rm() -> RiskManager {
    RiskManager::new(cfg())
}

fn is_kind(v: &RiskViolation, tag: &str) -> bool {
    match (v, tag) {
        (RiskViolation::OrderSizeExceeded     { .. }, "size")       => true,
        (RiskViolation::PositionLimitExceeded { .. }, "position")   => true,
        (RiskViolation::DailyLossExceeded     { .. }, "daily_loss") => true,
        (RiskViolation::DrawdownExceeded      { .. }, "drawdown")   => true,
        (RiskViolation::KillSwitchActive,             "kill")       => true,
        (RiskViolation::OrderRateExceeded     { .. }, "rate")       => true,
        _ => false,
    }
}

// ── 1. Happy path ─────────────────────────────────────────────────────────────

#[test]
fn order_within_all_limits_is_ok() {
    let mut r = rm();
    // qty=0.01 << max_order_size=0.05, position stays at 0.01 << 0.1, price=50k
    r.check_order(OrderSide::Buy, 0.01, 50_000.0).expect("should pass all checks");
}

// ── 2. Order size ─────────────────────────────────────────────────────────────

#[test]
fn order_over_max_size_is_rejected() {
    let mut r = rm();
    let res = r.check_order(OrderSide::Buy, 0.06, 50_000.0); // 0.06 > 0.05
    let err = res.unwrap_err();
    assert!(is_kind(&err, "size"), "expected OrderSizeExceeded, got: {err}");
}

// ── 3. Position limits ────────────────────────────────────────────────────────

#[test]
fn position_plus_order_exceeds_limit_is_rejected() {
    let mut r = rm();
    // Set position to 0.09 (just below 0.1 limit)
    r.on_fill(OrderSide::Buy, 0.09, 50_000.0);
    // Adding 0.02 would bring it to 0.11 > 0.10
    let err = r.check_order(OrderSide::Buy, 0.02, 50_000.0).unwrap_err();
    assert!(is_kind(&err, "position"), "expected PositionLimitExceeded, got: {err}");
}

#[test]
fn position_exactly_at_limit_is_ok() {
    let mut r = rm();
    // Position 0.09 + order 0.01 = exactly 0.10 = limit → should pass
    r.on_fill(OrderSide::Buy, 0.09, 50_000.0);
    r.check_order(OrderSide::Buy, 0.01, 50_000.0).expect("exact limit should pass");
}

#[test]
fn buy_at_max_long_rejected_sell_accepted() {
    let mut r = rm();
    // Fill to max long position
    r.on_fill(OrderSide::Buy, 0.1, 50_000.0);

    // Another buy would take position to 0.1 + ε > 0.1
    let err = r.check_order(OrderSide::Buy, 0.001, 50_000.0).unwrap_err();
    assert!(is_kind(&err, "position"));

    // But a sell reduces position — should pass (projected = 0.1 − 0.001 = 0.099)
    r.check_order(OrderSide::Sell, 0.001, 50_000.0).expect("reducing sell should pass");
}

// ── 4. Daily loss ─────────────────────────────────────────────────────────────

#[test]
fn daily_loss_exceeds_limit_activates_kill_switch() {
    let mut r = rm();
    // Push PnL to −101 (> max_daily_loss=100)
    r.update_pnl(-101.0, 0.0);
    assert!(r.is_killed(), "kill switch should have fired");

    let err = r.check_order(OrderSide::Buy, 0.01, 50_000.0).unwrap_err();
    assert!(is_kind(&err, "kill"), "expected KillSwitchActive, got: {err}");
}

// ── 5. Drawdown ───────────────────────────────────────────────────────────────

#[test]
fn drawdown_exceeds_limit_activates_kill_switch() {
    let mut r = rm();
    // Peak at +150
    r.update_pnl(150.0, 0.0);
    // Drawdown to −60 → drawdown = 150 − (−60) = 210 > max_drawdown=200
    r.update_pnl(-60.0, 0.0);
    assert!(r.is_killed(), "kill switch should have fired on drawdown");

    let err = r.check_order(OrderSide::Buy, 0.01, 50_000.0).unwrap_err();
    assert!(is_kind(&err, "kill"));
}

// ── 6. Kill switch ────────────────────────────────────────────────────────────

#[test]
fn kill_switch_blocks_all_orders() {
    let mut r = rm();
    r.manual_kill();
    let err = r.check_order(OrderSide::Buy, 0.01, 50_000.0).unwrap_err();
    assert!(is_kind(&err, "kill"), "expected KillSwitchActive");
}

#[test]
fn kill_switch_reset_allows_orders() {
    let mut r = rm();
    r.manual_kill();
    r.reset_kill_switch();
    r.check_order(OrderSide::Buy, 0.01, 50_000.0).expect("should pass after reset");
}

// ── 7. Rate limiting ──────────────────────────────────────────────────────────

#[test]
fn rate_limit_blocks_sixth_order_within_one_second() {
    let mut r = rm();
    // Place 5 orders rapidly (well within 1 second)
    for _ in 0..5 {
        r.check_order(OrderSide::Buy, 0.001, 50_000.0).expect("first 5 should pass");
    }
    // 6th order in the same second should be blocked
    let err = r.check_order(OrderSide::Buy, 0.001, 50_000.0).unwrap_err();
    assert!(is_kind(&err, "rate"), "expected OrderRateExceeded, got: {err}");
}

#[test]
fn rate_limit_resets_after_one_second() {
    let mut r = rm();
    for _ in 0..5 {
        r.check_order(OrderSide::Buy, 0.001, 50_000.0).unwrap();
    }
    // Wait for the sliding window to expire
    std::thread::sleep(std::time::Duration::from_millis(1100));
    // 6th order is now in a fresh window
    r.check_order(OrderSide::Buy, 0.001, 50_000.0)
        .expect("should pass after window resets");
}

// ── 8. on_fill ────────────────────────────────────────────────────────────────

#[test]
fn on_fill_updates_position_correctly() {
    let mut r = rm();
    r.on_fill(OrderSide::Buy,  0.05, 50_000.0);
    assert!((r.position() - 0.05).abs() < 1e-9, "position should be +0.05");
    r.on_fill(OrderSide::Sell, 0.02, 50_000.0);
    assert!((r.position() - 0.03).abs() < 1e-9, "position should be +0.03");
    r.on_fill(OrderSide::Sell, 0.03, 50_000.0);
    assert!(r.position().abs() < 1e-9, "position should be flat");
}

// ── 9. reset_daily ────────────────────────────────────────────────────────────

#[test]
fn reset_daily_clears_pnl_and_peak() {
    let mut r = rm();
    r.update_pnl(50.0, 0.0);   // peak=50
    r.update_pnl(-20.0, 0.0);  // daily_pnl=−20, drawdown=70
    r.reset_daily();
    assert_eq!(r.daily_pnl(), 0.0);
    assert_eq!(r.drawdown(), 0.0, "peak should also reset");
}

// ── 10. status() utilization ──────────────────────────────────────────────────

#[test]
fn status_reports_correct_utilization() {
    let mut r = rm();
    r.on_fill(OrderSide::Buy, 0.05, 50_000.0); // position = 0.05 = 50% of max
    r.update_pnl(-50.0, 0.0);                  // loss = 50 = 50% of max_daily_loss

    let s = r.status();
    assert!((s.position_utilization - 0.5).abs() < 1e-6,
            "position_utilization should be 0.5, got {}", s.position_utilization);
    assert!((s.loss_utilization - 0.5).abs() < 1e-6,
            "loss_utilization should be 0.5, got {}", s.loss_utilization);
}
