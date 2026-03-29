use super::book::OrderBook;
use super::order::{ExecutionReport, Order, Trade};
use super::types::{OrderType, Price, Quantity, Side};

// ── Entry point ───────────────────────────────────────────────────────────────

/// Route an incoming order to the appropriate matching strategy.
pub fn process_order(book: &mut OrderBook, incoming: Order) -> ExecutionReport {
    match incoming.order_type {
        OrderType::Limit => process_limit(book, incoming),
        OrderType::Market => process_market(book, incoming),
        OrderType::ImmediateOrCancel => process_ioc(book, incoming),
        OrderType::FillOrKill => process_fok(book, incoming),
    }
}

// ── Limit ─────────────────────────────────────────────────────────────────────

/// Match at the limit price or better; rest the residual on the book.
fn process_limit(book: &mut OrderBook, mut incoming: Order) -> ExecutionReport {
    let mut trades = Vec::new();
    let price_limit = incoming.price; // Price is Copy
    sweep(book, &mut incoming, Some(price_limit), &mut trades);

    if incoming.quantity.0 > 0 {
        // Unmatched remainder rests on the book.
        book.insert_resting(incoming.clone());
        ExecutionReport::partial(trades, incoming)
    } else {
        ExecutionReport::filled(trades)
    }
}

// ── Market ────────────────────────────────────────────────────────────────────

/// Match at any available price; residual is cancelled (never rests).
fn process_market(book: &mut OrderBook, mut incoming: Order) -> ExecutionReport {
    let mut trades = Vec::new();
    sweep(book, &mut incoming, None, &mut trades);
    ExecutionReport::filled(trades)
}

// ── IOC ───────────────────────────────────────────────────────────────────────

/// Immediate-or-Cancel: match at the limit price or better; cancel residual.
fn process_ioc(book: &mut OrderBook, mut incoming: Order) -> ExecutionReport {
    let mut trades = Vec::new();
    let price_limit = incoming.price;
    sweep(book, &mut incoming, Some(price_limit), &mut trades);
    ExecutionReport::filled(trades)
}

// ── FOK ───────────────────────────────────────────────────────────────────────

/// Fill-or-Kill: reject entirely if the full quantity cannot be filled at the
/// limit price. No partial fills and no resting residual.
fn process_fok(book: &mut OrderBook, incoming: Order) -> ExecutionReport {
    if !can_fill_fully(book, &incoming) {
        return ExecutionReport::rejected();
    }
    let price_limit = incoming.price;
    let mut incoming = incoming;
    let mut trades = Vec::new();
    sweep(book, &mut incoming, Some(price_limit), &mut trades);
    // Guaranteed by can_fill_fully; assert in debug builds only.
    debug_assert_eq!(incoming.quantity.0, 0, "FOK pre-check must guarantee full fill");
    ExecutionReport::filled(trades)
}

/// Return `true` iff there is sufficient resting liquidity to fully fill
/// `incoming` at its limit price.
fn can_fill_fully(book: &OrderBook, incoming: &Order) -> bool {
    let needed = incoming.quantity.0;
    let available: u64 = match incoming.side {
        Side::Bid => book
            .asks
            .iter()
            .take_while(|(p, _)| **p <= incoming.price)
            .flat_map(|(_, level)| level.iter())
            .map(|o| o.quantity.0)
            .sum(),
        Side::Ask => book
            .bids
            .iter()
            .take_while(|(rp, _)| rp.0 >= incoming.price)
            .flat_map(|(_, level)| level.iter())
            .map(|o| o.quantity.0)
            .sum(),
    };
    available >= needed
}

// ── Core sweep ────────────────────────────────────────────────────────────────

/// Walk the opposite side of the book and execute fills until either
/// `incoming` is exhausted, there is no more resting liquidity, or the price
/// condition (`price_limit`) is no longer satisfied.
fn sweep(
    book: &mut OrderBook,
    incoming: &mut Order,
    price_limit: Option<Price>,
    trades: &mut Vec<Trade>,
) {
    match incoming.side {
        Side::Bid => sweep_against_asks(book, incoming, price_limit, trades),
        Side::Ask => sweep_against_bids(book, incoming, price_limit, trades),
    }
}

fn sweep_against_asks(
    book: &mut OrderBook,
    incoming: &mut Order,
    price_limit: Option<Price>,
    trades: &mut Vec<Trade>,
) {
    loop {
        if incoming.quantity.0 == 0 {
            break;
        }

        let ask_price = match book.asks.keys().next().copied() {
            Some(p) => p,
            None => break,
        };

        // For limit/IOC/FOK: only cross if ask_price <= incoming.price.
        if let Some(lim) = price_limit {
            if ask_price > lim {
                break;
            }
        }

        execute_at_ask_level(book, incoming, ask_price, trades);
    }
}

fn sweep_against_bids(
    book: &mut OrderBook,
    incoming: &mut Order,
    price_limit: Option<Price>,
    trades: &mut Vec<Trade>,
) {
    loop {
        if incoming.quantity.0 == 0 {
            break;
        }

        let bid_price = match book.bids.keys().next().map(|rp| rp.0) {
            Some(p) => p,
            None => break,
        };

        // For limit/IOC/FOK: only cross if bid_price >= incoming.price.
        if let Some(lim) = price_limit {
            if bid_price < lim {
                break;
            }
        }

        execute_at_bid_level(book, incoming, bid_price, trades);
    }
}

/// Drain a single ask price level, filling `incoming` FIFO. If a maker is
/// partially filled it is returned to the front of the level with its
/// remaining quantity (preserving priority per Cont et al., 2010, §3.1).
fn execute_at_ask_level(
    book: &mut OrderBook,
    incoming: &mut Order,
    ask_price: Price,
    trades: &mut Vec<Trade>,
) {
    loop {
        if incoming.quantity.0 == 0 {
            break;
        }

        let maker = match book.pop_front_ask(ask_price) {
            Some(o) => o,
            None => break,
        };

        let fill_qty = incoming.quantity.0.min(maker.quantity.0);
        trades.push(Trade {
            price: ask_price,
            quantity: Quantity(fill_qty),
            maker_id: maker.id,
            taker_id: incoming.id,
            timestamp: incoming.timestamp.max(maker.timestamp),
        });

        incoming.quantity = Quantity(incoming.quantity.0 - fill_qty);
        let maker_rem = maker.quantity.0 - fill_qty;

        if maker_rem > 0 {
            let mut partial = maker;
            partial.quantity = Quantity(maker_rem);
            book.insert_resting_front(partial);
            break; // maker still has quantity → stop draining this level
        }
        // maker fully consumed → continue to next maker at the same level
    }
}

/// Mirror of `execute_at_ask_level` for the bid side.
fn execute_at_bid_level(
    book: &mut OrderBook,
    incoming: &mut Order,
    bid_price: Price,
    trades: &mut Vec<Trade>,
) {
    loop {
        if incoming.quantity.0 == 0 {
            break;
        }

        let maker = match book.pop_front_bid(bid_price) {
            Some(o) => o,
            None => break,
        };

        let fill_qty = incoming.quantity.0.min(maker.quantity.0);
        trades.push(Trade {
            price: bid_price,
            quantity: Quantity(fill_qty),
            maker_id: maker.id,
            taker_id: incoming.id,
            timestamp: incoming.timestamp.max(maker.timestamp),
        });

        incoming.quantity = Quantity(incoming.quantity.0 - fill_qty);
        let maker_rem = maker.quantity.0 - fill_qty;

        if maker_rem > 0 {
            let mut partial = maker;
            partial.quantity = Quantity(maker_rem);
            book.insert_resting_front(partial);
            break;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{
        book::OrderBook,
        order::Order,
        types::{OrderId, OrderType, Price, Quantity, Side, Timestamp},
    };

    fn order(id: u64, side: Side, price: f64, qty: u64, ts: u64, ot: OrderType) -> Order {
        Order::new(
            OrderId(id),
            side,
            Price::from_f64(price),
            Quantity(qty),
            Timestamp(ts),
            ot,
        )
    }

    // ── T1: Limit Buy below best ask rests on the book ────────────────────────

    #[test]
    fn limit_buy_below_ask_rests() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 101.0, 10, 1, OrderType::Limit));

        let report = book.add_limit_order(order(2, Side::Bid, 100.0, 5, 2, OrderType::Limit));

        assert!(report.trades.is_empty(), "no trade should occur");
        assert!(report.remaining.is_some(), "order must rest on book");
        assert_eq!(report.remaining.unwrap().quantity, Quantity(5));
        assert_eq!(book.best_bid().unwrap().0, Price::from_f64(100.0));
        // Ask still untouched
        assert_eq!(book.best_ask().unwrap().0, Price::from_f64(101.0));
    }

    // ── T2: Limit Buy >= best ask triggers a trade ────────────────────────────

    #[test]
    fn limit_buy_at_ask_matches() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 10, 1, OrderType::Limit));

        let report = book.add_limit_order(order(2, Side::Bid, 100.0, 10, 2, OrderType::Limit));

        assert_eq!(report.trades.len(), 1);
        assert_eq!(report.trades[0].price, Price::from_f64(100.0));
        assert_eq!(report.trades[0].quantity, Quantity(10));
        assert_eq!(report.trades[0].maker_id, OrderId(1));
        assert_eq!(report.trades[0].taker_id, OrderId(2));
        assert!(report.remaining.is_none(), "fully filled, nothing should rest");
        assert!(book.best_ask().is_none(), "ask level must be empty");
    }

    #[test]
    fn limit_buy_above_ask_matches_at_ask_price() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 5, 1, OrderType::Limit));

        // Aggressive bid at 105 — trade executes at the resting ask price (100).
        let report = book.add_limit_order(order(2, Side::Bid, 105.0, 5, 2, OrderType::Limit));

        assert_eq!(report.trades.len(), 1);
        assert_eq!(report.trades[0].price, Price::from_f64(100.0));
    }

    // ── T3: Market Buy sweeps multiple ask levels ─────────────────────────────

    #[test]
    fn market_buy_sweeps_ask_levels() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 5, 1, OrderType::Limit));
        book.add_limit_order(order(2, Side::Ask, 101.0, 5, 2, OrderType::Limit));
        book.add_limit_order(order(3, Side::Ask, 102.0, 5, 3, OrderType::Limit));

        // Buy 12 lots — consumes level 100 (5) + level 101 (5) + partial 102 (2).
        let report =
            book.add_limit_order(order(4, Side::Bid, 0.0, 12, 4, OrderType::Market));

        assert_eq!(report.trades.len(), 3);
        let total_filled: u64 = report.trades.iter().map(|t| t.quantity.0).sum();
        assert_eq!(total_filled, 12);
        assert!(report.remaining.is_none());

        // Level 100 and 101 fully consumed; 102 has 3 remaining.
        assert_eq!(book.best_ask().unwrap().0, Price::from_f64(102.0));
        assert_eq!(book.best_ask().unwrap().1, Quantity(3));
    }

    #[test]
    fn market_buy_partial_fill_when_book_has_less_qty() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 3, 1, OrderType::Limit));

        // Buy 10 but only 3 available.
        let report =
            book.add_limit_order(order(2, Side::Bid, 0.0, 10, 2, OrderType::Market));

        assert_eq!(report.trades.len(), 1);
        assert_eq!(report.trades[0].quantity, Quantity(3));
        assert!(report.remaining.is_none(), "residual of market order is cancelled");
        assert!(book.best_ask().is_none());
    }

    // ── T4: FOK rejected when insufficient liquidity ──────────────────────────

    #[test]
    fn fok_insufficient_liquidity_rejected() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 5, 1, OrderType::Limit));

        let report =
            book.add_limit_order(order(2, Side::Bid, 100.0, 10, 2, OrderType::FillOrKill));

        assert!(report.trades.is_empty(), "FOK must produce no trades on rejection");
        assert!(report.remaining.is_none());
        // Resting ask must be untouched.
        assert_eq!(book.best_ask().unwrap().1, Quantity(5));
    }

    #[test]
    fn fok_rejected_due_to_price_constraint() {
        let mut book = OrderBook::new();
        // 10 lots available but at price 101, above the FOK limit of 100.
        book.add_limit_order(order(1, Side::Ask, 101.0, 10, 1, OrderType::Limit));

        let report =
            book.add_limit_order(order(2, Side::Bid, 100.0, 10, 2, OrderType::FillOrKill));

        assert!(report.trades.is_empty());
    }

    #[test]
    fn fok_sufficient_liquidity_fills_completely() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 10, 1, OrderType::Limit));

        let report =
            book.add_limit_order(order(2, Side::Bid, 100.0, 10, 2, OrderType::FillOrKill));

        assert_eq!(report.trades.len(), 1);
        assert_eq!(report.trades[0].quantity, Quantity(10));
        assert!(report.remaining.is_none());
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn fok_fills_across_multiple_levels() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 5, 1, OrderType::Limit));
        book.add_limit_order(order(2, Side::Ask, 101.0, 5, 2, OrderType::Limit));

        let report =
            book.add_limit_order(order(3, Side::Bid, 101.0, 10, 3, OrderType::FillOrKill));

        assert_eq!(report.trades.len(), 2);
        let total: u64 = report.trades.iter().map(|t| t.quantity.0).sum();
        assert_eq!(total, 10);
    }

    // ── T5: Cancel removes order and cleans the index ─────────────────────────

    #[test]
    fn cancel_removes_order_and_cleans_index() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Bid, 100.0, 10, 1, OrderType::Limit));
        book.add_limit_order(order(2, Side::Bid, 99.0, 5, 2, OrderType::Limit));

        let cancelled = book.cancel_order(OrderId(1)).unwrap();
        assert_eq!(cancelled.quantity, Quantity(10));
        assert!(!book.order_index.contains_key(&OrderId(1)));
        // Level at 100 fully pruned; best bid moves down.
        assert_eq!(book.best_bid().unwrap().0, Price::from_f64(99.0));
    }

    #[test]
    fn cancel_unknown_id_returns_none() {
        let mut book = OrderBook::new();
        assert!(book.cancel_order(OrderId(999)).is_none());
    }

    #[test]
    fn cancelled_order_is_not_matched() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 10, 1, OrderType::Limit));
        book.cancel_order(OrderId(1));

        let report =
            book.add_limit_order(order(2, Side::Bid, 100.0, 10, 2, OrderType::Limit));

        assert!(report.trades.is_empty(), "cancelled order must not match");
        assert_eq!(book.best_bid().unwrap().0, Price::from_f64(100.0));
    }

    // ── T6: Modify down preserves time priority ───────────────────────────────

    #[test]
    fn modify_down_preserves_priority() {
        let mut book = OrderBook::new();
        // order 1 arrives first (older), order 2 second.
        book.add_limit_order(order(1, Side::Bid, 100.0, 10, 1, OrderType::Limit));
        book.add_limit_order(order(2, Side::Bid, 100.0, 10, 2, OrderType::Limit));

        // Reduce order 1 from 10 → 5. It must remain at the head of the queue.
        assert!(book.modify_order(OrderId(1), Quantity(5)));

        // Market sell 5: should fill against order 1 (older, still at front).
        let report =
            book.add_limit_order(order(3, Side::Ask, 0.0, 5, 3, OrderType::Market));

        assert_eq!(report.trades.len(), 1);
        assert_eq!(report.trades[0].maker_id, OrderId(1));
        assert_eq!(report.trades[0].quantity, Quantity(5));
        // order 2 (qty 10) is untouched.
        assert_eq!(book.best_bid().unwrap().1, Quantity(10));
    }

    #[test]
    fn modify_up_is_rejected_without_state_change() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Bid, 100.0, 5, 1, OrderType::Limit));

        // Attempt to increase quantity — must be rejected.
        assert!(!book.modify_order(OrderId(1), Quantity(20)));
        assert_eq!(book.best_bid().unwrap().1, Quantity(5));
    }

    // ── T7: Market order on empty book produces no trade ──────────────────────

    #[test]
    fn market_order_empty_book_no_trade() {
        let mut book = OrderBook::new();
        let report =
            book.add_limit_order(order(1, Side::Bid, 0.0, 10, 1, OrderType::Market));

        assert!(report.trades.is_empty());
        assert!(report.remaining.is_none());
        // Market order residual must not rest.
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn market_sell_empty_book_no_trade() {
        let mut book = OrderBook::new();
        let report =
            book.add_limit_order(order(1, Side::Ask, 0.0, 10, 1, OrderType::Market));

        assert!(report.trades.is_empty());
        assert!(report.remaining.is_none());
    }

    // ── Additional: partial fill, IOC, depth ─────────────────────────────────

    #[test]
    fn limit_partial_fill_rests_remainder() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 3, 1, OrderType::Limit));

        let report =
            book.add_limit_order(order(2, Side::Bid, 100.0, 10, 2, OrderType::Limit));

        assert_eq!(report.trades.len(), 1);
        assert_eq!(report.trades[0].quantity, Quantity(3));
        let rem = report.remaining.unwrap();
        assert_eq!(rem.id, OrderId(2));
        assert_eq!(rem.quantity, Quantity(7));
        assert_eq!(book.best_bid().unwrap().1, Quantity(7));
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn ioc_partial_match_cancels_residual() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 3, 1, OrderType::Limit));

        let report = book.add_limit_order(
            order(2, Side::Bid, 100.0, 10, 2, OrderType::ImmediateOrCancel),
        );

        assert_eq!(report.trades[0].quantity, Quantity(3));
        assert!(report.remaining.is_none(), "IOC residual must not rest");
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn ioc_no_match_no_rest() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 105.0, 10, 1, OrderType::Limit));

        let report = book.add_limit_order(
            order(2, Side::Bid, 100.0, 10, 2, OrderType::ImmediateOrCancel),
        );

        assert!(report.trades.is_empty());
        assert!(report.remaining.is_none());
        assert!(book.best_bid().is_none()); // must not rest
    }

    #[test]
    fn fifo_priority_within_price_level() {
        let mut book = OrderBook::new();
        // Three asks at the same price, arriving in order.
        book.add_limit_order(order(1, Side::Ask, 100.0, 5, 1, OrderType::Limit));
        book.add_limit_order(order(2, Side::Ask, 100.0, 5, 2, OrderType::Limit));
        book.add_limit_order(order(3, Side::Ask, 100.0, 5, 3, OrderType::Limit));

        let report =
            book.add_limit_order(order(4, Side::Bid, 100.0, 5, 4, OrderType::Market));

        assert_eq!(report.trades.len(), 1);
        assert_eq!(report.trades[0].maker_id, OrderId(1), "oldest maker matches first");
    }

    #[test]
    fn partial_maker_retains_front_position() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 10, 1, OrderType::Limit));
        book.add_limit_order(order(2, Side::Ask, 100.0, 10, 2, OrderType::Limit));

        // First buy 4 lots — partially fills order 1.
        let r1 = book.add_limit_order(order(3, Side::Bid, 100.0, 4, 3, OrderType::Market));
        assert_eq!(r1.trades[0].maker_id, OrderId(1));

        // Second buy 6 lots — should drain the remaining 6 of order 1 before touching order 2.
        let r2 = book.add_limit_order(order(4, Side::Bid, 100.0, 6, 4, OrderType::Market));
        assert_eq!(r2.trades[0].maker_id, OrderId(1));
        assert_eq!(r2.trades[0].quantity, Quantity(6));
    }

    #[test]
    fn ask_limit_matches_against_bid() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Bid, 100.0, 10, 1, OrderType::Limit));

        let report =
            book.add_limit_order(order(2, Side::Ask, 100.0, 10, 2, OrderType::Limit));

        assert_eq!(report.trades.len(), 1);
        assert_eq!(report.trades[0].price, Price::from_f64(100.0));
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn fully_filled_report_has_no_remaining() {
        let mut book = OrderBook::new();
        book.add_limit_order(order(1, Side::Ask, 100.0, 5, 1, OrderType::Limit));

        let report =
            book.add_limit_order(order(2, Side::Bid, 100.0, 5, 2, OrderType::Limit));

        assert!(report.is_fully_filled());
        assert!(report.remaining.is_none());
        assert_eq!(report.filled_quantity(), Quantity(5));
    }
}
