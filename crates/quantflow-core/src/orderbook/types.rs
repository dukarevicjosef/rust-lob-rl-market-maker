use std::fmt;
use std::ops::{Add, Sub};

use serde::{Deserialize, Serialize};

/// Fixed-point price representation: 1.00 == 100_000_000 (10^8 ticks per unit).
/// Using i64 to support synthetic instruments and negative spreads.
pub const PRICE_SCALE: i64 = 100_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Price(pub i64);

impl Price {
    #[inline]
    pub const fn from_ticks(ticks: i64) -> Self {
        Price(ticks)
    }

    /// Construct from a float — lossy, for testing and config only.
    /// Hot-path code must use tick arithmetic directly.
    pub fn from_f64(v: f64) -> Self {
        Price((v * PRICE_SCALE as f64).round() as i64)
    }

    #[inline]
    pub fn ticks(self) -> i64 {
        self.0
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / PRICE_SCALE as f64
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let abs = self.0.unsigned_abs();
        let sign = if self.0 < 0 { "-" } else { "" };
        let scale = PRICE_SCALE as u64;
        write!(f, "{}{}.{:08}", sign, abs / scale, abs % scale)
    }
}

impl Add for Price {
    type Output = Price;
    #[inline]
    fn add(self, rhs: Price) -> Price {
        Price(self.0 + rhs.0)
    }
}

impl Sub for Price {
    type Output = Price;
    #[inline]
    fn sub(self, rhs: Price) -> Price {
        Price(self.0 - rhs.0)
    }
}

// ── Quantity ─────────────────────────────────────────────────────────────────

/// Order quantity in integer lots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Quantity(pub u64);

impl Quantity {
    #[inline]
    pub const fn new(lots: u64) -> Self {
        Quantity(lots)
    }

    #[inline]
    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── Identifiers & timestamps ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OrderId(pub u64);

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Nanosecond-resolution wall-clock timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(pub u64);

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ns", self.0)
    }
}

// ── Side ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    #[inline]
    pub fn opposite(self) -> Side {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Bid => write!(f, "BID"),
            Side::Ask => write!(f, "ASK"),
        }
    }
}

// ── OrderType ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    /// Rests at limit price; price-time priority.
    Limit,
    /// Crosses immediately; any residual is cancelled.
    Market,
    /// Crosses immediately; residual is cancelled (same behaviour as Market for our LOB).
    ImmediateOrCancel,
    /// Must fill completely or the entire order is rejected.
    FillOrKill,
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn price_scale_round_trip() {
        let p = Price::from_f64(1.0);
        assert_eq!(p.ticks(), PRICE_SCALE);
        assert_eq!(p.to_f64(), 1.0);
    }

    #[test]
    fn price_display_integer() {
        assert_eq!(Price::from_f64(1.0).to_string(), "1.00000000");
        assert_eq!(Price::from_f64(100.5).to_string(), "100.50000000");
    }

    #[test]
    fn price_display_negative() {
        assert_eq!(Price::from_f64(-0.5).to_string(), "-0.50000000");
    }

    #[test]
    fn price_display_sub_unit() {
        // 0.00000001 == 1 tick
        let p = Price::from_ticks(1);
        assert_eq!(p.to_string(), "0.00000001");
    }

    #[test]
    fn price_add_sub() {
        let a = Price::from_f64(1.5);
        let b = Price::from_f64(0.25);
        assert_eq!((a + b).to_f64(), 1.75);
        assert_eq!((a - b).to_f64(), 1.25);
    }

    #[test]
    fn price_sub_negative() {
        let a = Price::from_f64(1.0);
        let b = Price::from_f64(2.0);
        assert_eq!((a - b).ticks(), -PRICE_SCALE);
    }

    #[test]
    fn price_ordering() {
        let low = Price::from_f64(99.0);
        let high = Price::from_f64(100.0);
        assert!(low < high);
        assert!(high > low);
        assert_eq!(low, low);
    }

    #[test]
    fn price_ordering_negative() {
        let neg = Price::from_ticks(-1);
        let zero = Price::from_ticks(0);
        assert!(neg < zero);
    }

    #[test]
    fn price_tick_arithmetic_exact() {
        // Verify no floating-point drift for common tick offsets.
        let base = Price::from_ticks(10_000_000_000); // 100.00
        let tick = Price::from_ticks(1);
        assert_eq!((base + tick).ticks(), 10_000_000_001);
    }

    #[test]
    fn side_opposite() {
        assert_eq!(Side::Bid.opposite(), Side::Ask);
        assert_eq!(Side::Ask.opposite(), Side::Bid);
    }

    #[test]
    fn quantity_ordering() {
        assert!(Quantity::new(10) > Quantity::new(5));
        assert_eq!(Quantity::new(0), Quantity::new(0));
    }
}
