/// Integration tests for ParquetRecorder.
///
/// Network tests are marked #[ignore] and only run with:
///   cargo test --test recorder_tests -- --include-ignored
use quantflow_core::data::binance_ws::BinanceMessage;
use quantflow_core::data::recorder::ParquetRecorder;

// ── Parquet round-trip (no network) ──────────────────────────────────────────

#[test]
fn recorder_creates_parquet_files() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut recorder = ParquetRecorder::new(dir.path(), "btcusdt").unwrap();
    let stats = recorder.finish().unwrap();
    assert_eq!(stats.depth_count, 0);
    assert_eq!(stats.trade_count, 0);
    assert_eq!(stats.ticker_count, 0);

    // Three Parquet files must exist even when empty.
    let raw = dir.path().join("btcusdt").join("raw");
    let files: Vec<_> = std::fs::read_dir(&raw)
        .unwrap()
        .flatten()
        .collect();
    assert_eq!(files.len(), 3, "expected depth, trades, bookticker files");
}

#[test]
fn recorder_counts_messages_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let mut recorder = ParquetRecorder::new(dir.path(), "btcusdt").unwrap();

    // Feed synthetic depth, trade and ticker messages.
    let depth_json = r#"{"stream":"btcusdt@depth","data":{"e":"depthUpdate","E":1000,"s":"BTCUSDT","U":1,"u":2,"b":[["70100.00","0.5"]],"a":[["70101.00","0.3"]]}}"#;
    let trade_json = r#"{"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1000,"s":"BTCUSDT","a":1,"p":"70100.00","q":"0.01","f":1,"l":1,"T":1000,"m":false}}"#;
    let ticker_json = r#"{"stream":"btcusdt@bookTicker","data":{"u":1,"s":"BTCUSDT","b":"70100.00","B":"0.5","a":"70101.00","A":"0.3"}}"#;

    for _ in 0..5 {
        let wrapper: serde_json::Value = serde_json::from_str(depth_json).unwrap();
        let d = serde_json::from_value(wrapper["data"].clone()).unwrap();
        recorder.record(BinanceMessage::Depth(d)).unwrap();
    }
    for _ in 0..3 {
        let wrapper: serde_json::Value = serde_json::from_str(trade_json).unwrap();
        let t = serde_json::from_value(wrapper["data"].clone()).unwrap();
        recorder.record(BinanceMessage::Trade(t)).unwrap();
    }
    for _ in 0..7 {
        let wrapper: serde_json::Value = serde_json::from_str(ticker_json).unwrap();
        let b = serde_json::from_value(wrapper["data"].clone()).unwrap();
        recorder.record(BinanceMessage::BookTicker(b)).unwrap();
    }

    let stats = recorder.finish().unwrap();
    assert_eq!(stats.depth_count, 5);
    assert_eq!(stats.trade_count, 3);
    assert_eq!(stats.ticker_count, 7);
    assert!(stats.bytes_written > 0, "bytes_written must reflect file sizes");
}

#[test]
fn recorder_parquet_files_are_readable() {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let dir = tempfile::tempdir().unwrap();
    let mut recorder = ParquetRecorder::new(dir.path(), "btcusdt").unwrap();

    let trade_json = r#"{"e":"aggTrade","E":1000,"s":"BTCUSDT","a":1,"p":"70100.00","q":"0.01","f":1,"l":1,"T":1000,"m":false}"#;
    for _ in 0..10 {
        let t = serde_json::from_str(trade_json).unwrap();
        recorder.record(BinanceMessage::Trade(t)).unwrap();
    }
    recorder.finish().unwrap();

    let raw = dir.path().join("btcusdt").join("raw");
    let trade_file: Vec<_> = std::fs::read_dir(&raw)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().contains("trades"))
        .collect();
    assert_eq!(trade_file.len(), 1);

    let reader = SerializedFileReader::new(
        std::fs::File::open(trade_file[0].path()).unwrap(),
    )
    .unwrap();
    let meta = reader.metadata();
    let total_rows: i64 = (0..meta.num_row_groups())
        .map(|i| meta.row_group(i).num_rows())
        .sum();
    assert_eq!(total_rows, 10);
}

// ── Network integration test (ignored by default) ────────────────────────────

#[tokio::test]
#[ignore = "requires Binance WebSocket connectivity"]
async fn recorder_captures_live_data_10s() {
    use quantflow_core::data::binance_ws::BinanceClient;

    let dir = tempfile::tempdir().unwrap();
    let mut recorder = ParquetRecorder::new(dir.path(), "btcusdt").unwrap();

    let client = BinanceClient::new("btcusdt", true);
    let (tx, mut rx) = tokio::sync::mpsc::channel(10_000);

    let handle = tokio::spawn(async move {
        let _ = client.stream(tx).await;
    });

    let start = tokio::time::Instant::now();
    while start.elapsed() < tokio::time::Duration::from_secs(10) {
        match tokio::time::timeout(
            tokio::time::Duration::from_secs(5),
            rx.recv(),
        )
        .await
        {
            Ok(Some(msg)) => recorder.record(msg).unwrap(),
            _ => break,
        }
    }
    handle.abort();

    let stats = recorder.finish().unwrap();
    assert!(stats.depth_count > 0, "should have received depth updates");
    assert!(stats.bytes_written > 0);

    let raw = dir.path().join("btcusdt").join("raw");
    assert!(raw.join(format!("{}_depth.parquet", chrono::Utc::now().format("%Y-%m-%d"))).exists());
}
