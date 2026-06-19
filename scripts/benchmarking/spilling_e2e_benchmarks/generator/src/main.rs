/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

//! TCP CSV row producer for the spilling end-to-end benchmark.
//!
//! Listens on ONE port and streams `id,value1,value2,timestamp\n` rows to every accepted
//! client. `id` and `value1` are drawn from a precomputed lookup table seeded by `--seed`
//! (deterministic); `value2` is a fixed ASCII payload of `--var-size` bytes (all `'a'`,
//! chosen because it contains neither `,` nor `\n` and so cannot confuse the CSV parser);
//! `timestamp` is a per-connection monotonic counter starting at 0.
//!
//! One process per source — the Python orchestrator spawns 2 × `len(WINDOWS)` of these
//! per benchmark cell, one per logical stream side.

use std::io;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::net::TcpListener;
use tokio::time::Instant;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "spilling-bench-tcp-gen",
    about = "TCP CSV producer for spill bench"
)]
struct Args {
    /// Port to bind.
    #[arg(long, default_value_t = 9100)]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    bind: IpAddr,

    /// Bytes of varsized payload (`value2` column). 0 emits an empty field.
    #[arg(long, default_value_t = 16)]
    var_size: usize,

    /// Size of the (id, value1) lookup table.
    #[arg(long, default_value_t = 65_536)]
    lookup_size: usize,

    /// PRNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Target ingestion rate in tuples/sec per connection. 0 = unbounded (as fast as the socket accepts).
    #[arg(long, default_value_t = 0)]
    rate: u64,
}

/// Sleep duration to hold `rate` tuples/sec after emitting `ts` tuples in `elapsed`. None = don't sleep.
/// Returns None when unbounded (rate 0) or already behind schedule.
fn pace_delay(rate: u64, ts: u64, elapsed: Duration) -> Option<Duration> {
    if rate == 0 {
        return None;
    }
    let expected = Duration::from_secs_f64(ts as f64 / rate as f64);
    expected.checked_sub(elapsed)
}

// Xorshift64* — deterministic, fine for benchmark data.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

fn build_lookup(seed: u64, size: usize) -> Vec<(u64, u64)> {
    let mut rng = Xorshift64::new(seed);
    let mut out = Vec::with_capacity(size);
    for _ in 0..size {
        let id = rng.next_u64() % 5_000;
        let value1 = rng.next_u64() % 500_000;
        out.push((id, value1));
    }
    out
}

async fn handle_connection(
    sock: tokio::net::TcpStream,
    peer: std::net::SocketAddr,
    port: u16,
    rate: u64,
    lookup: Arc<Vec<(u64, u64)>>,
    payload: Arc<Vec<u8>>,
) {
    eprintln!("accept port={} peer={} rate={}", port, peer, rate);
    let _ = sock.set_nodelay(true);
    let mut writer = BufWriter::with_capacity(64 * 1024, sock);

    let mut ts: u64 = 0;
    let mut idx: usize = 0;
    let lookup_len = lookup.len();
    let pace_start = Instant::now();

    let mut id_buf = itoa::Buffer::new();
    let mut val_buf = itoa::Buffer::new();
    let mut ts_buf = itoa::Buffer::new();

    loop {
        let (id, value1) = lookup[idx % lookup_len];
        let id_s = id_buf.format(id).as_bytes();
        let val_s = val_buf.format(value1).as_bytes();
        let ts_s = ts_buf.format(ts).as_bytes();

        let res = async {
            writer.write_all(id_s).await?;
            writer.write_all(b",").await?;
            writer.write_all(val_s).await?;
            writer.write_all(b",").await?;
            writer.write_all(&payload).await?;
            writer.write_all(b",").await?;
            writer.write_all(ts_s).await?;
            writer.write_all(b"\n").await
        }
        .await;

        if let Err(e) = res {
            match e.kind() {
                io::ErrorKind::BrokenPipe
                | io::ErrorKind::ConnectionReset
                | io::ErrorKind::UnexpectedEof => {
                    eprintln!("disconnect port={} peer={} tuples={}", port, peer, ts);
                }
                _ => {
                    eprintln!(
                        "write error port={} peer={} kind={:?}: {}",
                        port,
                        peer,
                        e.kind(),
                        e
                    );
                }
            }
            return;
        }

        ts = ts.wrapping_add(1);
        idx = idx.wrapping_add(1);

        // Rate pacing checked once per chunk, not per tuple, so we never sleep sub-µs.
        // Ceiling: tokio sleep resolution (~1ms) bounds accuracy; a 1024-tuple chunk keeps the
        // error well under that even at 1 MTup/s. Flush first so the chunk lands during the sleep.
        if rate > 0 && ts % 1024 == 0 {
            if let Some(d) = pace_delay(rate, ts, pace_start.elapsed()) {
                if writer.flush().await.is_err() {
                    return;
                }
                tokio::time::sleep(d).await;
            }
        }
    }
}

async fn run_listener(
    listener: TcpListener,
    port: u16,
    rate: u64,
    lookup: Arc<Vec<(u64, u64)>>,
    payload: Arc<Vec<u8>>,
) {
    loop {
        match listener.accept().await {
            Ok((sock, peer)) => {
                let lookup = lookup.clone();
                let payload = payload.clone();
                tokio::spawn(async move {
                    handle_connection(sock, peer, port, rate, lookup, payload).await;
                });
            }
            Err(e) => {
                eprintln!("accept error port={}: {}", port, e);
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> io::Result<()> {
    let args = Args::parse();
    let lookup = Arc::new(build_lookup(args.seed, args.lookup_size));
    let payload = Arc::new(vec![b'a'; args.var_size]);
    eprintln!(
        "lookup built: size={} seed={} var_size={} rate={} (id [0,5000), value1 [0,500000))",
        lookup.len(),
        args.seed,
        args.var_size,
        args.rate
    );

    let addr = std::net::SocketAddr::new(args.bind, args.port);
    let listener = TcpListener::bind(addr).await.map_err(|e| {
        eprintln!("bind failed addr={}: {}", addr, e);
        e
    })?;

    // READY handshake — the Python runner blocks on this line before submitting queries.
    println!("READY");
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    eprintln!("listening on {}", addr);

    tokio::spawn(
        async move { run_listener(listener, args.port, args.rate, lookup, payload).await },
    );

    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        _ = tokio::signal::ctrl_c() => { eprintln!("SIGINT received, shutting down"); }
        _ = sigterm.recv() => { eprintln!("SIGTERM received, shutting down"); }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacing() {
        // rate 0 is unbounded: never sleep.
        assert_eq!(pace_delay(0, 1_000_000, Duration::from_secs(0)), None);
        // Behind schedule (2s elapsed for 1s of tuples): never sleep.
        assert_eq!(pace_delay(1000, 1000, Duration::from_secs(2)), None);
        // 1000 tuples at 1000/s should take 1s; only 0.5s elapsed -> sleep the remaining 0.5s.
        assert_eq!(
            pace_delay(1000, 1000, Duration::from_millis(500)),
            Some(Duration::from_millis(500))
        );
    }
}
