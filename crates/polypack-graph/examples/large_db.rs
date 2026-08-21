//! Example: a 1M-node property graph — scale, persistence, and large-db queries
//!
//! Rust port of `examples/large-db.ts`. Seeds a synthetic graph to disk with
//! a filesystem-backed `Storage`, then reopens it to demonstrate the
//! features that matter at scale:
//!   - insert throughput against a real on-disk store
//!   - LRU working set: only `hot_cache_max` nodes stay loaded; the rest live on disk
//!   - persisted queries that scan the FULL backing store without warming it
//!   - in-memory queries over the hot working set after `warm()`
//!   - ownership cascade across persisted data
//!
//! Data is deterministic (seeded mulberry32, the same PRNG as the TS
//! example). Schema (fan-out 3×3×3):
//!
//!     user ─[OWNS, owned]→ document ─[CONTAINS, owned]→ section ─[CONTAINS, owned]→ chunk
//!     user ─[FOLLOWS]→ user
//!     document ─[CITES]→ document
//!
//! Documents carry 8-dim vectors drawn near one of 12 topic centroids, so
//! `similar_to` returns explainably coherent results. Text embeddings are
//! skipped to keep a single vector dimension in the index, same as the TS
//! version — plug in your own `EmbeddingProvider` to replace the centroids.
//!
//! Run:
//!   cargo run --release --example large_db --manifest-path crates/Cargo.toml -- --count 1000000
//!   cargo run --release --example large_db --manifest-path crates/Cargo.toml -- --count 100000  # quick smoke test
//!   cargo run --release --example large_db --manifest-path crates/Cargo.toml -- --wipe           # regenerate the store
//!
//! Flags:
//!   --count N   total nodes (default 1,000,000)
//!   --hot N     hot_cache_max LRU size (default 50,000)
//!   --dir PATH  store directory (default target/.polypack-large-db)
//!   --wipe      delete the store before seeding
//!   --seed N    PRNG seed (default 42)
//!
//! Unlike the TS version's `BinaryStoreAdapter`, this crate's `Store` keeps
//! its full node/edge/vector maps resident in memory too, so a 1M-node run
//! still uses a few GB of RSS. Use `--count 100000` to smoke-test.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use polypack_core::query::Direction;
use polypack_core::vector::cosine;
use polypack_core::{Node, PolypackError, Result, Storage, StoreConfig};
use polypack_graph::{AggregateOp, EdgeOwnership, Graph, GraphConfig, OrderDirection};
use serde_json::json;

// ────────────────────────────────────────────────────────────
// Config
// ────────────────────────────────────────────────────────────

const DEFAULT_COUNT: usize = 1_000_000;
const DEFAULT_HOT: usize = 50_000;
const TOPICS: [&str; 12] = [
    "graphs", "databases", "search", "compilers", "systems", "ml", "security", "storage", "networks",
    "distributed", "parsing", "runtime",
];
const DIMS: usize = 8;
const COUNTRIES: [&str; 10] = ["us", "de", "jp", "gb", "fr", "br", "in", "ca", "au", "nl"];
const FLUSH_EVERY: usize = 25_000;

// ────────────────────────────────────────────────────────────
// CLI
// ────────────────────────────────────────────────────────────

struct Args {
    count: usize,
    hot_cache_max: usize,
    store_dir: PathBuf,
    seed: u32,
    wipe: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let value_of = |flag: &str| -> Option<String> {
        if let Some(eq) = argv.iter().find(|a| a.starts_with(&format!("{flag}="))) {
            return Some(eq[flag.len() + 1..].to_string());
        }
        let i = argv.iter().position(|a| a == flag)?;
        argv.get(i + 1).cloned()
    };
    Args {
        count: value_of("--count").and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_COUNT),
        hot_cache_max: value_of("--hot").and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_HOT),
        store_dir: value_of("--dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.polypack-large-db")),
        seed: value_of("--seed").and_then(|s| s.parse().ok()).unwrap_or(42),
        wipe: argv.iter().any(|a| a == "--wipe"),
    }
}

// ────────────────────────────────────────────────────────────
// PRNG + synthetic data
// ────────────────────────────────────────────────────────────

/// Same PRNG as the TS example's `mulberry32`, bit-for-bit: 32-bit wrapping
/// arithmetic, seeded, deterministic.
struct Mulberry32(u32);

impl Mulberry32 {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let mut t = self.0;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

fn build_centroids(seed: u32) -> HashMap<&'static str, Vec<f64>> {
    let mut rng = Mulberry32::new(seed ^ 0x51ab);
    let mut centroids = HashMap::new();
    for topic in TOPICS {
        let mut v = Vec::with_capacity(DIMS);
        let mut norm = 0.0;
        for _ in 0..DIMS {
            let x = rng.next() * 2.0 - 1.0;
            v.push(x);
            norm += x * x;
        }
        let n = norm.sqrt();
        centroids.insert(topic, v.into_iter().map(|x| x / n).collect());
    }
    centroids
}

/// 8-dim vector: topic centroid + small noise, so cosine similarity clusters.
fn doc_vector(rng: &mut Mulberry32, centroid: &[f64]) -> Vec<f64> {
    centroid.iter().map(|c| c + (rng.next() * 2.0 - 1.0) * 0.05).collect()
}

// Node id helpers — zero-padded so lexicographic order matches insertion order.
fn uid(i: usize) -> String {
    format!("u{i:05}")
}
fn did(i: usize) -> String {
    format!("d{i:06}")
}
fn sid(i: usize) -> String {
    format!("s{i:07}")
}
fn cid(i: usize) -> String {
    format!("c{i:07}")
}

// ────────────────────────────────────────────────────────────
// Formatting helpers
// ────────────────────────────────────────────────────────────

fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 { format!("{:.2}s", ms / 1000.0) } else { format!("{ms:.0}ms") }
}

fn fmt_rate(n: usize, ms: f64) -> String {
    format!("{}/s", fmt_num((n as f64 / (ms / 1000.0)).round() as usize))
}

fn fmt_mb(bytes: u64) -> String {
    format!("{:.0}MB", bytes as f64 / (1024.0 * 1024.0))
}

/// Best-effort resident-set size — Linux only (reads `/proc/self/status`);
/// unlike Node's `process.memoryUsage()`, Rust has no cross-platform
/// heap-size query built in, and RSS here is diagnostic only.
fn rss_mb() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(kb) = line.strip_prefix("VmRSS:") {
                    if let Ok(kb) = kb.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                        return fmt_mb(kb * 1024);
                    }
                }
            }
        }
    }
    "n/a".to_string()
}

fn store_size_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

fn section(title: &str) {
    println!("\n── {title} ──");
}

// ────────────────────────────────────────────────────────────
// Filesystem storage (mirrors polypack-node's FsStorage)
// ────────────────────────────────────────────────────────────

struct FsStorage {
    dir: PathBuf,
}

impl Storage for FsStorage {
    fn read(&self, name: &str) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.dir.join(name)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(PolypackError::Storage(e.to_string())),
        }
    }

    fn write(&mut self, name: &str, data: &[u8]) -> Result<()> {
        std::fs::create_dir_all(&self.dir).map_err(|e| PolypackError::Storage(e.to_string()))?;
        // Write-then-rename: a crash mid-write leaves the previous snapshot
        // intact instead of a torn file.
        let tmp_path = self.dir.join(format!("{name}.tmp"));
        let mut file = std::fs::File::create(&tmp_path).map_err(|e| PolypackError::Storage(e.to_string()))?;
        file.write_all(data).map_err(|e| PolypackError::Storage(e.to_string()))?;
        file.sync_all().map_err(|e| PolypackError::Storage(e.to_string()))?;
        drop(file);
        std::fs::rename(&tmp_path, self.dir.join(name)).map_err(|e| PolypackError::Storage(e.to_string()))
    }

    fn append(&mut self, name: &str, data: &[u8]) -> Result<()> {
        std::fs::create_dir_all(&self.dir).map_err(|e| PolypackError::Storage(e.to_string()))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join(name))
            .map_err(|e| PolypackError::Storage(e.to_string()))?;
        file.write_all(data).map_err(|e| PolypackError::Storage(e.to_string()))
    }

    fn delete(&mut self, name: &str) -> Result<()> {
        let _ = std::fs::remove_file(self.dir.join(name));
        Ok(())
    }

    fn exists(&self, name: &str) -> Result<bool> {
        Ok(self.dir.join(name).exists())
    }
}

// ────────────────────────────────────────────────────────────
// Phase A — Seed
// ────────────────────────────────────────────────────────────

fn add_nodes_in_chunks(graph: &mut Graph, count: usize, mut make: impl FnMut(usize) -> Node) -> Result<()> {
    let mut batch = Vec::with_capacity(FLUSH_EVERY.min(count));
    for i in 0..count {
        batch.push(make(i));
        if batch.len() >= FLUSH_EVERY {
            graph.add_nodes(std::mem::take(&mut batch))?;
            graph.flush()?;
        }
    }
    if !batch.is_empty() {
        graph.add_nodes(batch)?;
        graph.flush()?;
    }
    Ok(())
}

fn seed(graph: &mut Graph, args: &Args, centroids: &HashMap<&'static str, Vec<f64>>) -> Result<()> {
    let users = (args.count / 40).max(1);
    let docs = users * 3;
    let sections = users * 9;
    let chunks = users * 27;
    let total_nodes = users + docs + sections + chunks;
    let total_edges = users * 5 + docs * 6 + sections * 3;
    let mut rng = Mulberry32::new(args.seed);
    let now = 0i64;

    println!("  Seeding {} nodes + {} edges ({} users)", fmt_num(total_nodes), fmt_num(total_edges), fmt_num(users));
    println!("  Topics: {}\n", TOPICS.join(", "));

    let t0 = Instant::now();
    add_nodes_in_chunks(graph, users, |i| Node {
        id: uid(i),
        node_type: "user".into(),
        data: json!({
            "name": format!("user-{i}"),
            "country": COUNTRIES[(rng.next() * COUNTRIES.len() as f64) as usize],
            "rep": (rng.next() * 1000.0).floor() as i64,
        })
        .as_object()
        .unwrap()
        .clone(),
        vector: None,
        inserted_at: now,
        updated_at: now,
        revision: 0,
        activation: None, ..Default::default()
})?;
    add_nodes_in_chunks(graph, docs, |i| {
        let topic = TOPICS[(rng.next() * TOPICS.len() as f64) as usize];
        Node {
            id: did(i),
            node_type: "document".into(),
            data: json!({ "title": format!("doc-{i}"), "topic": topic, "score": (rng.next() * 101.0).floor() as i64 })
                .as_object()
                .unwrap()
                .clone(),
            vector: Some(doc_vector(&mut rng, &centroids[topic])),
            inserted_at: now,
            updated_at: now,
            revision: 0,
        activation: None, ..Default::default()
}
    })?;
    add_nodes_in_chunks(graph, sections, |i| Node {
        id: sid(i),
        node_type: "section".into(),
        data: json!({ "heading": format!("section-{i}") }).as_object().unwrap().clone(),
        vector: None,
        inserted_at: now,
        updated_at: now,
        revision: 0,
        activation: None, ..Default::default()
})?;
    add_nodes_in_chunks(graph, chunks, |i| {
        let r = rng.next();
        let quality = if r < 0.33 { "low" } else if r < 0.66 { "mid" } else { "high" };
        Node {
            id: cid(i),
            node_type: "chunk".into(),
            data: json!({ "text": format!("chunk-{i} text"), "score": (rng.next() * 101.0).floor() as i64, "quality": quality })
                .as_object()
                .unwrap()
                .clone(),
            vector: None,
            inserted_at: now,
            updated_at: now,
            revision: 0,
        activation: None, ..Default::default()
}
    })?;
    let node_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  nodes   {} in {}  ({})  rss={}",
        fmt_num(total_nodes),
        fmt_ms(node_ms),
        fmt_rate(total_nodes, node_ms),
        rss_mb()
    );

    // ── Edges (owned hierarchy + reference links) ──
    let t1 = Instant::now();
    let mut since_report = 0usize;
    for i in 0..users {
        for j in 0..3 {
            graph.add_edge(&uid(i), "OWNS", &did(i * 3 + j), None, EdgeOwnership::Owned)?;
        }
        for _ in 0..2 {
            let target = (rng.next() * users as f64) as usize;
            graph.add_edge(&uid(i), "FOLLOWS", &uid(target), None, EdgeOwnership::Reference)?;
        }
        since_report += 5;
        if since_report >= FLUSH_EVERY {
            graph.flush()?;
            since_report = 0;
        }
    }
    for i in 0..docs {
        for j in 0..3 {
            graph.add_edge(&did(i), "CONTAINS", &sid(i * 3 + j), None, EdgeOwnership::Owned)?;
        }
        for _ in 0..3 {
            let target = (rng.next() * docs as f64) as usize;
            graph.add_edge(&did(i), "CITES", &did(target), None, EdgeOwnership::Reference)?;
        }
        since_report += 6;
        if since_report >= FLUSH_EVERY {
            graph.flush()?;
            since_report = 0;
        }
    }
    for i in 0..sections {
        for j in 0..3 {
            graph.add_edge(&sid(i), "CONTAINS", &cid(i * 3 + j), None, EdgeOwnership::Owned)?;
        }
        since_report += 3;
        if since_report >= FLUSH_EVERY {
            graph.flush()?;
            since_report = 0;
        }
    }
    graph.flush()?;
    let edge_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("  edges   {} in {}  ({})  rss={}", fmt_num(total_edges), fmt_ms(edge_ms), fmt_rate(total_edges, edge_ms), rss_mb());

    println!("  on-disk store: {} in {}", fmt_mb(store_size_bytes(&args.store_dir)), args.store_dir.display());
    Ok(())
}

// ────────────────────────────────────────────────────────────
// Phase B — Reopen and query
// ────────────────────────────────────────────────────────────

fn demo(graph: &mut Graph, args: &Args, centroids: &HashMap<&'static str, Vec<f64>>) -> Result<()> {
    let persisted = graph.persisted_size()?;
    println!("  Reopened {} persisted nodes (hot_cache_max={})", fmt_num(persisted), fmt_num(args.hot_cache_max));
    println!("  Cold state: loaded_size={} persisted_size={}", fmt_num(graph.loaded_size()), fmt_num(persisted));

    // ── 1. LRU working set: restore nodes one at a time ──
    section("1. LRU working set — lazy restore via get_node_safe");
    let t0 = Instant::now();
    let mut loaded = 0;
    for i in 0..=2000 {
        if graph.get_node_safe(&uid(i))?.is_some() {
            loaded += 1;
        }
    }
    println!(
        "    loaded_size 0 → {} after probing 2,001 users in {}",
        fmt_num(loaded),
        fmt_ms(t0.elapsed().as_secs_f64() * 1000.0)
    );
    println!("    (nodes are fetched from the backing store, never warmed in bulk)");

    // ── 2. Persisted queries — the full store, without warming it ──
    section("2. Persisted queries over the full backing store");
    let n_users = graph.query_persisted().where_node_type(vec!["user".into()]).count()?;
    let n_docs = graph.query_persisted().where_node_type(vec!["document".into()]).count()?;
    let n_chunks = graph.query_persisted().where_node_type(vec!["chunk".into()]).count()?;
    let n_sections = graph.query_persisted().where_node_type(vec!["section".into()]).count()?;
    println!(
        "    counts: users={} docs={} sections={} chunks={}",
        fmt_num(n_users),
        fmt_num(n_docs),
        fmt_num(n_sections),
        fmt_num(n_chunks)
    );

    let graphs_vec = &centroids["graphs"];
    let tq = Instant::now();
    let top_docs = graph
        .query_persisted()
        .where_node_type(vec!["document".into()])
        .similar_to(graphs_vec.clone(), 0.8, Some(10))
        .to_array()?;
    let mut topics_seen: HashMap<String, usize> = HashMap::new();
    for d in &top_docs {
        let topic = d.data.get("topic").and_then(|v| v.as_str()).unwrap_or("?").to_string();
        *topics_seen.entry(topic).or_insert(0) += 1;
    }
    let mix = topics_seen.iter().map(|(k, v)| format!("{k}×{v}")).collect::<Vec<_>>().join(", ");
    println!(
        "    similar_to('graphs' centroid, threshold 0.8, top 10) in {} — topic mix: {mix}",
        fmt_ms(tq.elapsed().as_secs_f64() * 1000.0)
    );
    if let Some(top) = top_docs.first() {
        let score = top.vector.as_ref().map(|v| cosine(graphs_vec, v).unwrap_or(0.0)).unwrap_or(0.0);
        println!("    top result: {}  (score {:.3})", top.data.get("title").and_then(|v| v.as_str()).unwrap_or("?"), score);
    }

    let t_traverse = Instant::now();
    let subtree = graph
        .query_persisted()
        .where_node_type(vec!["document".into()])
        .where_field("title", json!("doc-21"))
        .traverse("CONTAINS", 2, Direction::Out)
        .to_array()?;
    println!(
        "    traverse: doc-21 → CONTAINS depth 2 = {} nodes ({})",
        fmt_num(subtree.len()),
        fmt_ms(t_traverse.elapsed().as_secs_f64() * 1000.0)
    );

    let page = graph
        .query_persisted()
        .where_node_type(vec!["document".into()])
        .order_by("score", OrderDirection::Desc)
        .limit(5)
        .to_array()?;
    let titles = page.iter().map(|d| d.data.get("title").and_then(|v| v.as_str()).unwrap_or("?")).collect::<Vec<_>>().join(", ");
    println!("    order_by('score', desc).limit(5): {titles}");

    // ── 3. warm() fills the hot working set ──
    section("3. Warm the hot working set");
    let tw = Instant::now();
    graph.warm()?;
    println!(
        "    warm() in {}: loaded_size={} vectors={}",
        fmt_ms(tw.elapsed().as_secs_f64() * 1000.0),
        fmt_num(graph.loaded_size()),
        fmt_num(graph.vectors().size())
    );

    // ── 4. In-memory queries over the hot working set ──
    section("4. In-memory queries over the hot working set");
    let avg_chunk = graph.query().where_node_type(vec!["chunk".into()]).aggregate("score", AggregateOp::Avg);
    println!("    aggregate('score', avg) over {} loaded chunks = {:.2}", fmt_num(avg_chunk.count), avg_chunk.value);

    let quality = graph.query().where_node_type(vec!["chunk".into()]).group_aggregate("score", AggregateOp::Avg, "quality");
    let quality_str = quality.iter().map(|q| format!("{}={:.1} ({})", q.key, q.value, q.count)).collect::<Vec<_>>().join("  ");
    println!("    group_aggregate('score', avg, by 'quality'): {quality_str}");

    // Pull a slice of documents and the user-7 subtree into the hot set.
    let u = 7;
    let d_idx = [u * 3, u * 3 + 1, u * 3 + 2];
    let s_idx: Vec<usize> = d_idx.iter().flat_map(|&d| [d * 3, d * 3 + 1, d * 3 + 2]).collect();
    let c_idx: Vec<usize> = s_idx.iter().flat_map(|&s| [s * 3, s * 3 + 1, s * 3 + 2]).collect();
    let t_restore = Instant::now();
    for i in 0..500 {
        graph.get_node_safe(&did(i))?;
    }
    graph.get_node_safe(&uid(u))?;
    for &di in &d_idx {
        graph.get_node_safe(&did(di))?;
    }
    for &si in &s_idx {
        graph.get_node_safe(&sid(si))?;
    }
    for &ci in &c_idx {
        graph.get_node_safe(&cid(ci))?;
    }
    let restored = 500 + 1 + d_idx.len() + s_idx.len() + c_idx.len();
    println!(
        "    restored {} nodes (500 docs + user-7 subtree) in {}",
        fmt_num(restored),
        fmt_ms(t_restore.elapsed().as_secs_f64() * 1000.0)
    );

    let hot_top = graph.query().where_node_type(vec!["document".into()]).similar_to(graphs_vec.clone(), 0.5, Some(10)).to_array();
    let mut hot_topics: HashMap<String, usize> = HashMap::new();
    for d in &hot_top {
        let topic = d.data.get("topic").and_then(|v| v.as_str()).unwrap_or("?").to_string();
        *hot_topics.entry(topic).or_insert(0) += 1;
    }
    let hot_mix = hot_topics.iter().map(|(k, v)| format!("{k}×{v}")).collect::<Vec<_>>().join(", ");
    let hot_doc_count = graph.query().where_node_type(vec!["document".into()]).to_array().len();
    println!("    similar_to('graphs' centroid) over {} hot docs — topic mix: {hot_mix}", fmt_num(hot_doc_count));

    let cluster_groups: Vec<(String, Vec<f64>)> = TOPICS[..4].iter().map(|t| (t.to_string(), centroids[t].clone())).collect();
    let clusters = graph.query().where_node_type(vec!["document".into()]).group_by_vector(&cluster_groups, "score", AggregateOp::Avg, 0.5);
    let cluster_str = clusters
        .iter()
        .map(|c| format!("{}={:.1} ({})", if c.key == "null" { "other" } else { &c.key }, c.value, c.count))
        .collect::<Vec<_>>()
        .join("  ");
    println!("    group_by_vector (4 centroids, 'score', avg, thr 0.5): {cluster_str}");

    let joined = graph
        .query()
        .where_node_type(vec!["document".into()])
        .join("OWNS", Direction::In, Some(Box::new(|n: &Node| n.data.get("name").and_then(|v| v.as_str()) == Some("user-7"))))
        .to_array();
    println!("    join('OWNS', in, user-7) → {} docs owned by user-7", fmt_num(joined.len()));

    let bfs = graph
        .query()
        .where_field("name", json!("user-7"))
        .traverse("OWNS", 1, Direction::Out)
        .traverse("CONTAINS", 2, Direction::Out)
        .to_array();
    println!("    BFS traversal (user-7 → OWNS → CONTAINS depth 2) → {} nodes", fmt_num(bfs.len()));

    // ── 5. Ownership cascade across persisted data ──
    section("5. Ownership cascade");
    let before = graph.persisted_size()?;
    let before_chunks = graph.query_persisted().where_node_type(vec!["chunk".into()]).count()?;
    let removed = graph.remove_node_safe(&uid(7))?;
    graph.flush()?;
    let after = graph.persisted_size()?;
    let after_chunks = graph.query_persisted().where_node_type(vec!["chunk".into()]).count()?;
    if removed {
        println!("    remove_node_safe('u00007') cascaded its owned subtree (3 docs + 9 sections + 27 chunks)");
        println!("    persisted_size: {} → {}   chunks: {} → {}", fmt_num(before), fmt_num(after), fmt_num(before_chunks), fmt_num(after_chunks));
    } else {
        println!("    u00007 was already removed by a previous run — skipping cascade");
    }

    Ok(())
}

// ────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────

fn store_exists(dir: &Path) -> bool {
    std::fs::read_dir(dir).map(|mut e| e.next().is_some()).unwrap_or(false)
}

fn main() -> Result<()> {
    let args = parse_args();
    println!("polypack-graph — large-db example");
    println!(
        "  store: {}  count={}  hot_cache_max={}  seed={}",
        args.store_dir.display(),
        fmt_num(args.count),
        fmt_num(args.hot_cache_max),
        args.seed
    );

    if args.wipe {
        let _ = std::fs::remove_dir_all(&args.store_dir);
    }
    let centroids = build_centroids(args.seed);

    let needs_seed = args.wipe || !store_exists(&args.store_dir);
    if needs_seed {
        let mut graph = Graph::open(
            Box::new(FsStorage { dir: args.store_dir.clone() }),
            StoreConfig::default(),
            GraphConfig { hot_cache_max: args.hot_cache_max, ..GraphConfig::default() },
        )?;
        section("Phase A — seed");
        seed(&mut graph, &args, &centroids)?;
        graph.dispose()?;
        println!("\n✓ Seeded. Reopening the store…");
    }

    section("Phase B — reopen and query");
    let mut graph = Graph::open(
        Box::new(FsStorage { dir: args.store_dir.clone() }),
        StoreConfig::default(),
        GraphConfig { hot_cache_max: args.hot_cache_max, ..GraphConfig::default() },
    )?;
    demo(&mut graph, &args, &centroids)?;
    graph.dispose()?;

    println!("\n✓ Done. rss={}", rss_mb());
    println!("  Re-run with --wipe to regenerate the store, or --count 100000 for a quick run.");
    Ok(())
}
