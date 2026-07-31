//! Cross-language comparison benchmark.
//!
//! Mirrors `benchmarks/run-ts.ts`: a fresh `mulberry32(42)` per case, values
//! consumed id-major then dim-minor, and the same JSON result shape. Build
//! time, query-latency percentiles, and Recall@10 are reported per case.
//!
//! Usage:
//!   cargo run --release --bin compare -- [--case <name>] [--out <path>]
//! With no `--case`, every case runs in one process (memory is then cumulative
//! peak). Pass a single case for a fair per-case peak-RSS comparison.

use polypack_core::hnsw::{HnswConfig, HnswIndex};
use polypack_core::rng::Mulberry32;
use polypack_core::vector::{ExactIndex, DistanceFn};
use serde::Serialize;
use std::collections::HashSet;
use std::io::Write;
use std::time::Instant;

const TOP_K: usize = 10;

#[derive(Serialize, Clone)]
struct CaseResult {
    name: String,
    index: String,
    count: usize,
    dims: usize,
    build_ms: f64,
    query_count: usize,
    avg_ms: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    recall10: f64,
    heap_used_mb: f64,
    max_rss_mb: f64,
}

struct CaseSpec {
    index: &'static str,
    count: usize,
    dims: usize,
    queries: usize,
}

fn cases() -> Vec<CaseSpec> {
    vec![
        CaseSpec { index: "exact", count: 10_000, dims: 8, queries: 1000 },
        CaseSpec { index: "exact", count: 100_000, dims: 8, queries: 1000 },
        CaseSpec { index: "exact", count: 500_000, dims: 8, queries: 500 },
        CaseSpec { index: "exact", count: 10_000, dims: 384, queries: 500 },
        CaseSpec { index: "exact", count: 100_000, dims: 384, queries: 200 },
        CaseSpec { index: "hnsw", count: 10_000, dims: 8, queries: 1000 },
        CaseSpec { index: "hnsw", count: 100_000, dims: 8, queries: 1000 },
        CaseSpec { index: "hnsw", count: 10_000, dims: 384, queries: 500 },
    ]
}

fn case_name(c: &CaseSpec) -> String {
    format!("{}-{}-{}", c.index, c.count, c.dims)
}

fn generate(count: usize, dims: usize) -> Vec<Vec<f64>> {
    let mut rng = Mulberry32::new(42);
    (0..count)
        .map(|_| (0..dims).map(|_| rng.next_f64() * 2.0 - 1.0).collect())
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * sorted.len() as f64).ceil() as usize - 1;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

/// Peak resident set size (VmHWM) and current RSS in MB from /proc when
/// available; 0 otherwise.
fn memory_mb() -> (f64, f64) {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return (0.0, 0.0);
    };
    let mut peak_kb = 0u64;
    let mut rss_kb = 0u64;
    for line in status.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            peak_kb = parse_kb(v);
        } else if let Some(v) = line.strip_prefix("VmRSS:") {
            rss_kb = parse_kb(v);
        }
    }
    (rss_kb as f64 / 1024.0, peak_kb as f64 / 1024.0)
}

fn parse_kb(v: &str) -> u64 {
    v.split_whitespace().next().and_then(|n| n.parse().ok()).unwrap_or(0)
}

fn run_case(spec: &CaseSpec) -> CaseResult {
    let data = generate(spec.count, spec.dims);
    let queries = generate(spec.queries, spec.dims);

    let exact_build = Instant::now();
    let mut exact = ExactIndex::new(DistanceFn::Cosine);
    for (i, v) in data.iter().enumerate() {
        exact.add(&format!("v{i}"), v).unwrap();
    }
    let exact_build_ms = exact_build.elapsed().as_secs_f64() * 1000.0;

    let hnsw_build = Instant::now();
    let mut hnsw = HnswIndex::new(HnswConfig { ef_construction: 200, ef_search: 300, ..Default::default() }, 7);
    if spec.index == "hnsw" {
        for (i, v) in data.iter().enumerate() {
            hnsw.add(&format!("v{i}"), v).unwrap();
        }
    }
    let hnsw_build_ms = hnsw_build.elapsed().as_secs_f64() * 1000.0;

    if spec.index == "exact" {
        let mut lat = Vec::with_capacity(spec.queries);
        for q in &queries {
            let t = Instant::now();
            exact.query(q, TOP_K, 0.0).unwrap();
            lat.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mut sorted = lat.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (rss, peak) = memory_mb();
        return CaseResult {
            name: case_name(spec),
            index: "exact".into(),
            count: spec.count,
            dims: spec.dims,
            build_ms: exact_build_ms,
            query_count: spec.queries,
            avg_ms: lat.iter().sum::<f64>() / lat.len() as f64,
            p50: percentile(&sorted, 50.0),
            p95: percentile(&sorted, 95.0),
            p99: percentile(&sorted, 99.0),
            recall10: f64::NAN,
            heap_used_mb: rss,
            max_rss_mb: peak,
        };
    }

    let mut lat = Vec::with_capacity(spec.queries);
    let mut hits = 0usize;
    for q in &queries {
        let exact_ids: HashSet<String> = exact
            .query(q, TOP_K, 0.0)
            .unwrap()
            .into_iter()
            .map(|s| s.id)
            .collect();
        let t = Instant::now();
        let ann = hnsw.query(q, TOP_K, 0.0).unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
        hits += ann.iter().filter(|s| exact_ids.contains(&s.id)).count();
    }
    let mut sorted = lat.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (rss, peak) = memory_mb();
    CaseResult {
        name: case_name(spec),
        index: "hnsw".into(),
        count: spec.count,
        dims: spec.dims,
        build_ms: hnsw_build_ms,
        query_count: spec.queries,
        avg_ms: lat.iter().sum::<f64>() / lat.len() as f64,
        p50: percentile(&sorted, 50.0),
        p95: percentile(&sorted, 95.0),
        p99: percentile(&sorted, 99.0),
        recall10: hits as f64 / (spec.queries * TOP_K) as f64,
        heap_used_mb: rss,
        max_rss_mb: peak,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case_arg = args
        .windows(2)
        .find(|w| w[0] == "--case")
        .map(|w| w[1].clone());
    let out_arg = args
        .windows(2)
        .find(|w| w[0] == "--out")
        .map(|w| w[1].clone());

    let all = cases();
    let selected: Vec<&CaseSpec> = match &case_arg {
        Some(name) => {
            let found: Vec<&CaseSpec> = all.iter().filter(|c| case_name(c) == *name).collect();
            if found.is_empty() {
                eprintln!("unknown case {name}; known: {}", all.iter().map(case_name).collect::<Vec<_>>().join(", "));
                std::process::exit(1);
            }
            found
        }
        None => all.iter().collect(),
    };

    let mut results = Vec::with_capacity(selected.len());
    for spec in selected {
        eprintln!("{} ...", case_name(spec));
        let r = run_case(spec);
        eprintln!(
            "  build={:.0}ms recall@10={:.1}% p50={:.3}ms p95={:.3}ms p99={:.3}ms rss={:.0}MB peak={:.0}MB",
            r.build_ms,
            r.recall10 * 100.0,
            r.p50,
            r.p95,
            r.p99,
            r.heap_used_mb,
            r.max_rss_mb
        );
        results.push(r);
    }

    let payload = serde_json::json!({
        "engine": "rust",
        "hnswConfig": { "M": 16, "efConstruction": 200, "efSearch": 300 },
        "topK": TOP_K,
        "memoryNote": "heapUsedMB is VmRSS; maxRssMB is VmHWM (peak RSS)",
        "results": results,
    });
    let json = serde_json::to_string_pretty(&payload).unwrap();
    match out_arg {
        Some(path) => {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(json.as_bytes()).unwrap();
        }
        None => println!("{json}"),
    }
}
