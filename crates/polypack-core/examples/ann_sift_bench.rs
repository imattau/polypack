//! ANN-Benchmarks-style comparison: builds Polypack's HnswIndex over the
//! standard `sift-128-euclidean` dataset (http://ann-benchmarks.com) and
//! traces a recall@10 / QPS curve by varying `ef_search` against one build,
//! the same methodology ann-benchmarks itself uses to produce its published
//! per-algorithm curves.
//!
//! Reads flat `<f32; f32>`/`<i32>` binary dumps exported from the dataset's
//! HDF5 file (see benchmarks/ann-sift-report.md for the export script) — not
//! the HDF5 file directly, to avoid adding an hdf5 crate dependency to
//! polypack-core for a one-off benchmark.
//!
//! Usage:
//!   cargo run --release --example ann_sift_bench --manifest-path crates/Cargo.toml -- \
//!     --dir /path/to/exported/files --count 1000000 --queries 1000 \
//!     --ef-search 50,100,200,400,800
use polypack_core::hnsw::{HnswConfig, HnswIndex};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    dir: PathBuf,
    count: usize,
    queries: usize,
    ef_search: Vec<usize>,
    m: usize,
    ef_construction: usize,
    out: PathBuf,
}

fn args() -> Args {
    let raw: Vec<String> = env::args().collect();
    let mut dir = PathBuf::from(".");
    let mut count = usize::MAX;
    let mut queries = usize::MAX;
    let mut ef_search = vec![50, 100, 200, 400, 800];
    let mut m = 16;
    let mut ef_construction = 200;
    let mut out = PathBuf::from("benchmarks/results/ann-sift-rust.json");
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--dir" => { dir = PathBuf::from(&raw[i + 1]); i += 2; }
            "--count" => { count = raw[i + 1].parse().expect("--count must be a number"); i += 2; }
            "--queries" => { queries = raw[i + 1].parse().expect("--queries must be a number"); i += 2; }
            "--ef-search" => { ef_search = raw[i + 1].split(',').map(|s| s.parse().expect("--ef-search must be comma-separated numbers")).collect(); i += 2; }
            "--m" => { m = raw[i + 1].parse().expect("--m must be a number"); i += 2; }
            "--ef-construction" => { ef_construction = raw[i + 1].parse().expect("--ef-construction must be a number"); i += 2; }
            "--out" => { out = PathBuf::from(&raw[i + 1]); i += 2; }
            _ => { i += 1; }
        }
    }
    Args { dir, count, queries, ef_search, m, ef_construction, out }
}

fn read_f32_matrix(path: &std::path::Path, rows: usize, dims: usize) -> Vec<Vec<f64>> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    assert_eq!(bytes.len(), rows * dims * 4, "unexpected file size for {}", path.display());
    bytes
        .chunks_exact(dims * 4)
        .map(|row| (0..dims).map(|i| f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap()) as f64).collect())
        .collect()
}

fn read_i32_matrix(path: &std::path::Path, rows: usize, cols: usize) -> Vec<Vec<i32>> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    assert_eq!(bytes.len(), rows * cols * 4, "unexpected file size for {}", path.display());
    bytes
        .chunks_exact(cols * 4)
        .map(|row| (0..cols).map(|i| i32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap())).collect())
        .collect()
}

fn recall_at_10(got_ids: &[String], ground_truth_row: &[i32]) -> f64 {
    let gt: std::collections::HashSet<i32> = ground_truth_row.iter().copied().collect();
    let hits = got_ids.iter().filter(|id| gt.contains(&id[1..].parse::<i32>().unwrap())).count();
    hits as f64 / ground_truth_row.len() as f64
}

fn main() {
    let a = args();
    let meta = fs::read_to_string(a.dir.join("sift_meta.txt")).expect("sift_meta.txt");
    let mut parts = meta.split_whitespace();
    let full_train: usize = parts.next().unwrap().parse().unwrap();
    let dims: usize = parts.next().unwrap().parse().unwrap();
    let full_test: usize = parts.next().unwrap().parse().unwrap();

    let count = a.count.min(full_train);
    let queries = a.queries.min(full_test);
    println!("dataset: sift-128-euclidean, indexing {count}/{full_train} base vectors ({dims}-dim), {queries} queries");

    let train = read_f32_matrix(&a.dir.join("sift_train.f32bin"), full_train, dims);
    let train = &train[..count];
    let test = read_f32_matrix(&a.dir.join("sift_test.f32bin"), full_test, dims);
    let test = &test[..queries];
    let gt_cosine = read_i32_matrix(&a.dir.join("sift_gt_cosine.i32bin"), full_test, 10);
    let gt_euclidean = read_i32_matrix(&a.dir.join("sift_gt_euclidean.i32bin"), full_test, 10);

    let config = HnswConfig { m: a.m, mmax0: a.m * 2, ef_construction: a.ef_construction, ef_search: a.ef_construction, ..Default::default() };
    let mut index = HnswIndex::new(config, 42).expect("valid config");

    let build_start = Instant::now();
    for (i, vector) in train.iter().enumerate() {
        index.add(&format!("n{i}"), vector).expect("add");
        if (i + 1) % 100_000 == 0 { println!("  built {}/{count}", i + 1); }
    }
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    println!("build: {build_ms:.1}ms ({:.0} vectors/sec)", count as f64 / (build_ms / 1000.0));

    let mut cases = Vec::new();
    for &ef in &a.ef_search {
        let query_start = Instant::now();
        let mut cosine_recalls = Vec::with_capacity(queries);
        let mut euclidean_recalls = Vec::with_capacity(queries);
        for (i, q) in test.iter().enumerate() {
            // -1.0 is below any possible cosine score (range [-1, 1]) — accept everything.
            let got = index.query_with_ef_search(q, 10, -1.0, ef).expect("query");
            let ids: Vec<String> = got.into_iter().map(|s| s.id).collect();
            cosine_recalls.push(recall_at_10(&ids, &gt_cosine[i]));
            euclidean_recalls.push(recall_at_10(&ids, &gt_euclidean[i]));
        }
        let query_ms = query_start.elapsed().as_secs_f64() * 1000.0;
        let qps = queries as f64 / (query_ms / 1000.0);
        let mean_cosine_recall = cosine_recalls.iter().sum::<f64>() / queries as f64;
        let mean_euclidean_recall = euclidean_recalls.iter().sum::<f64>() / queries as f64;
        println!(
            "ef_search={ef:<5} qps={qps:>9.1}  recall@10(cosine gt)={mean_cosine_recall:.4}  recall@10(euclidean gt)={mean_euclidean_recall:.4}"
        );
        cases.push(json!({
            "efSearch": ef,
            "qps": qps,
            "queryMs": query_ms,
            "recallAt10Cosine": mean_cosine_recall,
            "recallAt10Euclidean": mean_euclidean_recall,
        }));
    }

    let result = json!({
        "schemaVersion": 1,
        "dataset": "sift-128-euclidean",
        "fullTrainCount": full_train,
        "indexedCount": count,
        "queries": queries,
        "dims": dims,
        "m": a.m,
        "efConstruction": a.ef_construction,
        "buildMs": build_ms,
        "buildVectorsPerSec": count as f64 / (build_ms / 1000.0),
        "cases": cases,
    });
    if let Some(parent) = a.out.parent() { fs::create_dir_all(parent).unwrap(); }
    fs::write(&a.out, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!("Wrote {}", a.out.display());
}
