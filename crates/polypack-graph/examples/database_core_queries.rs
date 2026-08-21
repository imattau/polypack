//! Cross-binding read benchmark — Rust lane.

use polypack_core::{DistanceFn, ExactIndex, HnswConfig, InMemoryStorage, Node, StoreConfig};
use polypack_graph::{Graph, GraphConfig, OrderDirection};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const DIMS: usize = 32;
const TOP_K: usize = 10;

struct Args { count: usize, iterations: usize, out: PathBuf }
fn args() -> Args {
    let raw: Vec<String> = env::args().collect(); let mut count = 10_000; let mut iterations = 20; let mut out = PathBuf::from("benchmarks/results/database-core-queries-rust.json"); let mut i = 1;
    while i < raw.len() { match raw[i].as_str() { "--count" => { count = raw[i+1].parse().unwrap(); i += 2; }, "--iterations" => { iterations = raw[i+1].parse().unwrap(); i += 2; }, "--out" => { out = raw[i+1].clone().into(); i += 2; }, _ => i += 1 } }
    Args { count, iterations, out }
}
struct Rng(u32);
impl Rng { fn next(&mut self) -> f64 { self.0 = self.0.wrapping_add(0x6d2b79f5); let mut t = (self.0 ^ (self.0 >> 15)).wrapping_mul(self.0 | 1); t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(self.0 | 61))) ^ t; ((t ^ (t >> 14)) as f64) / 4294967296.0 } }
fn vectors(count: usize, seed: u32) -> Vec<Vec<f64>> { let mut rng = Rng(seed); (0..count).map(|_| (0..DIMS).map(|_| rng.next() * 2.0 - 1.0).collect()).collect() }
fn node(i: usize, value: Vec<f64>) -> Node { Node { id: format!("n{i}"), node_type: ["user", "post", "comment"][i % 3].into(), data: { let mut d = serde_json::Map::new(); d.insert("score".into(), json!(i % 1000)); d.insert("bucket".into(), json!(i % 10)); d.insert("value".into(), json!(i)); d }, vector: Some(value), inserted_at: i as i64, updated_at: i as i64, revision: 0, activation: None, ..Default::default() } }
fn summary(mut times: Vec<f64>) -> serde_json::Value { times.sort_by(|a,b| a.partial_cmp(b).unwrap()); let at = |p: f64| times[((times.len() as f64 * p).ceil() as usize).saturating_sub(1).min(times.len()-1)]; json!({"p50Ms": at(0.5), "p95Ms": at(0.95), "p99Ms": at(0.99)}) }
fn main() {
    let a = args(); assert!(a.count > 0); let data = vectors(a.count, 42); let query = vectors(1, 43).remove(0); let config = GraphConfig { hnsw: HnswConfig { ef_search: 300, ..Default::default() }, ..Default::default() }; let mut graph = Graph::open(Box::new(InMemoryStorage::new()), StoreConfig::default(), config).unwrap();
    graph.add_nodes(data.iter().enumerate().map(|(i, value)| node(i, value.clone())).collect()).unwrap(); graph.flush().unwrap();
    let expected = std::cmp::min(25, if a.count <= 22 { 0 } else { (a.count - 23) / 30 + 1 });
    let mut times = Vec::new(); let mut query_count = 0;
    for _ in 0..a.iterations { let t=Instant::now(); query_count = graph.query().where_node_type(vec!["post".into()]).where_field("bucket", json!(2)).order_by("score", OrderDirection::Desc).limit(25).ids().len(); times.push(t.elapsed().as_secs_f64()*1000.0); }
    assert_eq!(query_count, expected);
    let q = query; let mut exact = ExactIndex::new(DistanceFn::Cosine); for (i, value) in data.iter().enumerate() { exact.add(&format!("n{i}"), value).unwrap(); }
    let mut exact_times=Vec::new(); let mut exact_count=0; for _ in 0..a.iterations { let t=Instant::now(); exact_count=exact.query(&q, TOP_K, 0.0).unwrap().len(); exact_times.push(t.elapsed().as_secs_f64()*1000.0); } assert_eq!(exact_count, TOP_K);
    let mut hnsw_times=Vec::new(); let mut hnsw_count=0; for _ in 0..a.iterations { let t=Instant::now(); hnsw_count=graph.similar_to(&q, 0.0, Some(TOP_K)).unwrap().len(); hnsw_times.push(t.elapsed().as_secs_f64()*1000.0); } assert_eq!(hnsw_count, TOP_K);
    let exact_ids: std::collections::HashSet<_> = exact.query(&q, TOP_K, 0.0).unwrap().into_iter().map(|x| x.id).collect(); let hnsw_ids = graph.similar_to(&q, 0.0, Some(TOP_K)).unwrap(); let recall = hnsw_ids.iter().filter(|x| exact_ids.contains(&x.0)).count() as f64 / TOP_K as f64;
    let mut graph_query = summary(times); graph_query["resultCount"] = json!(query_count);
    let mut exact_result = summary(exact_times); exact_result["resultCount"] = json!(TOP_K);
    let mut hnsw_result = summary(hnsw_times); hnsw_result["resultCount"] = json!(TOP_K); hnsw_result["recallAtK"] = json!(recall);
    let result=json!({"schemaVersion":1,"lang":"rust","count":a.count,"dimensions":DIMS,"iterations":a.iterations,"topK":TOP_K,"dataSeed":42,"querySeed":43,"graphQuery": graph_query,"hotQuery": {"resultCount":query_count},"exactVector": exact_result,"hnswVector": hnsw_result});
    if let Some(parent)=a.out.parent(){fs::create_dir_all(parent).unwrap();} fs::write(&a.out, serde_json::to_string_pretty(&result).unwrap()).unwrap(); println!("{}",serde_json::to_string_pretty(&result).unwrap()); println!("Wrote {}",a.out.display());
}
