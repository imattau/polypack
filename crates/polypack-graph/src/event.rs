//! Change notifications, mirrors `GraphChangeEvent` in `src/types.ts`.
//!
//! `graph.ts` publishes these on an rxjs `Subject`, coalesced by
//! `startBatch`/`endBatch`. Rust has no equivalent hot-observable primitive
//! baked into the language, so `Graph` takes a plain callback (see
//! `Graph::on_change`) — swap for a `tokio::sync::broadcast` channel if
//! multi-consumer fanout is needed later.

#[derive(Clone, Debug, PartialEq)]
pub enum GraphChangeEvent {
    NodeAdded { node_id: Option<String>, node_type: String },
    NodeUpdated { node_id: String, node_type: String },
    NodeRemoved { node_id: String, node_type: String },
    EdgeAdded { edge_id: String, edge_type: String, source: String, target: String },
    EdgeUpdated { edge_id: String, edge_type: String, source: String, target: String },
    EdgeRemoved { edge_type: String, source: String, target: String },
    ActivationUpdated { node_id: String, node_type: String, delta: f64, reason: Option<String>, context: Option<String> },
    InhibitionUpdated { node_id: String, node_type: String, delta: f64, reason: Option<String> },
}
