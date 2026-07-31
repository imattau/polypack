//! Declarative query envelopes matching `specification/query-plan.schema.json`.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct QueryPlan {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_types: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Vec<AttributeFilter>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_filter: Option<EdgeFilter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub traversal: Option<Vec<TraversalStep>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joins: Option<Vec<Join>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity: Option<Similarity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Order>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase", tag = "operator")]
pub enum AttributeFilter {
    Eq { field: String, value: serde_json::Value },
    Range { field: String, above: Option<f64>, below: Option<f64> },
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EdgeFilter {
    #[serde(rename = "type")]
    pub edge_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TraversalStep {
    pub edge_type: String,
    pub direction: Direction,
    pub depth: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Join {
    pub edge_type: String,
    pub direction: Direction,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Out,
    In,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Similarity {
    pub vector: Vec<f64>,
    pub threshold: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Order {
    pub field: String,
    pub direction: OrderDirection,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OrderDirection {
    Asc,
    Desc,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct QueryResult {
    pub ids: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serde_round_trip_matches_schema_shape() {
        let plan = QueryPlan {
            node_types: Some(vec!["document".into()]),
            attributes: Some(vec![
                AttributeFilter::Eq { field: "category".into(), value: json!("science") },
                AttributeFilter::Range { field: "score".into(), above: Some(0.7), below: None },
            ]),
            traversal: Some(vec![TraversalStep {
                edge_type: "REFERENCES".into(),
                direction: Direction::Out,
                depth: 2,
            }]),
            similarity: Some(Similarity { vector: vec![0.1, 0.2], threshold: 0.5, top_k: Some(20) }),
            order: Some(Order { field: "updatedAt".into(), direction: OrderDirection::Desc }),
            offset: Some(0),
            limit: Some(20),
            ..Default::default()
        };
        let s = serde_json::to_string(&plan).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed["nodeTypes"][0], json!("document"));
        assert_eq!(parsed["attributes"][0]["operator"], json!("eq"));
        assert_eq!(parsed["traversal"][0]["direction"], json!("out"));
        assert_eq!(parsed["order"]["direction"], json!("desc"));
        assert_eq!(parsed["similarity"]["topK"], json!(20));
        let back: QueryPlan = serde_json::from_str(&s).unwrap();
        assert_eq!(plan, back);
    }
}
