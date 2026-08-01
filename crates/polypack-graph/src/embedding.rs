//! Pluggable text embedding, the Rust counterpart to `src/embedding.ts`.
//!
//! `EmbeddingProvider` is a trait rather than a `Graph` field — this crate
//! has no async runtime dependency, so `embed` is synchronous (TS allows a
//! provider to return a `Promise`; a hosted/API-backed Rust provider would
//! need its own blocking wrapper or an async variant of this trait, neither
//! of which exists here yet). Once a provider produces a vector, nothing
//! downstream (`Graph::add_node`, `HnswIndex`, `GraphQuery::similar_to`)
//! cares how it was produced — this module only turns text into a `Vec<f64>`.

use std::collections::HashMap;

use polypack_core::{PolypackError, Result};

/// Pluggable text embedding contract for local models, APIs, or custom
/// logic. Mirrors `EmbeddingProvider` in `src/embedding.ts`.
pub trait EmbeddingProvider {
    /// Expected output dimensionality, if known. `create_embedding` uses
    /// this to validate a provider's output.
    fn dimensions(&self) -> Option<usize> {
        None
    }

    fn embed(&self, text: &str) -> Vec<f64>;
}

/// Configuration for [`FeatureHashEmbedding`].
#[derive(Clone, Copy, Debug)]
pub struct FeatureHashEmbeddingOptions {
    pub dimensions: usize,
}

impl Default for FeatureHashEmbeddingOptions {
    fn default() -> Self {
        Self { dimensions: 384 }
    }
}

/// Dependency-free lexical embedding based on hashed word frequencies.
/// Deterministic and normalized for cosine similarity; requires no model
/// download. Applications needing semantic meaning should supply a
/// different [`EmbeddingProvider`]. Mirrors `FeatureHashEmbedding`.
#[derive(Clone, Copy, Debug)]
pub struct FeatureHashEmbedding {
    dimensions: usize,
}

impl FeatureHashEmbedding {
    pub fn new(options: FeatureHashEmbeddingOptions) -> Result<Self> {
        if options.dimensions == 0 {
            return Err(PolypackError::InvalidArgument(
                "embedding dimensions must be a positive integer".into(),
            ));
        }
        Ok(Self { dimensions: options.dimensions })
    }
}

impl Default for FeatureHashEmbedding {
    fn default() -> Self {
        Self::new(FeatureHashEmbeddingOptions::default()).expect("default dimensions are positive")
    }
}

impl EmbeddingProvider for FeatureHashEmbedding {
    fn dimensions(&self) -> Option<usize> {
        Some(self.dimensions)
    }

    fn embed(&self, text: &str) -> Vec<f64> {
        let mut vector = vec![0.0f64; self.dimensions];

        // Splits on runs of non-letter/non-number characters, mirroring the
        // TS `/[^\p{L}\p{N}]+/u` split; `char::is_alphanumeric` covers the
        // same Unicode Letter/Number categories.
        for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
            // Two independently-seeded 32-bit hashes mixed together — a
            // feature-hashing trick. Hashes UTF-16 code units (not `char`s)
            // to bit-for-bit match the TS version's `charCodeAt` iteration,
            // including surrogate-pair splitting for astral-plane text.
            let mut first: u32 = 0x811c_9dc5;
            let mut second: u32 = 0x6b8b_4567;
            for unit in word.encode_utf16() {
                let code = unit as u32;
                first = (first ^ code).wrapping_mul(0x0100_0193);
                second = (second ^ code).wrapping_mul(0x5bd1_e995);
            }
            let bucket = ((first ^ second) as usize) % self.dimensions;
            vector[bucket] += 1.0;
        }

        let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        vector
    }
}

/// Validate a provider's raw output: non-empty, all-finite, and matching
/// its declared dimensionality if any. Mirrors `createEmbedding` — the TS
/// version's `typeof text !== 'string'` check is redundant here since `text`
/// is already a `&str`.
pub fn create_embedding(provider: &dyn EmbeddingProvider, text: &str) -> Result<Vec<f64>> {
    let vector = provider.embed(text);
    if vector.is_empty() {
        return Err(PolypackError::InvalidArgument(
            "embedding provider must return at least one dimension".into(),
        ));
    }
    if !vector.iter().all(|v| v.is_finite()) {
        return Err(PolypackError::InvalidArgument("embedding must contain finite values".into()));
    }
    if let Some(expected) = provider.dimensions() {
        if vector.len() != expected {
            return Err(PolypackError::InvalidArgument(format!(
                "embedding provider returned {} dimensions; expected {expected}",
                vector.len()
            )));
        }
    }
    Ok(vector)
}

/// Build weighted embedding text by repeating fields according to their
/// weight (default 1), so a bag-of-words embedding treats heavier fields as
/// more significant. `fields` is a slice rather than a map to preserve
/// field order, matching iteration order of a JS object in the TS version.
///
/// ```
/// # use polypack_graph::build_embedding_text;
/// # use std::collections::HashMap;
/// let mut weights = HashMap::new();
/// weights.insert("subject", 3);
/// let text = build_embedding_text(&[("subject", "Hello"), ("content", "World")], Some(&weights));
/// assert_eq!(text, "Hello Hello Hello World");
/// ```
pub fn build_embedding_text(fields: &[(&str, &str)], weights: Option<&HashMap<&str, usize>>) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for (key, value) in fields {
        if value.is_empty() {
            continue;
        }
        let weight = weights.and_then(|w| w.get(key)).copied().unwrap_or(1);
        for _ in 0..weight {
            parts.push(value);
        }
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_hash_embedding_is_deterministic() {
        let e = FeatureHashEmbedding::default();
        assert_eq!(e.embed("hello world"), e.embed("hello world"));
    }

    #[test]
    fn feature_hash_embedding_is_normalized() {
        let e = FeatureHashEmbedding::default();
        let v = e.embed("hello world foo bar baz");
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn feature_hash_embedding_of_empty_text_is_the_zero_vector() {
        let e = FeatureHashEmbedding::default();
        let v = e.embed("   ");
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn feature_hash_embedding_ignores_case_and_punctuation() {
        let e = FeatureHashEmbedding::default();
        assert_eq!(e.embed("Hello, World!"), e.embed("hello world"));
    }

    #[test]
    fn feature_hash_embedding_has_the_configured_dimensions() {
        let e = FeatureHashEmbedding::new(FeatureHashEmbeddingOptions { dimensions: 16 }).unwrap();
        assert_eq!(e.embed("anything").len(), 16);
        assert_eq!(e.dimensions(), Some(16));
    }

    #[test]
    fn feature_hash_embedding_rejects_zero_dimensions() {
        assert!(FeatureHashEmbedding::new(FeatureHashEmbeddingOptions { dimensions: 0 }).is_err());
    }

    #[test]
    fn feature_hash_embedding_handles_astral_plane_text() {
        // A surrogate pair (UTF-16 code units), exercised to confirm the
        // per-code-unit hashing loop doesn't panic or misbehave.
        let e = FeatureHashEmbedding::default();
        let v = e.embed("hello 😀 world");
        assert!(v.iter().any(|x| *x != 0.0));
    }

    #[test]
    fn create_embedding_rejects_a_non_finite_vector() {
        struct BadProvider;
        impl EmbeddingProvider for BadProvider {
            fn embed(&self, _text: &str) -> Vec<f64> {
                vec![1.0, f64::NAN]
            }
        }
        assert!(create_embedding(&BadProvider, "x").is_err());
    }

    #[test]
    fn create_embedding_rejects_an_empty_vector() {
        struct EmptyProvider;
        impl EmbeddingProvider for EmptyProvider {
            fn embed(&self, _text: &str) -> Vec<f64> {
                Vec::new()
            }
        }
        assert!(create_embedding(&EmptyProvider, "x").is_err());
    }

    #[test]
    fn create_embedding_rejects_a_dimension_mismatch() {
        struct MismatchedProvider;
        impl EmbeddingProvider for MismatchedProvider {
            fn dimensions(&self) -> Option<usize> {
                Some(10)
            }
            fn embed(&self, _text: &str) -> Vec<f64> {
                vec![1.0, 2.0]
            }
        }
        assert!(create_embedding(&MismatchedProvider, "x").is_err());
    }

    #[test]
    fn create_embedding_accepts_a_valid_feature_hash_vector() {
        let e = FeatureHashEmbedding::default();
        assert!(create_embedding(&e, "hello world").is_ok());
    }

    #[test]
    fn build_embedding_text_joins_fields_with_default_weight() {
        let text = build_embedding_text(&[("subject", "Hello"), ("content", "World")], None);
        assert_eq!(text, "Hello World");
    }

    #[test]
    fn build_embedding_text_repeats_weighted_fields() {
        let mut weights = HashMap::new();
        weights.insert("subject", 3);
        let text = build_embedding_text(&[("subject", "Hello"), ("content", "World")], Some(&weights));
        assert_eq!(text, "Hello Hello Hello World");
    }

    #[test]
    fn build_embedding_text_skips_empty_fields() {
        let text = build_embedding_text(&[("a", "x"), ("b", ""), ("c", "y")], None);
        assert_eq!(text, "x y");
    }
}
