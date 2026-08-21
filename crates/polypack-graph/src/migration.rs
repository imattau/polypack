//! Application record migrations for the Rust graph API.

use polypack_core::{Edge, Node, PolypackError, Result};

/// One contiguous application-schema migration step.
pub struct MigrationDefinition {
    pub from: u32,
    pub to: u32,
    pub migrate_node: Box<dyn Fn(Node) -> Result<Node> + Send + Sync>,
    pub migrate_edge: Option<Box<dyn Fn(Edge) -> Result<Edge> + Send + Sync>>,
}

impl MigrationDefinition {
    pub fn new(
        from: u32,
        to: u32,
        migrate_node: impl Fn(Node) -> Result<Node> + Send + Sync + 'static,
    ) -> Self {
        Self { from, to, migrate_node: Box::new(migrate_node), migrate_edge: None }
    }

    pub fn with_edge_migration(
        mut self,
        migrate_edge: impl Fn(Edge) -> Result<Edge> + Send + Sync + 'static,
    ) -> Self {
        self.migrate_edge = Some(Box::new(migrate_edge));
        self
    }
}

/// Controls a migration run. Records are processed in ID order.
pub struct MigrationOptions {
    pub dry_run: bool,
    pub batch_size: usize,
    pub resume_after_node: Option<String>,
    pub resume_after_edge: Option<String>,
    pub on_progress: Option<Box<dyn Fn(MigrationProgress) + Send + Sync>>,
}

impl Default for MigrationOptions {
    fn default() -> Self {
        Self { dry_run: false, batch_size: 1000, resume_after_node: None, resume_after_edge: None, on_progress: None }
    }
}

/// Migration counters suitable for progress reporting by callers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MigrationReport {
    pub from: u32,
    pub to: u32,
    pub processed_nodes: usize,
    pub processed_edges: usize,
    pub migrated_nodes: usize,
    pub migrated_edges: usize,
    pub dry_run: bool,
}

pub type MigrationProgress = MigrationReport;

/// Registry of contiguous application-schema migration steps.
#[derive(Default)]
pub struct MigrationRegistry {
    definitions: Vec<MigrationDefinition>,
}

impl MigrationRegistry {
    pub fn register(&mut self, definition: MigrationDefinition) -> Result<()> {
        if definition.to <= definition.from {
            return Err(PolypackError::InvalidArgument("migration target must be greater than source".into()));
        }
        if self.definitions.iter().any(|existing| existing.from == definition.from) {
            return Err(PolypackError::InvalidArgument(format!("migration from version {} is already registered", definition.from)));
        }
        self.definitions.push(definition);
        Ok(())
    }

    pub fn is_empty(&self) -> bool { self.definitions.is_empty() }

    fn path(&self, from: u32, to: u32) -> Result<Vec<&MigrationDefinition>> {
        if from == to { return Ok(Vec::new()); }
        let mut current = from;
        let mut path = Vec::new();
        while current < to {
            let step = self.definitions.iter().find(|definition| definition.from == current).ok_or_else(|| {
                PolypackError::InvalidArgument(format!("no migration registered from version {current}"))
            })?;
            if step.to > to {
                return Err(PolypackError::InvalidArgument(format!("migration from {current} overshoots target version {to}")));
            }
            current = step.to;
            path.push(step);
        }
        Ok(path)
    }

    pub(crate) fn migrate_node(&self, node: Node, from: u32, to: u32) -> Result<Node> {
        let mut node = node;
        for definition in self.path(from, to)? { node = (definition.migrate_node)(node)?; }
        Ok(node)
    }

    pub(crate) fn migrate_edge(&self, edge: Edge, from: u32, to: u32) -> Result<Edge> {
        let mut edge = edge;
        for definition in self.path(from, to)? {
            if let Some(migrate_edge) = &definition.migrate_edge { edge = migrate_edge(edge)?; }
        }
        Ok(edge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str) -> Node {
        Node { id: id.into(), node_type: "record".into(), data: Default::default(), vector: None, inserted_at: 0, updated_at: 0, revision: 0, activation: None }
    }

    #[test]
    fn requires_contiguous_steps() {
        let mut registry = MigrationRegistry::default();
        registry.register(MigrationDefinition::new(1, 2, |mut node| { node.data.insert("v".into(), 2.into()); Ok(node) })).unwrap();
        assert!(registry.migrate_node(node("a"), 1, 3).is_err());
    }

    #[test]
    fn applies_steps_in_order() {
        let mut registry = MigrationRegistry::default();
        registry.register(MigrationDefinition::new(1, 2, |mut node| { node.data.insert("v".into(), 2.into()); Ok(node) })).unwrap();
        registry.register(MigrationDefinition::new(2, 3, |mut node| { node.data.insert("done".into(), true.into()); Ok(node) })).unwrap();
        let migrated = registry.migrate_node(node("a"), 1, 3).unwrap();
        assert_eq!(migrated.data.get("v").and_then(|value| value.as_i64()), Some(2));
        assert_eq!(migrated.data.get("done").and_then(|value| value.as_bool()), Some(true));
    }
}
