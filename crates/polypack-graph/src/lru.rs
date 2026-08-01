//! An O(1) least-recently-used order, used by [`crate::Graph`]'s hot cache.
//!
//! `graph.ts` gets this for free from a JS `Map` (`delete` + `set` reorders a
//! key to the end in O(1)). The earlier Rust port used a `Vec<String>` with
//! `retain`/`remove(0)`, which is O(n) per touch/evict. This is an intrusive
//! doubly-linked list over a slab (`Vec<Option<Node>>`) plus a `HashMap` index,
//! giving O(1) `touch`/`remove`/`pop_front`.

use std::collections::HashMap;

struct Node {
    id: String,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Least-recently-used order over a set of string ids. The front is the
/// least-recently-used end (next to evict); the back is the
/// most-recently-used end.
#[derive(Default)]
pub(crate) struct LruList {
    slots: Vec<Option<Node>>,
    free: Vec<usize>,
    index: HashMap<String, usize>,
    front: Option<usize>,
    back: Option<usize>,
}

impl LruList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    fn unlink(&mut self, slot: usize) {
        let (prev, next) = {
            let node = self.slots[slot].as_ref().expect("unlink on a live slot");
            (node.prev, node.next)
        };
        match prev {
            Some(p) => self.slots[p].as_mut().unwrap().next = next,
            None => self.front = next,
        }
        match next {
            Some(n) => self.slots[n].as_mut().unwrap().prev = prev,
            None => self.back = prev,
        }
    }

    fn push_back(&mut self, slot: usize) {
        let old_back = self.back;
        {
            let node = self.slots[slot].as_mut().expect("push_back on a live slot");
            node.prev = old_back;
            node.next = None;
        }
        match old_back {
            Some(b) => self.slots[b].as_mut().unwrap().next = Some(slot),
            None => self.front = Some(slot),
        }
        self.back = Some(slot);
    }

    fn free_slot(&mut self, slot: usize) {
        self.slots[slot] = None;
        self.free.push(slot);
    }

    /// Move `id` to the most-recently-used end, inserting it if it isn't
    /// already tracked. O(1).
    pub fn touch(&mut self, id: &str) {
        if let Some(&slot) = self.index.get(id) {
            self.unlink(slot);
            self.push_back(slot);
            return;
        }
        let slot = match self.free.pop() {
            Some(slot) => {
                self.slots[slot] = Some(Node { id: id.to_string(), prev: None, next: None });
                slot
            }
            None => {
                self.slots.push(Some(Node { id: id.to_string(), prev: None, next: None }));
                self.slots.len() - 1
            }
        };
        self.index.insert(id.to_string(), slot);
        self.push_back(slot);
    }

    /// Remove `id` from wherever it is in the order, if present. O(1).
    pub fn remove(&mut self, id: &str) {
        if let Some(slot) = self.index.remove(id) {
            self.unlink(slot);
            self.free_slot(slot);
        }
    }

    /// Remove and return the least-recently-used id, if any. O(1).
    pub fn pop_front(&mut self) -> Option<String> {
        let slot = self.front?;
        let id = self.slots[slot].as_ref().unwrap().id.clone();
        self.unlink(slot);
        self.index.remove(&id);
        self.free_slot(slot);
        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_appends_new_ids_in_order() {
        let mut lru = LruList::new();
        lru.touch("a");
        lru.touch("b");
        lru.touch("c");

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.pop_front(), Some("a".to_string()));
        assert_eq!(lru.pop_front(), Some("b".to_string()));
        assert_eq!(lru.pop_front(), Some("c".to_string()));
        assert_eq!(lru.pop_front(), None);
    }

    #[test]
    fn touching_an_existing_id_moves_it_to_the_back() {
        let mut lru = LruList::new();
        lru.touch("a");
        lru.touch("b");
        lru.touch("c");
        lru.touch("a"); // a is now most-recently-used

        assert_eq!(lru.pop_front(), Some("b".to_string()));
        assert_eq!(lru.pop_front(), Some("c".to_string()));
        assert_eq!(lru.pop_front(), Some("a".to_string()));
    }

    #[test]
    fn remove_drops_an_id_from_the_middle() {
        let mut lru = LruList::new();
        lru.touch("a");
        lru.touch("b");
        lru.touch("c");
        lru.remove("b");

        assert_eq!(lru.len(), 2);
        assert_eq!(lru.pop_front(), Some("a".to_string()));
        assert_eq!(lru.pop_front(), Some("c".to_string()));
    }

    #[test]
    fn remove_of_the_front_or_back_relinks_correctly() {
        let mut lru = LruList::new();
        lru.touch("a");
        lru.touch("b");
        lru.touch("c");
        lru.remove("a"); // front
        lru.remove("c"); // back

        assert_eq!(lru.len(), 1);
        assert_eq!(lru.pop_front(), Some("b".to_string()));
    }

    #[test]
    fn remove_of_a_missing_id_is_a_noop() {
        let mut lru = LruList::new();
        lru.touch("a");
        lru.remove("missing");

        assert_eq!(lru.len(), 1);
    }

    #[test]
    fn freed_slots_are_reused() {
        let mut lru = LruList::new();
        lru.touch("a");
        lru.touch("b");
        lru.remove("a");
        lru.touch("c"); // should reuse a's freed slot rather than growing

        assert_eq!(lru.len(), 2);
        assert_eq!(lru.pop_front(), Some("b".to_string()));
        assert_eq!(lru.pop_front(), Some("c".to_string()));
    }

    #[test]
    fn pop_front_on_an_empty_list_is_none() {
        let mut lru = LruList::new();
        assert_eq!(lru.pop_front(), None);
    }

    #[test]
    fn touch_is_idempotent_for_repeated_touches_of_the_same_id() {
        let mut lru = LruList::new();
        lru.touch("a");
        lru.touch("a");
        lru.touch("a");

        assert_eq!(lru.len(), 1);
        assert_eq!(lru.pop_front(), Some("a".to_string()));
        assert_eq!(lru.pop_front(), None);
    }
}
