//! The bounding volume hierarchy of parry2d-f64 0.31.1, the geometry library
//! under the Rapier physics engine: a binary tree of boxes in two dimensions
//! with `f64` coordinates, the precision of this crate.
//!
//! The tree holds one leaf per item, keyed by the item's slot. Inserts and
//! updates only record the item's bounds; [`refresh_tree`] brings the tree up
//! to date in one of the ways of [`Maintenance`], so that a frame pays for its
//! maintenance once, as an engine's broad phase does. An item that has no leaf
//! yet, or whose bounds left the box of its leaf, is pending until then: the
//! queries skip its stale leaf and test the item on its own, so that their
//! results are exact between refreshes too, as with the grids.
//!
//! Parry counts touching boxes as intersecting and this crate does not, so
//! every candidate is tested again with [`Rectangle::intersects`]. Items with
//! a NaN corner, which meet nothing, stay pending without a leaf, so that they
//! reach neither the builders nor the boxes of the tree.
//!
//! [`refresh_tree`]: ParryBvh::refresh_tree

use crate::handle::{ItemHandle, Slots};
use crate::rect::Rectangle;
use parry2d_f64::bounding_volume::Aabb;
use parry2d_f64::math::Vector;
use parry2d_f64::partitioning::{Bvh, BvhWorkspace};

pub use parry2d_f64::partitioning::BvhBuildStrategy;

/// How [`ParryBvh::refresh_tree`] brings the tree up to date once every
/// pending item's box is in its leaf.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Maintenance {
    /// Rebuild the whole tree over its leaves with `Bvh::rebuild`, binned SAH
    /// or PLOC, reusing the tree's buffers.
    Rebuild(BvhBuildStrategy),
    /// Refit the ancestors of the changed leaves, keeping the topology.
    Refit,
    /// [`Refit`](Self::Refit), then `optimize_incremental`, which rebuilds a
    /// few subtrees on every call. The queries see the rebuilt subtrees where
    /// the optimizer appended them, outside the depth-first order of a refit.
    RefitOptimize,
    /// `optimize_incremental`, then a refit, which also lays the nodes out in
    /// depth-first order: the order of Rapier's broad phase when its optimizer
    /// runs before the pair traversal.
    OptimizeRefit,
    /// Remove the leaf of every pending item and insert it again by SAH
    /// descent, as Rapier's broad phase does when fewer than 1/16 of its
    /// leaves changed, and Avian's when fewer than 15% did.
    Reinsert,
}

/// Position in `pending` of an item that is not pending.
const NOT_PENDING: u32 = u32::MAX;

struct Leaf<T> {
    bounds: Rectangle,
    /// Box of the item's leaf: its bounds grown by the margin when the leaf
    /// was last written. Meaningful only while `in_tree`.
    leaf: Rectangle,
    item: T,
    in_tree: bool,
    /// Position in `pending`, or `NOT_PENDING`.
    pending: u32,
}

pub struct ParryBvh<T: Copy> {
    tree: Bvh,
    workspace: BvhWorkspace,
    maintenance: Maintenance,
    margin: f64,
    /// Every slot's item; a removed item's slot keeps it until reused.
    leaves: Vec<Leaf<T>>,
    slots: Slots<()>,
    /// Slots of the live items whose bounds the tree does not cover.
    pending: Vec<u32>,
    /// Items pending when the last refresh started.
    last_maintained: usize,
}

impl<T: Copy> ParryBvh<T> {
    /// An empty tree. The leaves of moved items are their bounds grown by
    /// `margin` on every side, so that an item that moves less keeps its leaf;
    /// a tree rebuilt at every refresh holds exact bounds and takes no margin.
    pub fn new(maintenance: Maintenance, margin: f64) -> Self {
        assert!(
            margin.is_finite() && margin >= 0.0,
            "The margin must be finite and non-negative."
        );
        assert!(
            margin == 0.0 || !matches!(maintenance, Maintenance::Rebuild(_)),
            "A rebuilt tree takes no margin."
        );
        Self {
            tree: Bvh::new(),
            workspace: BvhWorkspace::default(),
            maintenance,
            margin,
            leaves: Vec::new(),
            slots: Slots::new(),
            pending: Vec::new(),
            last_maintained: 0,
        }
    }

    /// Insert an item, pending until the next refresh, and return the handle
    /// that identifies it for [`update`](Self::update) and
    /// [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let handle = self.slots.allocate(());
        let slot = handle.index();
        let leaf = Leaf {
            bounds,
            leaf: Rectangle::default(),
            item,
            in_tree: false,
            pending: NOT_PENDING,
        };
        if slot == self.leaves.len() {
            self.leaves.push(leaf);
        } else {
            self.leaves[slot] = leaf;
        }
        self.mark_pending(slot);
        handle
    }

    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        self.slots.get(handle);
        let slot = handle.index();
        if self.leaves[slot].in_tree {
            // Parry merges the ancestors again, so the tree stays exact.
            self.tree.remove(slot as u32);
            self.leaves[slot].in_tree = false;
        }
        self.unmark_pending(slot);
        self.slots.free(handle);
    }

    /// Record the item's new bounds; the item becomes pending unless they
    /// stay inside its leaf. Panics if the handle has been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.slots.get(handle);
        let slot = handle.index();
        let leaf = &mut self.leaves[slot];
        leaf.bounds = bounds;
        let covered = leaf.in_tree && contains(&leaf.leaf, &leaf.bounds);
        if leaf.pending == NOT_PENDING && !covered {
            self.mark_pending(slot);
        }
    }

    /// Give every pending item a leaf covering its bounds and maintain the
    /// tree as [`Maintenance`] says. Items with a NaN corner stay pending,
    /// without a leaf.
    pub fn refresh_tree(&mut self) {
        self.last_maintained = self.pending.len();
        if self.pending.is_empty() {
            return;
        }
        let mut pending = std::mem::take(&mut self.pending);
        let mut kept = 0;
        for index in 0..pending.len() {
            let slot = pending[index];
            let leaf = &mut self.leaves[slot as usize];
            if leaf.bounds.has_nan() {
                if leaf.in_tree {
                    self.tree.remove(slot);
                    leaf.in_tree = false;
                }
                leaf.pending = kept as u32;
                pending[kept] = slot;
                kept += 1;
                continue;
            }
            leaf.pending = NOT_PENDING;
            leaf.leaf = grown(&leaf.bounds, self.margin);
            let aabb = to_aabb(&leaf.leaf);
            if self.maintenance == Maintenance::Reinsert {
                if leaf.in_tree {
                    self.tree.remove(slot);
                }
                // A new leaf's ancestors are merged on the way down.
                self.tree.insert(aabb, slot);
            } else {
                // Writes the leaf box, or inserts a new leaf, leaving the
                // ancestors to the refit or rebuild below.
                self.tree.insert_or_update_partially(aabb, slot, 0.0);
            }
            leaf.in_tree = true;
        }
        pending.truncate(kept);
        self.pending = pending;

        match self.maintenance {
            Maintenance::Rebuild(strategy) => self.tree.rebuild(&mut self.workspace, strategy),
            Maintenance::Refit => self.tree.refit(&mut self.workspace),
            Maintenance::RefitOptimize => {
                self.tree.refit(&mut self.workspace);
                self.tree.optimize_incremental(&mut self.workspace);
            }
            Maintenance::OptimizeRefit => {
                self.tree.optimize_incremental(&mut self.workspace);
                self.tree.refit(&mut self.workspace);
            }
            Maintenance::Reinsert => {}
        }
    }

    /// Items that were pending when the last refresh started: inserted, or
    /// moved out of their leaves.
    pub fn last_maintained(&self) -> usize {
        self.last_maintained
    }

    /// Pass every item whose bounds meet `bounds` to `callback`.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        let query = to_aabb(bounds);
        for slot in self.tree.intersect_aabb(&query) {
            let leaf = &self.leaves[slot as usize];
            if leaf.pending == NOT_PENDING && leaf.bounds.intersects(bounds) {
                callback(leaf.item);
            }
        }
        for &slot in &self.pending {
            let leaf = &self.leaves[slot as usize];
            if leaf.bounds.intersects(bounds) {
                callback(leaf.item);
            }
        }
    }

    /// Report every intersecting pair of distinct items once, in either
    /// order: the leaf pairs of parry's traversal of the tree against itself,
    /// then every pending item against the tree and the later pending items.
    pub fn for_each_pair(&self, mut callback: impl FnMut(T, T)) {
        // The traversal keeps only a stack in the workspace.
        let mut workspace = BvhWorkspace::default();
        self.tree
            .traverse_bvtt_single_tree::<false>(&mut workspace, &mut |a, b| {
                let (a, b) = (&self.leaves[a as usize], &self.leaves[b as usize]);
                if a.pending == NOT_PENDING
                    && b.pending == NOT_PENDING
                    && a.bounds.intersects(&b.bounds)
                {
                    callback(a.item, b.item);
                }
            });
        for (index, &slot) in self.pending.iter().enumerate() {
            let leaf = &self.leaves[slot as usize];
            for other in self.tree.intersect_aabb(&to_aabb(&leaf.bounds)) {
                let other = &self.leaves[other as usize];
                if other.pending == NOT_PENDING && other.bounds.intersects(&leaf.bounds) {
                    callback(leaf.item, other.item);
                }
            }
            for &later in &self.pending[index + 1..] {
                let later = &self.leaves[later as usize];
                if later.bounds.intersects(&leaf.bounds) {
                    callback(leaf.item, later.item);
                }
            }
        }
    }

    fn mark_pending(&mut self, slot: usize) {
        self.leaves[slot].pending = self.pending.len() as u32;
        self.pending.push(slot as u32);
    }

    fn unmark_pending(&mut self, slot: usize) {
        let position = self.leaves[slot].pending;
        if position == NOT_PENDING {
            return;
        }
        self.pending.swap_remove(position as usize);
        if let Some(&moved) = self.pending.get(position as usize) {
            self.leaves[moved as usize].pending = position;
        }
        self.leaves[slot].pending = NOT_PENDING;
    }
}

fn to_aabb(bounds: &Rectangle) -> Aabb {
    Aabb::new(
        Vector::new(bounds.min_x, bounds.min_y),
        Vector::new(bounds.max_x, bounds.max_y),
    )
}

fn grown(bounds: &Rectangle, margin: f64) -> Rectangle {
    Rectangle::new(
        bounds.min_x - margin,
        bounds.min_y - margin,
        bounds.max_x + margin,
        bounds.max_y + margin,
    )
}

/// Whether `inner` lies within `outer`, edges included; false if either has
/// a NaN corner.
fn contains(outer: &Rectangle, inner: &Rectangle) -> bool {
    outer.min_x <= inner.min_x
        && outer.min_y <= inner.min_y
        && inner.max_x <= outer.max_x
        && inner.max_y <= outer.max_y
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAINTENANCES: [Maintenance; 6] = [
        Maintenance::Rebuild(BvhBuildStrategy::Binned),
        Maintenance::Rebuild(BvhBuildStrategy::Ploc),
        Maintenance::Refit,
        Maintenance::RefitOptimize,
        Maintenance::OptimizeRefit,
        Maintenance::Reinsert,
    ];

    fn found(tree: &ParryBvh<usize>, query: &Rectangle) -> Vec<usize> {
        let mut found = Vec::new();
        tree.search(query, |id| found.push(id));
        found.sort_unstable();
        found
    }

    fn pairs(tree: &ParryBvh<usize>) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        tree.for_each_pair(|a, b| pairs.push((a.min(b), a.max(b))));
        pairs.sort_unstable();
        pairs
    }

    /// Trees of zero to three leaves, which parry keeps in a partial root.
    #[test]
    fn small_trees_stay_exact() {
        for maintenance in MAINTENANCES {
            let mut tree = ParryBvh::new(maintenance, 0.0);
            let everything = Rectangle::new(-10.0, -10.0, 10.0, 10.0);
            assert!(found(&tree, &everything).is_empty());
            tree.refresh_tree();
            let mut handles = Vec::new();
            for id in 0..3 {
                let x = id as f64;
                handles.push(tree.insert(Rectangle::new(x, 0.0, x + 1.5, 1.0), id));
                assert_eq!(found(&tree, &everything), (0..=id).collect::<Vec<_>>());
                tree.refresh_tree();
                assert_eq!(found(&tree, &everything), (0..=id).collect::<Vec<_>>());
            }
            assert_eq!(pairs(&tree), [(0, 1), (1, 2)], "{maintenance:?}");
            tree.remove(handles[1]);
            assert_eq!(pairs(&tree), [], "{maintenance:?}");
            tree.update(Rectangle::new(0.5, 0.0, 2.5, 1.0), handles[0]);
            assert_eq!(pairs(&tree), [(0, 2)], "{maintenance:?}");
            tree.refresh_tree();
            assert_eq!(pairs(&tree), [(0, 2)], "{maintenance:?}");
            tree.remove(handles[0]);
            tree.remove(handles[2]);
            tree.refresh_tree();
            assert!(found(&tree, &everything).is_empty());
        }
    }

    /// An item with a NaN corner meets nothing, and is found again once its
    /// bounds are finite, before and after a refresh.
    #[test]
    fn nan_items_meet_nothing_until_updated() {
        for maintenance in MAINTENANCES {
            let mut tree = ParryBvh::new(maintenance, 0.0);
            let query = Rectangle::new(0.0, 0.0, 4.0, 4.0);
            let others: Vec<_> = (0..8)
                .map(|id| tree.insert(Rectangle::new(id as f64, 0.0, id as f64 + 0.5, 0.5), id))
                .collect();
            let odd = tree.insert(Rectangle::new(f64::NAN, 1.0, 2.0, 2.0), 99);
            tree.refresh_tree();
            assert_eq!(found(&tree, &query), [0, 1, 2, 3], "{maintenance:?}");
            assert_eq!(tree.last_maintained(), 9);

            tree.update(Rectangle::new(1.0, 1.0, 2.0, 2.0), odd);
            assert_eq!(found(&tree, &query), [0, 1, 2, 3, 99], "{maintenance:?}");
            tree.refresh_tree();
            assert_eq!(found(&tree, &query), [0, 1, 2, 3, 99], "{maintenance:?}");

            // Back to NaN with a leaf in the tree: the leaf goes.
            tree.update(Rectangle::new(1.0, f64::NAN, 2.0, 2.0), odd);
            tree.refresh_tree();
            assert_eq!(found(&tree, &query), [0, 1, 2, 3], "{maintenance:?}");
            assert!(pairs(&tree).iter().all(|&(a, b)| a != 99 && b != 99));
            tree.remove(odd);
            tree.remove(others[0]);
            assert_eq!(found(&tree, &query), [1, 2, 3], "{maintenance:?}");
        }
    }

    /// A slot freed while its leaf is in the tree is reused by a new item,
    /// which must not inherit the old leaf.
    #[test]
    fn reused_slots_get_new_leaves() {
        for maintenance in MAINTENANCES {
            let mut tree = ParryBvh::new(maintenance, 0.0);
            let handles: Vec<_> = (0..16)
                .map(|id| tree.insert(Rectangle::new(id as f64, 0.0, id as f64 + 0.5, 0.5), id))
                .collect();
            tree.refresh_tree();
            tree.remove(handles[3]);
            let reused = tree.insert(Rectangle::new(100.0, 100.0, 101.0, 101.0), 50);
            assert_eq!(reused.index(), handles[3].index());
            let old_place = Rectangle::new(3.0, 0.0, 3.4, 0.4);
            assert!(found(&tree, &old_place).is_empty(), "{maintenance:?}");
            tree.refresh_tree();
            assert!(found(&tree, &old_place).is_empty(), "{maintenance:?}");
            assert_eq!(found(&tree, &Rectangle::new(99.0, 99.0, 102.0, 102.0)), [50]);
        }
    }

    /// With a margin, items that move less than it keep their leaves and are
    /// not maintained.
    #[test]
    fn margins_keep_leaves_of_small_moves() {
        let mut tree = ParryBvh::new(Maintenance::RefitOptimize, 2.0);
        let handles: Vec<_> = (0..10)
            .map(|id| tree.insert(Rectangle::new(10.0 * id as f64, 0.0, 10.0 * id as f64 + 1.0, 1.0), id))
            .collect();
        tree.refresh_tree();
        assert_eq!(tree.last_maintained(), 10);
        for (id, &handle) in handles.iter().enumerate() {
            let shift = if id < 3 { 2.5 } else { 1.5 };
            let x = 10.0 * id as f64 + shift;
            tree.update(Rectangle::new(x, 0.0, x + 1.0, 1.0), handle);
        }
        tree.refresh_tree();
        assert_eq!(tree.last_maintained(), 3);
        assert_eq!(found(&tree, &Rectangle::new(12.0, 0.0, 12.6, 0.5)), [1]);
    }

    #[test]
    #[should_panic(expected = "takes no margin")]
    fn rebuilt_trees_reject_a_margin() {
        ParryBvh::<usize>::new(Maintenance::Rebuild(BvhBuildStrategy::Binned), 1.0);
    }
}
