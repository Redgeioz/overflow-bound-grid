//! Packed R-trees built afresh from every item at each refresh, as a frame
//! that rebuilds its index would use them: the Hilbert-packed static index of
//! static_aabb2d_index 2.1.0 (a port of flatbush), and the R-tree of
//! geo-index 0.4.0 (flatbush's layout) packed by Hilbert curve or by STR.
//! Neither library updates a built tree.
//!
//! Inserts and updates only record the item's bounds; [`refresh_tree`]
//! rebuilds the tree from the live items, or defers the rebuild while few
//! entries changed. Until the rebuild an inserted or moved item is pending:
//! the queries skip its entry in the tree and test it on its own, so that
//! results are exact between refreshes too. While nothing is pending, as
//! right after a rebuild, the queries pass the library's candidates straight
//! to the exact test, which reads the bounds the tree was built from in the
//! order of insertion.
//!
//! Both libraries count touching boxes as intersecting; every candidate is
//! tested again with [`Rectangle::intersects`]. Items with a NaN corner or a
//! min above a max, which static_aabb2d_index rejects in debug builds, stay
//! pending outside the tree.
//!
//! [`refresh_tree`]: PackedRTree::refresh_tree

use crate::handle::{ItemHandle, Slots};
use crate::rect::Rectangle;
use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::{Node, RTree, RTreeBuilder, RTreeIndex};
use static_aabb2d_index::{StaticAABB2DIndex, StaticAABB2DIndexBuilder};
use std::cell::Cell;

/// The library and packing of the tree, and how it answers the queries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Packing {
    /// static_aabb2d_index, Hilbert-packed. Searches visit the tree with a
    /// reused stack; the pair query searches with every item, in the tree's
    /// order, and keeps the pairs whose other item was inserted later.
    StaticHilbert,
    /// [`StaticHilbert`](Self::StaticHilbert), but the pair query joins the
    /// tree with itself, written here on the library's public node arrays
    /// (`all_boxes`, `all_box_indices`, `level_bounds`), which it has no join
    /// of its own for: each node's children are paired among themselves,
    /// and every overlapping pair of nodes descends into the overlapping
    /// pairs of their children, so each pair of items is reached once.
    StaticHilbertJoin,
    /// geo-index, Hilbert-packed. Searches use `search`, which returns a new
    /// vector; the pair query joins the tree with itself by
    /// `intersection_candidates_with_other_tree`, which reports every pair
    /// in both orders and every item with itself.
    GeoHilbert,
    /// [`GeoHilbert`](Self::GeoHilbert) with the tree packed by STR.
    GeoStr,
    /// [`GeoHilbert`](Self::GeoHilbert), but searches walk the tree through
    /// the library's public `Node` traversal, allocating nothing.
    GeoHilbertWalk,
}

/// Position in `pending` of an item that is not pending.
const NOT_PENDING: u32 = u32::MAX;
/// Insertion index of an item without an entry in the tree.
const NOT_IN_TREE: u32 = u32::MAX;

struct Leaf<T> {
    bounds: Rectangle,
    item: T,
    live: bool,
    /// Insertion index of the item's entry in the built tree, or
    /// `NOT_IN_TREE`.
    entry: u32,
    /// Position in `pending`, or `NOT_PENDING`.
    pending: u32,
}

/// An item as the tree was built from it, by insertion index.
struct Entry<T> {
    bounds: Rectangle,
    item: T,
    slot: u32,
    /// Cleared when the item is removed, so that the queries read nothing
    /// but the entries while no item is pending.
    live: bool,
}

enum Tree {
    Empty,
    Static(StaticAABB2DIndex<f64>),
    Geo(RTree<f64>),
}

pub struct PackedRTree<T: Copy> {
    packing: Packing,
    /// Children per node of the built trees.
    node_size: usize,
    /// Whether a rebuild adds the items in the order of the last tree, so
    /// that the sort finds them nearly sorted and the entries lie in about
    /// the tree's order; else in the order of their slots.
    coherent: bool,
    /// Slots of the last tree's items in its order, for `coherent`.
    order: Vec<u32>,
    /// Share of the tree's entries that may have changed before a refresh
    /// rebuilds; until then the queries test the pending items one by one.
    defer: f64,
    /// Entries of removed items in the built tree.
    dead: usize,
    /// Trees built so far.
    rebuilds: usize,
    leaves: Vec<Leaf<T>>,
    slots: Slots<()>,
    /// Slots of the live items whose bounds the tree does not hold.
    pending: Vec<u32>,
    tree: Tree,
    /// The items the tree was built from, by insertion index.
    entries: Vec<Entry<T>>,
    /// The traversal stack of static_aabb2d_index's searches, kept between
    /// them; a cell makes the structure `Send` but not `Sync`.
    stack: Cell<Vec<usize>>,
    /// The stack of node pairs of [`Packing::StaticHilbertJoin`]: two
    /// positions in `all_boxes` and their level, or one twice for a node to
    /// pair its children.
    join: Cell<Vec<(usize, usize, usize)>>,
}

/// Whether a box can enter a tree: no NaN corner, no min above its max.
fn well_formed(bounds: &Rectangle) -> bool {
    bounds.min_x <= bounds.max_x && bounds.min_y <= bounds.max_y
}

impl<T: Copy> PackedRTree<T> {
    /// An empty tree with the libraries' default of 16 children per node.
    pub fn new(packing: Packing) -> Self {
        Self::with_node_size(packing, 16)
    }

    pub fn with_node_size(packing: Packing, node_size: usize) -> Self {
        assert!(
            (2..=u16::MAX as usize).contains(&node_size),
            "A node holds 2 to 65535 children."
        );
        Self {
            packing,
            node_size,
            coherent: false,
            order: Vec::new(),
            defer: 0.0,
            dead: 0,
            rebuilds: 0,
            leaves: Vec::new(),
            slots: Slots::new(),
            pending: Vec::new(),
            tree: Tree::Empty,
            entries: Vec::new(),
            stack: Cell::new(Vec::new()),
            join: Cell::new(Vec::new()),
        }
    }

    /// The same tree, but rebuilt from the items in the order of the last
    /// tree.
    pub fn coherent(mut self) -> Self {
        self.coherent = true;
        self
    }

    /// The same tree, but a refresh rebuilds it only once more than `share`
    /// of its entries have changed: items moved or inserted since the last
    /// rebuild, and entries of removed items.
    pub fn deferred(mut self, share: f64) -> Self {
        self.defer = share;
        self
    }

    /// Trees built so far.
    pub fn rebuilds(&self) -> usize {
        self.rebuilds
    }

    /// Insert an item, pending until the next refresh, and return the handle
    /// that identifies it for [`update`](Self::update) and
    /// [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let handle = self.slots.allocate(());
        let slot = handle.index();
        let leaf = Leaf {
            bounds,
            item,
            live: true,
            entry: NOT_IN_TREE,
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
        self.unmark_pending(slot);
        let leaf = &mut self.leaves[slot];
        leaf.live = false;
        // The entry stays in the tree until the next refresh; the queries
        // skip it.
        if leaf.entry != NOT_IN_TREE {
            self.entries[leaf.entry as usize].live = false;
            leaf.entry = NOT_IN_TREE;
            self.dead += 1;
        }
        self.slots.free(handle);
    }

    /// Record the item's new bounds; it is pending until the next refresh.
    /// Panics if the handle has been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.slots.get(handle);
        let slot = handle.index();
        self.leaves[slot].bounds = bounds;
        if self.leaves[slot].pending == NOT_PENDING {
            self.mark_pending(slot);
        }
    }

    /// Build the tree afresh from every live item that can enter it, unless
    /// no more than the deferred share of its entries changed: pending items
    /// that can enter the tree, and dead entries of removed items. Neither
    /// library can update a built tree, so a single change costs a whole
    /// rebuild, but no change costs nothing; items that cannot enter the tree
    /// stay pending and do not count.
    pub fn refresh_tree(&mut self) {
        if !matches!(self.tree, Tree::Empty) {
            let moved = self
                .pending
                .iter()
                .filter(|&&slot| well_formed(&self.leaves[slot as usize].bounds))
                .count();
            if (moved + self.dead) as f64 <= self.defer * self.entries.len() as f64 {
                return;
            }
        }
        self.dead = 0;
        self.rebuilds += 1;
        let count = self
            .leaves
            .iter()
            .filter(|leaf| leaf.live && well_formed(&leaf.bounds))
            .count();
        self.entries.clear();
        self.entries.reserve(count);
        let mut kept = 0;
        for index in 0..self.pending.len() {
            let slot = self.pending[index];
            let leaf = &mut self.leaves[slot as usize];
            if well_formed(&leaf.bounds) {
                leaf.pending = NOT_PENDING;
            } else {
                leaf.pending = kept as u32;
                self.pending[kept] = slot;
                kept += 1;
            }
        }
        self.pending.truncate(kept);
        for leaf in &mut self.leaves {
            leaf.entry = NOT_IN_TREE;
        }
        let slots = self.leaves.len();
        macro_rules! add_all {
            ($builder:expr) => {
                let mut add = |slot: usize| {
                    let leaf = &mut self.leaves[slot];
                    if leaf.live && leaf.entry == NOT_IN_TREE && well_formed(&leaf.bounds) {
                        let b = &leaf.bounds;
                        $builder.add(b.min_x, b.min_y, b.max_x, b.max_y);
                        leaf.entry = self.entries.len() as u32;
                        self.entries.push(Entry {
                            bounds: leaf.bounds.clone(),
                            item: leaf.item,
                            slot: slot as u32,
                            live: true,
                        });
                    }
                };
                if self.coherent {
                    for &slot in &self.order {
                        add(slot as usize);
                    }
                }
                // Every item not added yet: all of them, or the new ones.
                for slot in 0..slots {
                    add(slot);
                }
            };
        }
        self.tree = match self.packing {
            Packing::StaticHilbert | Packing::StaticHilbertJoin => {
                let mut builder =
                    StaticAABB2DIndexBuilder::new_with_node_size(count, self.node_size);
                add_all!(builder);
                Tree::Static(builder.build().expect("Every counted item was added."))
            }
            Packing::GeoHilbert | Packing::GeoStr | Packing::GeoHilbertWalk => {
                let mut builder =
                    RTreeBuilder::<f64>::new_with_node_size(count as u32, self.node_size as u16);
                add_all!(builder);
                Tree::Geo(if self.packing == Packing::GeoStr {
                    builder.finish::<STRSort>()
                } else {
                    builder.finish::<HilbertSort>()
                })
            }
        };
        if self.coherent {
            let entries = &self.entries;
            self.order.clear();
            match &self.tree {
                Tree::Empty => {}
                Tree::Static(tree) => {
                    self.order
                        .extend(tree.item_indices().iter().map(|&index| entries[index].slot));
                }
                Tree::Geo(tree) => {
                    let indices = tree.indices();
                    self.order.extend(
                        (0..entries.len()).map(|position| entries[indices.get(position)].slot),
                    );
                }
            }
        }
    }

    /// Whether an entry stands for its item as it is: not removed, and not
    /// moved since the tree was built.
    fn current(&self, entry: &Entry<T>) -> bool {
        entry.live && self.leaves[entry.slot as usize].pending == NOT_PENDING
    }

    /// Call `found` with the insertion index of every entry whose box meets
    /// `bounds`, edges included.
    fn candidates(&self, bounds: &Rectangle, mut found: impl FnMut(usize)) {
        let (min_x, min_y, max_x, max_y) = (bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y);
        match &self.tree {
            Tree::Empty => {}
            Tree::Static(tree) => {
                let mut stack = self.stack.take();
                tree.visit_query_with_stack(
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    &mut |index| found(index),
                    &mut stack,
                );
                self.stack.set(stack);
            }
            Tree::Geo(tree) if self.packing == Packing::GeoHilbertWalk => {
                if tree.num_items() > 0 {
                    walk(&tree.root(), bounds, &mut found);
                }
            }
            Tree::Geo(tree) => {
                if tree.num_items() > 0 {
                    for index in tree.search(min_x, min_y, max_x, max_y) {
                        found(index as usize);
                    }
                }
            }
        }
    }

    /// Pass every item whose bounds meet `bounds` to `callback`.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        if self.pending.is_empty() {
            self.candidates(bounds, |index| {
                let entry = &self.entries[index];
                if entry.live && entry.bounds.intersects(bounds) {
                    callback(entry.item);
                }
            });
            return;
        }
        self.candidates(bounds, |index| {
            let entry = &self.entries[index];
            if self.current(entry) && entry.bounds.intersects(bounds) {
                callback(entry.item);
            }
        });
        for &slot in &self.pending {
            let leaf = &self.leaves[slot as usize];
            if leaf.bounds.intersects(bounds) {
                callback(leaf.item);
            }
        }
    }

    /// Report every intersecting pair of distinct items once, in either
    /// order, as [`Packing`] says; then, if any item is pending, every
    /// pending item against the tree and the later pending items.
    pub fn for_each_pair(&self, mut callback: impl FnMut(T, T)) {
        let all_current = self.pending.is_empty();
        let pair = |a: usize, b: usize, callback: &mut dyn FnMut(T, T)| {
            let (a, b) = (&self.entries[a], &self.entries[b]);
            let current = if all_current {
                a.live && b.live
            } else {
                self.current(a) && self.current(b)
            };
            if current && a.bounds.intersects(&b.bounds) {
                callback(a.item, b.item);
            }
        };
        match &self.tree {
            Tree::Empty => {}
            Tree::Static(tree) if self.packing == Packing::StaticHilbertJoin => {
                let mut stack = self.join.take();
                self_join(tree, &mut stack, |a, b| pair(a, b, &mut callback));
                self.join.set(stack);
            }
            Tree::Static(tree) => {
                // In the tree's order, so that consecutive searches read
                // nearby nodes.
                let mut stack = self.stack.take();
                for (aabb, &a) in tree.item_boxes().iter().zip(tree.item_indices()) {
                    tree.visit_query_with_stack(
                        aabb.min_x,
                        aabb.min_y,
                        aabb.max_x,
                        aabb.max_y,
                        &mut |b: usize| {
                            if b > a {
                                pair(a, b, &mut callback);
                            }
                        },
                        &mut stack,
                    );
                }
                self.stack.set(stack);
            }
            Tree::Geo(tree) => {
                if tree.num_items() > 1 {
                    for (a, b) in tree.intersection_candidates_with_other_tree(tree) {
                        if a < b {
                            pair(a as usize, b as usize, &mut callback);
                        }
                    }
                }
            }
        }
        for (index, &slot) in self.pending.iter().enumerate() {
            let leaf = &self.leaves[slot as usize];
            self.candidates(&leaf.bounds, |other| {
                let other = &self.entries[other];
                if self.current(other) && other.bounds.intersects(&leaf.bounds) {
                    callback(leaf.item, other.item);
                }
            });
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

/// Call `found` with the insertion index of every leaf below `node` whose
/// box meets `bounds`, edges included, through geo-index's `Node`.
fn walk<T: RTreeIndex<f64>>(
    node: &Node<'_, f64, T>,
    bounds: &Rectangle,
    found: &mut impl FnMut(usize),
) {
    let meets = |node: &Node<'_, f64, T>| {
        node.min_x() <= bounds.max_x
            && bounds.min_x <= node.max_x()
            && node.min_y() <= bounds.max_y
            && bounds.min_y <= node.max_y()
    };
    if !meets(node) {
        return;
    }
    if node.is_leaf() {
        found(node.insertion_index_unchecked() as usize);
        return;
    }
    for child in node.children_unchecked() {
        if child.is_leaf() {
            if meets(&child) {
                found(child.insertion_index_unchecked() as usize);
            }
        } else {
            walk(&child, bounds, found);
        }
    }
}

/// Call `report` once with the insertion indices of every pair of distinct
/// items of `tree` whose boxes meet, edges included, by joining the tree with
/// itself on its public node arrays. Every level of a packed tree is full but
/// its last node, so both nodes of a pair are always on the same level.
fn self_join(
    tree: &StaticAABB2DIndex<f64>,
    stack: &mut Vec<(usize, usize, usize)>,
    mut report: impl FnMut(usize, usize),
) {
    let count = tree.count();
    if count < 2 {
        return;
    }
    let boxes = tree.all_boxes();
    let indices = tree.all_box_indices();
    let level_bounds = tree.level_bounds();
    let node_size = tree.node_size();
    // The children of the node at `position` on `level`, on the level below.
    let children = |position: usize, level: usize| {
        let start = indices[position];
        start..(start + node_size).min(level_bounds[level - 1])
    };
    let root = boxes.len() - 1;
    stack.clear();
    stack.push((root, root, level_bounds.len() - 1));
    while let Some((a, b, level)) = stack.pop() {
        if a == b {
            let children = children(a, level);
            for i in children.clone() {
                if level > 1 {
                    stack.push((i, i, level - 1));
                }
                for j in i + 1..children.end {
                    if boxes[i].overlaps_aabb(&boxes[j]) {
                        if level == 1 {
                            report(indices[i], indices[j]);
                        } else {
                            stack.push((i, j, level - 1));
                        }
                    }
                }
            }
        } else {
            let right = children(b, level);
            for i in children(a, level) {
                if !boxes[i].overlaps_aabb(&boxes[b]) {
                    continue;
                }
                for j in right.clone() {
                    if boxes[i].overlaps_aabb(&boxes[j]) {
                        if level == 1 {
                            report(indices[i], indices[j]);
                        } else {
                            stack.push((i, j, level - 1));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    const PACKINGS: [Packing; 5] = [
        Packing::StaticHilbert,
        Packing::StaticHilbertJoin,
        Packing::GeoHilbert,
        Packing::GeoStr,
        Packing::GeoHilbertWalk,
    ];

    fn found(tree: &PackedRTree<usize>, query: &Rectangle) -> Vec<usize> {
        let mut found = Vec::new();
        tree.search(query, |id| found.push(id));
        found.sort_unstable();
        found
    }

    fn pairs(tree: &PackedRTree<usize>) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        tree.for_each_pair(|a, b| pairs.push((a.min(b), a.max(b))));
        pairs.sort_unstable();
        pairs
    }

    /// Trees of zero to three items, below the node size, and touching boxes.
    #[test]
    fn small_trees_stay_exact() {
        for packing in PACKINGS {
            let mut tree = PackedRTree::new(packing);
            let everything = Rectangle::new(-10.0, -10.0, 10.0, 10.0);
            tree.refresh_tree();
            assert!(found(&tree, &everything).is_empty());
            assert!(pairs(&tree).is_empty());
            let mut handles = Vec::new();
            for id in 0..3 {
                let x = id as f64;
                handles.push(tree.insert(Rectangle::new(x, 0.0, x + 1.5, 1.0), id));
                assert_eq!(found(&tree, &everything), (0..=id).collect::<Vec<_>>());
                tree.refresh_tree();
                assert_eq!(found(&tree, &everything), (0..=id).collect::<Vec<_>>());
            }
            assert_eq!(pairs(&tree), [(0, 1), (1, 2)], "{packing:?}");
            // Touching at x = 1.5 does not count.
            assert_eq!(
                found(&tree, &Rectangle::new(1.5, 0.0, 1.9, 1.0)),
                [1],
                "{packing:?}"
            );
            // Removed right after a rebuild, with nothing pending.
            tree.remove(handles[1]);
            assert_eq!(found(&tree, &everything), [0, 2], "{packing:?}");
            assert_eq!(pairs(&tree), [], "{packing:?}");
            tree.update(Rectangle::new(0.5, 0.0, 2.5, 1.0), handles[0]);
            assert_eq!(pairs(&tree), [(0, 2)], "{packing:?}");
            tree.refresh_tree();
            assert_eq!(pairs(&tree), [(0, 2)], "{packing:?}");
        }
    }

    /// NaN and inverted boxes stay out of the trees; an inverted box can
    /// still meet others as `Rectangle::intersects` reads it.
    #[test]
    fn odd_boxes_stay_pending() {
        let items = [
            Rectangle::new(0.0, 0.0, 1.0, 1.0),
            Rectangle::new(0.5, f64::NAN, 2.0, 2.0),
            // Inverted in x, yet met by boxes spanning 0.5 to 1.0.
            Rectangle::new(1.0, 0.0, 0.5, 1.0),
            Rectangle::new(0.8, 0.2, 1.2, 0.6),
        ];
        let mut expected = Vec::new();
        for a in 0..items.len() {
            for b in a + 1..items.len() {
                if items[a].intersects(&items[b]) {
                    expected.push((a, b));
                }
            }
        }
        for packing in PACKINGS {
            let mut tree = PackedRTree::new(packing);
            let handles: Vec<_> = items
                .iter()
                .enumerate()
                .map(|(id, b)| tree.insert(b.clone(), id))
                .collect();
            for _ in 0..2 {
                assert_eq!(pairs(&tree), expected, "{packing:?}");
                tree.refresh_tree();
            }
            assert_eq!(
                found(&tree, &Rectangle::new(0.4, 0.0, 1.1, 1.0)),
                [0, 2, 3],
                "{packing:?}"
            );
            tree.update(Rectangle::new(0.5, 0.5, 2.0, 2.0), handles[1]);
            tree.refresh_tree();
            assert_eq!(
                found(&tree, &Rectangle::new(1.5, 1.5, 1.6, 1.6)),
                [1],
                "{packing:?}"
            );
        }
    }

    /// A slot freed and reused must not inherit the old item's entry.
    #[test]
    fn reused_slots_get_new_entries() {
        for packing in PACKINGS {
            let mut tree = PackedRTree::new(packing);
            let handles: Vec<_> = (0..40)
                .map(|id| tree.insert(Rectangle::new(id as f64, 0.0, id as f64 + 0.5, 0.5), id))
                .collect();
            tree.refresh_tree();
            tree.remove(handles[3]);
            let reused = tree.insert(Rectangle::new(100.0, 100.0, 101.0, 101.0), 50);
            assert_eq!(reused.index(), handles[3].index());
            let old_place = Rectangle::new(3.0, 0.0, 3.4, 0.4);
            assert!(found(&tree, &old_place).is_empty(), "{packing:?}");
            tree.refresh_tree();
            assert!(found(&tree, &old_place).is_empty(), "{packing:?}");
            assert_eq!(
                found(&tree, &Rectangle::new(99.0, 99.0, 102.0, 102.0)),
                [50]
            );
        }
    }

    fn random_box(rng: &mut StdRng) -> Rectangle {
        let (x, y) = (rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0));
        Rectangle::new(
            x,
            y,
            x + rng.gen_range(0.5..6.0),
            y + rng.gen_range(0.5..6.0),
        )
    }

    /// Searches with `queries`, with every live item's bounds and with random
    /// boxes, and the pairs, against brute force over `items` (by id).
    fn check(
        tree: &PackedRTree<usize>,
        items: &[Option<(ItemHandle, Rectangle)>],
        queries: &[Rectangle],
        rng: &mut StdRng,
        context: &str,
    ) {
        let live: Vec<(usize, &Rectangle)> = items
            .iter()
            .enumerate()
            .filter_map(|(id, item)| item.as_ref().map(|(_, bounds)| (id, bounds)))
            .collect();
        let mut queries = queries.to_vec();
        queries.extend(live.iter().map(|(_, bounds)| (*bounds).clone()));
        queries.extend((0..40).map(|_| random_box(rng)));
        for query in &queries {
            let expected: Vec<usize> = live
                .iter()
                .filter(|(_, bounds)| bounds.intersects(query))
                .map(|&(id, _)| id)
                .collect();
            assert_eq!(found(tree, query), expected, "{context}: search {query:?}");
        }
        let mut expected = Vec::new();
        for (index, &(a, first)) in live.iter().enumerate() {
            for &(b, second) in &live[index + 1..] {
                if first.intersects(second) {
                    expected.push((a, b));
                }
            }
        }
        assert_eq!(pairs(tree), expected, "{context}: pairs");
    }

    /// Up to three items move per frame, now and then one is removed, one
    /// inserted, or one removed and another inserted into its slot, and the
    /// queries come right after, before the refresh: the moved items must be
    /// found at their new bounds only, the removed ones nowhere. A tree that
    /// defers its rebuilds keeps stale entries for many frames; a refresh
    /// after a frame without changes, or with changes only to boxes that
    /// cannot enter the tree, rebuilds nothing.
    #[test]
    fn few_moves_are_found_before_and_between_rebuilds() {
        for packing in PACKINGS {
            for (coherent, defer) in [
                (false, 0.0),
                (true, 0.0),
                (true, 0.02),
                (false, 1.0),
                (true, 1.0),
            ] {
                let context = format!("{packing:?}, coherent {coherent}, defer {defer}");
                let mut rng = StdRng::seed_from_u64(5);
                let tree = PackedRTree::with_node_size(packing, 4).deferred(defer);
                let mut tree = if coherent { tree.coherent() } else { tree };
                let mut items: Vec<Option<(ItemHandle, Rectangle)>> = (0..300)
                    .map(|id| {
                        let bounds = random_box(&mut rng);
                        Some((tree.insert(bounds.clone(), id), bounds))
                    })
                    .collect();
                tree.refresh_tree();
                assert_eq!(tree.rebuilds(), 1, "{context}");
                let mut stale_frames = 0;
                for frame in 0..80 {
                    // Frames without moves: nothing at all, or a removal only.
                    let (moves, remove, insert) = match frame % 10 {
                        0 => (0, false, false),
                        5 => (0, true, false),
                        1 | 6 => (rng.gen_range(1..=3), true, false),
                        2 | 7 => (rng.gen_range(1..=3), false, true),
                        3 => (rng.gen_range(1..=3), true, true),
                        _ => (rng.gen_range(1..=3), false, false),
                    };
                    // Whether a box that can enter the tree changed.
                    let mut changed = false;
                    // The old bounds of the moved and removed items, where they
                    // must no longer be found.
                    let mut old = Vec::new();
                    for _ in 0..moves {
                        let id = rng.gen_range(0..items.len());
                        if let Some((handle, bounds)) = &mut items[id] {
                            old.push(bounds.clone());
                            *bounds = match rng.gen_range(0..10) {
                                0 => Rectangle::new(
                                    f64::NAN,
                                    bounds.min_y,
                                    bounds.max_x,
                                    bounds.max_y,
                                ),
                                // Inverted in x.
                                1 => Rectangle::new(
                                    bounds.max_x,
                                    bounds.min_y,
                                    bounds.min_x,
                                    bounds.max_y,
                                ),
                                2..=4 => random_box(&mut rng),
                                _ => {
                                    let (dx, dy) =
                                        (rng.gen_range(-2.0..2.0), rng.gen_range(-2.0..2.0));
                                    Rectangle::new(
                                        bounds.min_x + dx,
                                        bounds.min_y + dy,
                                        bounds.max_x + dx,
                                        bounds.max_y + dy,
                                    )
                                }
                            };
                            changed |= well_formed(bounds);
                            tree.update(bounds.clone(), *handle);
                        }
                    }
                    if remove {
                        let id = rng.gen_range(0..items.len());
                        if let Some((handle, bounds)) = items[id].take() {
                            changed |= tree.leaves[handle.index()].entry != NOT_IN_TREE;
                            old.push(bounds);
                            tree.remove(handle);
                        }
                    }
                    if insert {
                        let bounds = random_box(&mut rng);
                        let handle = tree.insert(bounds.clone(), items.len());
                        items.push(Some((handle, bounds)));
                        changed = true;
                    }
                    check(
                        &tree,
                        &items,
                        &old,
                        &mut rng,
                        &format!("{context}, frame {frame}, moved"),
                    );
                    let rebuilds = tree.rebuilds();
                    tree.refresh_tree();
                    if !changed {
                        assert_eq!(
                            tree.rebuilds(),
                            rebuilds,
                            "{context}, frame {frame}: rebuilt without a change"
                        );
                    }
                    if defer == 0.0 && changed {
                        assert_eq!(
                            tree.rebuilds(),
                            rebuilds + 1,
                            "{context}, frame {frame}: change not rebuilt"
                        );
                    }
                    if tree
                        .pending
                        .iter()
                        .any(|&slot| well_formed(&tree.leaves[slot as usize].bounds))
                    {
                        stale_frames += 1;
                    }
                    check(
                        &tree,
                        &items,
                        &old,
                        &mut rng,
                        &format!("{context}, frame {frame}, refreshed"),
                    );
                }
                // Whether the rebuilds were deferred as asked.
                if defer == 0.0 {
                    assert_eq!(stale_frames, 0, "{context}");
                } else if defer == 1.0 {
                    assert_eq!(tree.rebuilds(), 1, "{context}");
                    assert!(
                        stale_frames >= 70,
                        "{context}: {stale_frames} frames with stale entries"
                    );
                } else {
                    assert!(
                        tree.rebuilds() > 2 && stale_frames > 10,
                        "{context}: {} rebuilds, {stale_frames} stale frames",
                        tree.rebuilds()
                    );
                }
            }
        }
    }
}
