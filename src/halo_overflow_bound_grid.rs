use crate::cell_order::for_each_cell_hierarchically;
use crate::exact_overflow_bound_grid::DirectionalOverflow;
use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::layer_geometry::{
    LayerGeometry, Levels, build_levels, half_node_corners, scan_range, size_level, within_reach,
};
use crate::node_marks::{BoxedNode, NodeMarks};
use crate::overflow_bound::{cell_overflow, higher, lower};
use crate::rect::{EMPTY, NOWHERE, Rectangle};
use grid::*;

/// `(level, x, y)`
type Coord = (usize, usize, usize);
type Layers<T> = Vec<Grid<HaloOverflowBoundGridNode<T>>>;

/// [`SpanOverflowBoundGrid`] with the neighbor pruning of the ancestor
/// walk moved from the column and row spans of a level to the cells
/// themselves. It uses the storage, positioning and handles of
/// [`HierarchicalLooseGrid`]; every
/// node keeps a box covering its items, and apart from the nodes, in one
/// cache line per node, copies of its own box and of the sides of its eight
/// neighbors' boxes that face it, that is, how far each neighbor's items
/// reach towards the node. The walk reads that line on every level, and from
/// it alone decides which of the node and the eight neighbors within the
/// fixed reach (half a node) to read, instead of reading the node and judging
/// the neighbors by the span of their whole column or row, which on a dense
/// level nearly always meets the query. A node that is read is still skipped
/// when its box misses the query.
///
/// The pair queries go without the copies. Their walk comes to every node,
/// and at each reads the boxes of the four neighbors after it in row-major
/// order, once for all the items of the node.
///
/// The column and row spans of [`SpanOverflowBoundGrid`] remain: they trim
/// the edges of the scan over finer levels, and stand in for the copies next
/// to changed nodes of a stale level, as described below.
///
/// Insertion and update only widen the box of the target node and the spans
/// of its column and row, and mark the level stale and the node changed; the
/// copies in the neighbors are not updated. Until the refresh, the walk
/// judges the neighbors of a query cell next to a marked node of a stale
/// level by the spans, as [`SpanOverflowBoundGrid`] does and at its cost; a
/// cell with no marked neighbor still reads its neighbors' copies, which
/// only a store in a neighbor can leave behind, but reads its own node
/// whatever the copy of its box says, since a store in the node itself,
/// which the marks around it leave out, can leave that copy behind. Where
/// the last refresh found more than an eighth of a level's occupied nodes
/// marked, most cells have a marked neighbor, and the walk goes by the
/// touched rows instead, without reading the marks. With 1% of 160,000
/// balls moving between refreshes, the region queries before the refresh
/// took 0.95 times as long as when every touched row fell back, with every
/// ball moving they changed by about 1%, and after the refresh by at most
/// 1%, on a Core Ultra 9 275HX with rustc 1.99 and the 96-byte copies of
/// that time (`bench-results/stale-marks-2026-11-09.md`). The pair queries
/// read the boxes and so never fall back.
/// Pushing every grown side to the three neighbors it faces on each update
/// instead measured the update benchmark 50-90% slower than
/// [`SpanOverflowBoundGrid`], for a search gain realized only once a frame's updates are done.
/// Removal and moving out leave boxes, spans and copies unchanged, except
/// that a node left without items resets its box. Boxes and spans therefore
/// stay valid as items move, shrink or leave, but tighten only in
/// [`refresh_bounds`], which also brings the copies up to date and clears
/// the marks; the caller chooses when, e.g. once per frame after the updates.
/// Every store and take also marks its node, so that when few nodes changed,
/// the refresh visits only those and their neighbors' copies of them.
///
/// The walk also stops at the coarsest level below the root that holds items,
/// which a bitmask of the occupied levels keeps exact at all times.
///
/// Boxes and spans consist of item coordinates, so their checks compare the
/// same values as the intersection tests and need no rounding margin. The
/// copies are those coordinates rounded outwards to `f32` and compared with
/// the query rounded outwards, so they can only select a node too many. None
/// of these needs cell edges, which also keeps them cheap to maintain.
///
/// [`SpanOverflowBoundGrid`]: crate::SpanOverflowBoundGrid
/// [`HierarchicalLooseGrid`]: crate::HierarchicalLooseGrid
/// [`refresh_bounds`]: Self::refresh_bounds
pub struct HaloOverflowBoundGrid<T: Copy, const MAX_LEVEL: u8> {
    layers: Layers<T>,
    /// Indexed by level, parallel to `layers`.
    geometry: Vec<LayerGeometry>,
    /// Reciprocals of the node width and height, indexed by level. Position
    /// and search multiply by them instead of dividing by the node size, and
    /// `geometry` builds its cell edges with the same arithmetic.
    inverse_node_size: Vec<(f64, f64)>,
    /// Root size that the level of an item or query follows from, by
    /// [`sizing_root`](crate::layer_geometry::sizing_root).
    size_root: (f64, f64),
    /// Indexed by level, parallel to `layers`.
    lines: Vec<LineSpan>,
    /// Indexed by level, parallel to `layers`.
    marks: Box<[Marks]>,
    /// Copies of every node, indexed by level and then by node in row-major
    /// order. Kept apart from the nodes so that these stay 56 bytes for the
    /// scan and for updates, which never touch the copies; the walk reads one
    /// entry here on every level, and the nodes it selects.
    halo: Vec<Box<[Halo]>>,
    /// Bit `L` is set while level `L` holds items: storing an item sets the
    /// bit of its level, and taking the last item off a level clears it. The
    /// coarsest level below the root that holds items, where the ancestor walk
    /// of search stops, is therefore exact at all times, not only after a
    /// refresh. With all items on the finest level, stopping the walk there
    /// measured 10% faster than walking every level.
    occupied_levels: u64,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    /// Location of the entry of each live handle.
    slots: Slots<GridLocation>,
}

impl<T: Copy, const MAX_LEVEL: u8> HaloOverflowBoundGrid<T, MAX_LEVEL> {
    /// Create a new grid with the given bounds.
    ///
    /// Use square nodes, which usually makes searches more efficient.
    pub fn new(world_bounds: Rectangle) -> HaloOverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new grid with the given bounds.
    ///
    /// Fit the nodes to the given bounds instead of making them square.
    pub fn new_fit(world_bounds: Rectangle) -> HaloOverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(world_bounds: Rectangle) -> HaloOverflowBoundGrid<T, MAX_LEVEL> {
        let Levels {
            layers,
            root_bounds,
            geometry,
            inverse_node_size,
            size_root,
        } = build_levels::<_, FIT>(&world_bounds, MAX_LEVEL, HaloOverflowBoundGridNode::new);

        // The root's spans grow like those of every level but are never read:
        // search reads the root whole.
        let lines = layers
            .iter()
            .map(|grid| LineSpan::new(grid.cols(), grid.rows()))
            .collect();

        let marks = vec![Marks::default(); layers.len()].into_boxed_slice();
        let halo = layers
            .iter()
            .map(|grid| vec![NO_HALO; grid.rows() * grid.cols()].into_boxed_slice())
            .collect();

        HaloOverflowBoundGrid {
            layers,
            geometry,
            inverse_node_size,
            size_root,
            lines,
            marks,
            halo,
            occupied_levels: 0,
            root_bounds,
            world_bounds,
            slots: Slots::new(),
        }
    }

    /// Return the real bounds of the root.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root_bounds
    }

    /// Largest overflow beyond the cells on `level` since the last
    /// [`refresh_bounds`](Self::refresh_bounds), in world units, as implied
    /// by the spans of its columns and rows: negative when every item keeps
    /// that far off its cell edge, -infinity on a level without items. Search
    /// uses the spans directly.
    ///
    /// Panics unless `level` is in `2..=MAX_LEVEL`; the root has no neighbors
    /// to bound.
    pub fn overflow_bound(&self, level: usize) -> DirectionalOverflow {
        assert!((2..=MAX_LEVEL as usize).contains(&level), "invalid level");
        let geometry = &self.geometry[level];
        let node_size = (geometry.node_width, geometry.node_height);
        let origin = self.root_bounds.get_min();
        let lines = &self.lines[level];
        // Use the cell edges of `cell_overflow`, so the bound matches it item
        // by item.
        let mut largest = [f64::NEG_INFINITY; 4];
        for (x, &[min_x, max_x]) in lines.columns.iter().enumerate() {
            let span = Rectangle::new(min_x, origin.1, max_x, origin.1);
            let [left, right, ..] = cell_overflow(&span, origin, node_size, (x, 0));
            largest[0] = largest[0].max(left);
            largest[1] = largest[1].max(right);
        }
        for (y, &[min_y, max_y]) in lines.rows.iter().enumerate() {
            let span = Rectangle::new(origin.0, min_y, origin.0, max_y);
            let [.., top, bottom] = cell_overflow(&span, origin, node_size, (0, y));
            largest[2] = largest[2].max(top);
            largest[3] = largest[3].max(bottom);
        }
        let [left, right, top, bottom] = largest;
        DirectionalOverflow {
            left,
            right,
            top,
            bottom,
        }
    }

    /// On every level changed since the last refresh, shrink the boxes of the
    /// nodes that items were stored in, replaced in or taken from to the items
    /// held now, bring the copies of their sides in their neighbors and the
    /// spans of their columns and rows up to date, and clear the marks of the
    /// level. This undoes the growth from items that have since moved, shrunk
    /// or left. A level nothing was stored in or taken from is already exact
    /// and is skipped. Either way of refreshing a level below leaves the
    /// boxes, spans and copies computed from the items (a span end at zero
    /// may keep either sign, which no comparison tells apart).
    ///
    /// Where at most half the nodes of a level are marked, only the marked
    /// nodes are visited, after reading the marks of the touched rows: each shrinks its box and
    /// pushes its sides to the copies its eight neighbors hold of it, and a
    /// column or row span is rebuilt from its boxes only if a marked node held
    /// one of its ends and no longer reaches it, or a node in it emptied.
    /// Otherwise the level is swept row by row, visiting every node once and
    /// the items of the marked nodes: the boxes of those and the row spans of
    /// the touched rows shrink, every column span is rebuilt from the boxes, and every
    /// node in or next to a touched row pulls the sides it needs from its
    /// neighbors, one row behind the shrinking, rather than every node pushing
    /// its sides, which would first require resetting every copy.
    pub fn refresh_bounds(&mut self) {
        self.refresh_with(NodeMarks::is_sparse);
    }

    /// [`refresh_bounds`](Self::refresh_bounds) without the copies, for a
    /// caller whose queries until the next refresh are all pair queries, which
    /// read the boxes instead. The boxes and spans end up as after a full
    /// refresh, and [`for_each_pair`](Self::for_each_pair) and
    /// [`for_each_self_query`](Self::for_each_self_query) run as fast.
    ///
    /// [`search`](Self::search) stays exact, but on every level this refresh
    /// visited it judges the neighbors of a query cell by the spans, as
    /// [`SpanOverflowBoundGrid`](crate::SpanOverflowBoundGrid) does and at its
    /// cost. The next [`refresh_bounds`](Self::refresh_bounds) brings back the
    /// copies by sweeping each of those levels whole, since it can no longer
    /// tell which of them fell behind.
    ///
    /// It pays where many items move. With every one of 160,000 balls moving,
    /// the refresh took 0.60 times as long and a frame of pair queries 0.94
    /// times, with 10,000 balls 0.85 and 0.98; with 1% of them moving the
    /// frame did not change. A search after it took 1.12 to 1.16 times as
    /// long with 160,000 balls and 1.26 to 1.33 with 10,000, on a Core Ultra 9
    /// 275HX with rustc 1.99 (`bench-results/r3-pair-refresh-2026-11-10.md`),
    /// measured with the 96-byte copies of that time. Compressing them has
    /// since sped up the search after a full refresh but not the one after
    /// this, so the gap is likely larger now; it was not measured again.
    pub fn refresh_for_pairs(&mut self) {
        self.refresh_levels::<false>(NodeMarks::is_sparse);
    }

    /// [`refresh_bounds`](Self::refresh_bounds), visiting only the marked
    /// nodes of a level where `sparse(marked, occupied, all nodes)` holds and sweeping
    /// it row by row elsewhere.
    fn refresh_with(&mut self, sparse: impl Fn(usize, usize, usize) -> bool) {
        self.refresh_levels::<true>(sparse);
    }

    /// The refreshes of every level: with `COPIES` that of
    /// [`refresh_bounds`](Self::refresh_bounds), without it that of
    /// [`refresh_for_pairs`](Self::refresh_for_pairs).
    fn refresh_levels<const COPIES: bool>(&mut self, sparse: impl Fn(usize, usize, usize) -> bool) {
        for level in 1..=MAX_LEVEL as usize {
            let touched = std::mem::take(&mut self.marks[level]).touched;
            let lines = &mut self.lines[level];
            if !touched && !(COPIES && lines.copies_behind) {
                if lines.copies_behind {
                    // Nothing changed since the last refresh for pairs, which
                    // the `take` above must not undo.
                    self.marks[level] = Marks::COPIES_BEHIND;
                }
                continue;
            }
            let grid = &mut self.layers[level];
            let halo = &mut self.halo[level];
            if self.geometry[level].is_empty() {
                // Every node reset its box as it emptied; only the spans and
                // copies remain to reset.
                lines.clear();
                lines.clear_marks();
                halo.fill(NO_HALO);
                lines.copies_behind = false;
                continue;
            }
            let marked = lines.marks.count(&lines.row_touched);
            lines.dense_marks = NodeMarks::is_dense(marked, lines.marks.occupied());
            let rebuild_copies = COPIES && lines.copies_behind;
            if !rebuild_copies && sparse(marked, lines.marks.occupied(), grid.cols() * grid.rows())
            {
                let size = (grid.cols(), grid.rows());
                let LineSpan {
                    columns,
                    rows,
                    row_touched,
                    marks,
                    ..
                } = lines;
                marks.refresh(grid, columns, rows, row_touched, |node, bounds| {
                    if COPIES {
                        push_halo(halo, size, node, bounds)
                    }
                });
                row_touched.fill(false);
            } else {
                if rebuild_copies {
                    // Every row is then rebuilt and pulls its copies.
                    lines.row_touched.fill(true);
                }
                refresh_rows::<COPIES, T>(grid, lines, halo);
            }
            lines.copies_behind = !COPIES;
            if !COPIES {
                self.marks[level] = Marks::COPIES_BEHIND;
            }
        }
    }

    /// Record that an item was stored on `level`, so a box there may have
    /// grown past the copies its neighbors hold. `grow_lines`, which every
    /// store goes through, marks the row and the node.
    #[inline]
    fn mark_stored(&mut self, level: usize) {
        self.marks[level] = Marks::STORED;
    }

    /// Record that an item was taken from node `(x, y)` of `level`, so a box
    /// or span there may be looser than its items. Run only after
    /// [`take_entry`](Self::take_entry), which indexes the node with bounds
    /// checks.
    #[inline]
    fn mark_taken(&mut self, level: usize, x: usize, y: usize) {
        self.marks[level].touched = true;
        // SAFETY: `take_entry` indexed node `(x, y)` of this level's grid,
        // whose columns and rows the spans match.
        unsafe { self.lines[level].touch((x, y)) };
    }

    /// Widen the spans of the column and row of `coord` to cover `bounds`
    /// stored there, and mark the row and the node touched. Every store runs
    /// it. Once
    /// `cover` and the spans grew with `lower` and `higher`, the inliner left
    /// its call in `replace_bounds` out of line; that cost nothing measurable
    /// here, unlike in [`SpanOverflowBoundGrid`](crate::SpanOverflowBoundGrid),
    /// but it is forced inline as there.
    #[inline(always)]
    fn grow_lines(&mut self, bounds: &Rectangle, (level, x, y): Coord) {
        self.lines[level].grow((x, y), bounds);
    }

    /// Return the level and coordinates of the node that stores an item with
    /// `bounds`.
    pub fn position(&self, bounds: &Rectangle) -> Coord {
        let center = bounds.get_center();
        if !self.world_bounds.contains_point(center) {
            return (1, 0, 0);
        }

        let (width, height) = (bounds.get_width(), bounds.get_height());
        let level = self.level_for_size(width, height);
        // A size that `level_for_size` left out of the next finer level though
        // it fits a node there lies within the margin of that level's limit.
        if level < MAX_LEVEL as usize && self.geometry[level + 1].fits_node(width, height) {
            return self.position_near_limit(bounds, center, level);
        }
        let (x, y) = self.cell_containing(level, center);
        (level, x, y)
    }

    /// [`position`](Self::position) of an item within the margin of the size
    /// limit of the level finer than `level`: on that level if it stays within
    /// reach there, otherwise on `level`, far below the limit.
    #[cold]
    #[inline(never)]
    fn position_near_limit(&self, bounds: &Rectangle, center: (f64, f64), level: usize) -> Coord {
        let finer = level + 1;
        let (x, y) = self.cell_containing(finer, center);
        let origin = self.root_bounds.get_min();
        if within_reach(bounds, 0.0, origin, self.inverse_node_size[finer], (x, y)) {
            return (finer, x, y);
        }
        let (x, y) = self.cell_containing(level, center);
        (level, x, y)
    }

    /// Level whose nodes fit an object of the given size.
    ///
    /// Level `L` holds objects that fit across the root at least `2^(L - 1)`
    /// times but fewer than `2^L` times, clamped to `1..=MAX_LEVEL`.
    /// [`LayerGeometry`] caches the matching size limits of each level.
    fn level_for_size(&self, width: f64, height: f64) -> usize {
        // The bit length of `(root / size) as usize` on each axis, from the
        // exponents rather than by dividing.
        let level_x = size_level(self.size_root.0, width);
        let level_y = size_level(self.size_root.1, height);
        level_x.min(level_y).clamp(1, MAX_LEVEL as i64) as usize
    }

    /// Level to search from: that of the query's size, or, if that lies within
    /// the margin of the limit of the next finer level, that finer level if the
    /// query stays within reach there, so that a query of exactly the node size
    /// keeps its ancestor walk.
    #[inline]
    fn query_level(&self, bounds: &Rectangle) -> usize {
        let (width, height) = (bounds.get_width(), bounds.get_height());
        let level = self.level_for_size(width, height);
        if level < MAX_LEVEL as usize && self.geometry[level + 1].fits_node(width, height) {
            return self.query_level_near_limit(bounds, level);
        }
        level
    }

    #[cold]
    #[inline(never)]
    fn query_level_near_limit(&self, bounds: &Rectangle, level: usize) -> usize {
        let finer = level + 1;
        let cell = self.cell_containing(finer, bounds.get_center());
        let origin = self.root_bounds.get_min();
        if within_reach(bounds, 0.0, origin, self.inverse_node_size[finer], cell) {
            finer
        } else {
            level
        }
    }

    /// Cell of `level` whose area contains `point`, clamped into the grid.
    fn cell_containing(&self, level: usize, point: (f64, f64)) -> (usize, usize) {
        let grid = &self.layers[level];
        let (inverse_width, inverse_height) = self.inverse_node_size[level];
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let x = (((point.0 - origin_x) * inverse_width) as usize).min(grid.cols() - 1);
        let y = (((point.1 - origin_y) * inverse_height) as usize).min(grid.rows() - 1);
        (x, y)
    }

    fn get_root(&self) -> &HaloOverflowBoundGridNode<T> {
        unsafe { self.layers.get_unchecked(1).get_unchecked(0, 0) }
    }

    /// Insert an item into the grid and return the handle that identifies
    /// it for [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord @ (level, x, y) = self.position(&bounds);
        self.grow_lines(&bounds, coord);
        let grid = &mut self.layers[level];
        let node = &mut grid[y][x];
        let filled = node.items.is_empty();
        node.grow(&bounds);
        let handle = self
            .slots
            .allocate(GridLocation::new(coord, node.items.len()));
        node.items.push(Entry {
            bounds,
            item,
            handle,
        });
        self.mark_stored(level);
        if filled {
            self.lines[level].marks.filled();
        }
        self.occupy(level);
        handle
    }

    /// Remove an item from the grid.
    ///
    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let (level, x, y) = location.coord();
        let entry = self.take_entry(level, x, y, location.index as usize);
        debug_assert_eq!(entry.handle, handle);
        self.slots.free(handle);
        self.mark_taken(level, x, y);
        self.vacate(level);
    }

    /// Update the bounds of an item and, if necessary, its position in the grid.
    ///
    /// Call it only for items that moved or changed size. It does not compare
    /// the bounds with the old ones, so an unchanged item still marks its node
    /// and the next [`refresh_bounds`](Self::refresh_bounds) redoes that node.
    ///
    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let prev_coord @ (level, x, y) = location.coord();
        let i = location.index as usize;

        if self.stays_in_node(&bounds, prev_coord) {
            self.replace_bounds(prev_coord, i, bounds);
            return;
        }

        let curt_coord = self.position(&bounds);
        if curt_coord == prev_coord {
            self.replace_bounds(prev_coord, i, bounds);
            return;
        }

        let (nl, nx, ny) = curt_coord;
        let entry = self.take_entry(level, x, y, i);
        self.grow_lines(&bounds, curt_coord);
        let grid = &mut self.layers[nl];
        let new_node = &mut grid[ny][nx];
        let filled = new_node.items.is_empty();
        new_node.grow(&bounds);
        *self.slots.get_mut(handle) = GridLocation::new(curt_coord, new_node.items.len());
        new_node.items.push(Entry { bounds, ..entry });
        self.mark_stored(nl);
        if filled {
            self.lines[nl].marks.filled();
        }
        self.mark_taken(level, x, y);
        self.occupy(nl);
        self.vacate(level);
    }

    /// Overwrite the bounds of an entry that stays in its node, and grow the
    /// node's box and the spans of its column and row to cover them.
    fn replace_bounds(&mut self, coord: Coord, index: usize, bounds: Rectangle) {
        self.grow_lines(&bounds, coord);
        let (level, x, y) = coord;
        let node = &mut self.layers[level][y][x];
        node.grow(&bounds);
        node.items[index].bounds = bounds;
        self.mark_stored(level);
    }

    /// Count one more item on `level`.
    fn occupy(&mut self, level: usize) {
        self.geometry[level].object_count += 1;
        self.occupied_levels |= 1 << level;
    }

    /// Count one item fewer on `level`, and clear its bit, without a branch,
    /// if it is now empty.
    fn vacate(&mut self, level: usize) {
        self.geometry[level].object_count -= 1;
        let emptied = self.geometry[level].is_empty() as u64;
        self.occupied_levels &= !(emptied << level);
    }

    /// The coarsest level below the root that holds items, or `MAX_LEVEL + 1`
    /// if none does.
    fn coarsest_level(&self) -> usize {
        // Level zero is unused and the root is searched separately.
        ((self.occupied_levels & !0b11).trailing_zeros() as usize).min(MAX_LEVEL as usize + 1)
    }

    /// Remove the entry at `index` from a node and return it. The entry that
    /// fills the gap has its slot updated, so handles stay valid without a
    /// lookup.
    fn take_entry(&mut self, level: usize, x: usize, y: usize, index: usize) -> Entry<T> {
        let node = &mut self.layers[level][y][x];
        let entry = node.items.swap_remove(index);
        if let Some(moved) = node.items.get(index) {
            self.slots.get_mut(moved.handle).index = index as u32;
        }
        if node.items.is_empty() {
            // Nothing is left to cover; searches skip the node immediately.
            // The refresh can then no longer tell whether the box held an end
            // of its column's or row's span, so both are rebuilt.
            node.bounds = EMPTY;
            // SAFETY: node `(x, y)` of this level was just indexed with
            // bounds checks, and the marks match the level's grid.
            unsafe { self.lines[level].marks.emptied((x, y)) };
        }
        entry
    }

    /// Check cheaply whether `position` would still return `coord` for
    /// `bounds`.
    ///
    /// [`LayerGeometry::matches_layer`] and [`LayerGeometry::matches_cell`]
    /// explain why `position` returns `coord` when this holds. A center near a
    /// cell edge is left to `position`, even if it would return `coord`.
    fn stays_in_node(&self, bounds: &Rectangle, coord: Coord) -> bool {
        let (level, x, y) = coord;
        let geometry = &self.geometry[level];
        if !geometry.has_update_shortcut() {
            return false;
        }

        let center = bounds.get_center();
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let relative_center = (center.0 - origin_x, center.1 - origin_y);
        let is_finest_layer = level == MAX_LEVEL as usize;
        geometry.matches_layer(bounds.get_width(), bounds.get_height(), is_finest_layer)
            && geometry.matches_cell(relative_center, (x, y))
            && self.world_bounds.contains_point(center)
    }

    /// Call `callback` with every stored item and each stored item whose bounds
    /// intersect its own, the item itself included where its bounds meet
    /// themselves: what searching with the bounds of every item reports. The
    /// results come from the walk of [`for_each_pair`](Self::for_each_pair),
    /// each pair in both orders, so a result across levels is found by the
    /// coarser item's scan of the finer levels and no item walks up to the
    /// coarser ones; the results of one query are not reported together. The walk
    /// visits every cell down to the finest level that holds items, so where
    /// cells far outnumber items, searching item by item is faster. On the
    /// search bench's distribution, the `self_query` bench group measured 0.27
    /// ms, against 0.79 ms for the previous implementation, which searched with
    /// the bounds of every item in turn, cell by cell.
    pub fn for_each_self_query(&self, callback: impl FnMut(T, T)) {
        self.pair_walk::<true>(callback);
    }

    /// Call `callback` once for every pair of distinct stored items whose
    /// bounds intersect, with the two items in either order. On its own level
    /// an item reads only the items after it in its own node and the nodes
    /// after its own in row-major order, that is the neighbor to the right
    /// and those in the row below, each only if its box reaches the item;
    /// items in the root pair up with the root items after them. Below
    /// its level an item scans the finer levels as a search does, so a pair
    /// across levels is found once, by the item on the coarser level, and no
    /// item looks above its own level. The nodes are visited depth first
    /// through the levels and in Morton order within each, so that
    /// consecutive items read nearby cells, down to the finest level that
    /// holds items.
    ///
    /// Leaving each cross-level pair to its finer item instead, which walked
    /// the ancestor path as a search does, took 0.56 ms against 0.28 ms on the
    /// mixed distribution: nearly all items lie on the finest level, and each
    /// of them paid for a walk over the occupied levels above it.
    pub fn for_each_pair(&self, callback: impl FnMut(T, T)) {
        self.pair_walk::<false>(callback);
    }

    /// The walk of [`for_each_pair`](Self::for_each_pair) and, with
    /// `SELF_QUERY`, of [`for_each_self_query`](Self::for_each_self_query),
    /// which reports each pair in both orders and each item with itself where
    /// its bounds meet themselves. Inlined into both, so that each compiles
    /// to a walk of its own.
    #[inline(always)]
    fn pair_walk<const SELF_QUERY: bool>(&self, mut callback: impl FnMut(T, T)) {
        let layers = &self.layers;
        // Levels finer than the finest one holding items need no visit.
        let Some(finest) = (1..=MAX_LEVEL as usize)
            .rev()
            .find(|&level| !self.geometry[level].is_empty())
        else {
            return;
        };
        for_each_cell_hierarchically(
            finest,
            |level| (layers[level].cols(), layers[level].rows()),
            |level, x, y| {
                if !layers[level][y][x].items.is_empty() {
                    self.pair_node::<SELF_QUERY>(level, (x, y), finest, &mut callback);
                }
            },
        );
    }

    /// The pair walk at node `(x, y)` of `level`, which holds items: each
    /// item with the items after it in the node, with those of the four
    /// neighbors after the node in row-major order whose boxes reach it, and,
    /// on a level coarser than `finest`, with the items of the finer levels
    /// by its scan.
    ///
    /// The four neighbors are judged by their own boxes as they are now, read
    /// once for the node, not by the copies of them that the node holds: the
    /// walk comes to these nodes anyway, and a box covers its items at all
    /// times, so the walk reads neither the copies nor the stale marks and
    /// never falls back to the spans. Reading the node's copies and checking
    /// for stale ones once per item instead, the pair query took 1.07 times
    /// as long for 10,000 balls and 1.16 to 1.18 times for 160,000, whose
    /// finest level had 6 MB of copies, and before a refresh 1.15 to 1.26
    /// times, on a Core Ultra 9 275HX with rustc 1.99, with a copies record
    /// of 96 bytes (`bench-results/halo-review-2026-11-09.md`).
    ///
    /// Not inlined: inlined into the walk it measured 3% faster on these
    /// scenes, but with 500 to 2,000 balls on 9 levels, where nearly every
    /// node the walk visits holds no items, it took 1.3 to 1.5 times as long.
    #[inline(never)]
    fn pair_node<const SELF_QUERY: bool>(
        &self,
        level: usize,
        (x, y): (usize, usize),
        finest: usize,
        callback: &mut impl FnMut(T, T),
    ) {
        let grid = &self.layers[level];
        let node = &grid[y][x];
        let (has_left, has_right) = (x > 0, x + 1 < grid.cols());
        let has_below = y + 1 < grid.rows();
        // SAFETY: `(x, y)` is a node of `grid`, and a neighbor is read only
        // if the grid has it.
        let (right, below_left, below, below_right) = unsafe {
            (
                box_at(grid, has_right, x + 1, y),
                box_at(grid, has_left & has_below, x.wrapping_sub(1), y + 1),
                box_at(grid, has_below, x, y + 1),
                box_at(grid, has_right & has_below, x + 1, y + 1),
            )
        };
        // The sides that face the node: how far each neighbor's items reach
        // towards it.
        let (right, below) = (right.min_x, below.min_y);
        let below_left = [below_left.max_x, below_left.min_y];
        let below_right = [below_right.min_x, below_right.min_y];
        for (index, entry) in node.items.iter().enumerate() {
            let bounds = &entry.bounds;
            if SELF_QUERY && bounds.intersects(bounds) {
                callback(entry.item, entry.item);
            }
            let mut report = |found| {
                callback(entry.item, found);
                if SELF_QUERY {
                    callback(found, entry.item);
                }
            };
            // All four are computed up front, as in `search_3x3_by_copies`.
            let after = bounds.max_x > right;
            let down = bounds.max_y > below;
            let down_before = (bounds.min_x < below_left[0]) & (bounds.max_y > below_left[1]);
            let down_after = (bounds.max_x > below_right[0]) & (bounds.max_y > below_right[1]);
            node.pair_with_after(index, bounds, &mut report);
            // SAFETY: the box of a missing neighbor is `EMPTY`, which fails
            // every comparison above, so each node read here exists.
            unsafe {
                if after {
                    grid.get_unchecked(y, x + 1).pair_with(bounds, &mut report);
                }
                if down_before {
                    grid.get_unchecked(y + 1, x - 1)
                        .pair_with(bounds, &mut report);
                }
                if down {
                    grid.get_unchecked(y + 1, x).pair_with(bounds, &mut report);
                }
                if down_after {
                    grid.get_unchecked(y + 1, x + 1)
                        .pair_with(bounds, &mut report);
                }
            }
            if level < finest {
                self.pair_scan_down(level, bounds, &mut report);
            }
        }
    }

    /// Search from the bottom up. Call `callback` for each item found.
    ///
    /// Follows [`HierarchicalLooseGrid::search`], except that the ancestor
    /// walk reads a neighbor of the query cell only if the copies of its
    /// sides in the query cell's node reach the query (judged by the spans
    /// instead next to the changed nodes of a stale level), and the query
    /// cell's node only if the copy of its box meets the query or the level
    /// is stale; the scan drops edge columns and rows whose spans miss the
    /// query, and a node whose box misses the query is skipped without
    /// reading its items.
    ///
    /// [`HierarchicalLooseGrid::search`]: crate::HierarchicalLooseGrid::search
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        // Every comparison with NaN is false, so a query with a NaN corner
        // meets nothing, while for a NaN minimum the scans would report the
        // items of the nodes they take as inside it untested.
        if bounds.has_nan() {
            return;
        }
        let max_level = MAX_LEVEL as usize;
        let level = self.query_level(bounds);
        let (offset_x, offset_y) = self.root_bounds.get_min();

        let layers = &self.layers;
        if level != max_level {
            // The query's corners relative to the root origin, in half nodes
            // of the finest level; every level's scan range follows by shifts.
            let half = half_node_corners(
                [
                    bounds.min_x - offset_x,
                    bounds.min_y - offset_y,
                    bounds.max_x - offset_x,
                    bounds.max_y - offset_y,
                ],
                self.inverse_node_size[max_level],
            );

            layers[level + 1..=max_level]
                .iter()
                .zip(&self.geometry[level + 1..=max_level])
                .zip(&self.lines[level + 1..=max_level])
                .rev()
                .enumerate()
                .for_each(|(shift, ((grid, geometry), lines))| {
                    if geometry.is_empty() {
                        return;
                    }
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();

                    let [min_coord_x, min_coord_y, max_coord_x, max_coord_y] =
                        scan_range(half, shift as u32, (grid_width - 1, grid_height - 1));

                    // The fixed reach overshoots by about one column and row
                    // per side; drop an edge column or row whose span misses
                    // the query. Loops dropping more measured slower on large
                    // queries.
                    let column_reaches = |x: usize| {
                        let [low, high] = unsafe { *lines.columns.get_unchecked(x) };
                        high > bounds.min_x && low < bounds.max_x
                    };
                    let row_reaches = |y: usize| {
                        let [low, high] = unsafe { *lines.rows.get_unchecked(y) };
                        high > bounds.min_y && low < bounds.max_y
                    };
                    let (mut first_x, mut end_x) = (min_coord_x, max_coord_x + 1);
                    if first_x < end_x && !column_reaches(first_x) {
                        first_x += 1;
                    }
                    if end_x > first_x && !column_reaches(end_x - 1) {
                        end_x -= 1;
                    }
                    let (mut first_y, mut end_y) = (min_coord_y, max_coord_y + 1);
                    if first_y < end_y && !row_reaches(first_y) {
                        first_y += 1;
                    }
                    if end_y > first_y && !row_reaches(end_y - 1) {
                        end_y -= 1;
                    }

                    // Scan the remaining range.
                    for y in first_y..end_y {
                        for x in first_x..end_x {
                            let node = unsafe { grid.get_unchecked(y, x) };
                            if node.items.is_empty() {
                                continue;
                            }

                            if y > min_coord_y + 1
                                && y < max_coord_y - 1
                                && x > min_coord_x + 1
                                && x < max_coord_x - 1
                            {
                                // The node lies within the query, loose bounds
                                // included, so its items need no intersection test.
                                node.iter_items(&mut callback);
                            } else {
                                node.search_items(bounds, &mut callback);
                            }
                        }
                    }
                });
        }

        if level == 1 {
            self.get_root().search_items(bounds, &mut callback);
            return;
        }

        self.search_up_3x3(level, bounds, &mut callback);
    }

    /// Scan the levels finer than `level` over `bounds` for
    /// [`for_each_pair`](Self::for_each_pair) as [`search`](Self::search)
    /// scans the levels finer than a query's, a copy so that pair queries and
    /// searches share no code.
    fn pair_scan_down(&self, level: usize, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        // As in `search`: a NaN corner meets nothing, while for a NaN minimum
        // the scan would report the items of the nodes it takes as inside
        // `bounds` untested.
        if bounds.has_nan() {
            return;
        }
        let max_level = MAX_LEVEL as usize;
        let (offset_x, offset_y) = self.root_bounds.get_min();
        let layers = &self.layers;
        // The corners of `bounds` relative to the root origin, in half nodes
        // of the finest level; every level's scan range follows by shifts.
        let half = half_node_corners(
            [
                bounds.min_x - offset_x,
                bounds.min_y - offset_y,
                bounds.max_x - offset_x,
                bounds.max_y - offset_y,
            ],
            self.inverse_node_size[max_level],
        );

        layers[level + 1..=max_level]
            .iter()
            .zip(&self.geometry[level + 1..=max_level])
            .zip(&self.lines[level + 1..=max_level])
            .rev()
            .enumerate()
            .for_each(|(shift, ((grid, geometry), lines))| {
                if geometry.is_empty() {
                    return;
                }
                let grid_width = grid.cols();
                let grid_height = grid.rows();

                let [min_coord_x, min_coord_y, max_coord_x, max_coord_y] =
                    scan_range(half, shift as u32, (grid_width - 1, grid_height - 1));

                // The fixed reach overshoots by about one column and row
                // per side; drop an edge column or row whose span misses
                // the query. Loops dropping more measured slower on large
                // queries.
                let column_reaches = |x: usize| {
                    let [low, high] = unsafe { *lines.columns.get_unchecked(x) };
                    high > bounds.min_x && low < bounds.max_x
                };
                let row_reaches = |y: usize| {
                    let [low, high] = unsafe { *lines.rows.get_unchecked(y) };
                    high > bounds.min_y && low < bounds.max_y
                };
                let (mut first_x, mut end_x) = (min_coord_x, max_coord_x + 1);
                if first_x < end_x && !column_reaches(first_x) {
                    first_x += 1;
                }
                if end_x > first_x && !column_reaches(end_x - 1) {
                    end_x -= 1;
                }
                let (mut first_y, mut end_y) = (min_coord_y, max_coord_y + 1);
                if first_y < end_y && !row_reaches(first_y) {
                    first_y += 1;
                }
                if end_y > first_y && !row_reaches(end_y - 1) {
                    end_y -= 1;
                }

                // Scan the remaining range.
                for y in first_y..end_y {
                    for x in first_x..end_x {
                        let node = unsafe { grid.get_unchecked(y, x) };
                        if node.items.is_empty() {
                            continue;
                        }

                        if y > min_coord_y + 1
                            && y < max_coord_y - 1
                            && x > min_coord_x + 1
                            && x < max_coord_x - 1
                        {
                            // The node lies within `bounds`, loose bounds
                            // included, so its items need no intersection test.
                            node.pair_all(callback);
                        } else {
                            node.pair_with(bounds, callback);
                        }
                    }
                }
            });
    }

    fn search_up_3x3(&self, level: usize, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        let (mut x, mut y) = self.cell_containing(level, bounds.get_center());
        // Rounded once per query, not on every level.
        let query = outwards(bounds);

        // The root is searched below; levels coarser than this hold nothing.
        let coarsest = self.coarsest_level().min(level + 1);
        self.layers[coarsest..=level]
            .iter()
            .zip(&self.geometry[coarsest..=level])
            .zip(&self.lines[coarsest..=level])
            .zip(&self.marks[coarsest..=level])
            .zip(&self.halo[coarsest..=level])
            .rev()
            .for_each(|((((grid, geometry), lines), marks), halo)| {
                if !geometry.is_empty() {
                    // The unchecked reads of both searches rely on the walk
                    // staying in the grid.
                    debug_assert!(
                        x < grid.cols() && y < grid.rows(),
                        "walk cell ({x}, {y}) outside {}x{}",
                        grid.cols(),
                        grid.rows()
                    );
                    // Copies can be stale only on a stale level, so the
                    // marks are read only there.
                    if marks.stale && lines.around_changed((x, y), (grid.cols(), grid.rows())) {
                        search_3x3_by_spans(grid, lines, (x, y), bounds, callback);
                    } else {
                        search_3x3_by_copies(
                            grid,
                            halo,
                            (x, y),
                            bounds,
                            query,
                            marks.stale,
                            callback,
                        );
                    }
                }
                // Advance along the ancestor path even on an empty level.
                x >>= 1;
                y >>= 1;
            });

        self.get_root().search_items(bounds, callback);
    }
}

/// Search node `(x, y)` of `grid`, unless the copy of its box misses the
/// query on a level that is not `stale`, and those of its eight neighbors
/// whose copies in `halo` reach the query, where no neighbor's copies are
/// behind its box. `query` is `bounds` rounded outwards, by [`outwards`].
#[inline]
fn search_3x3_by_copies<T: Copy>(
    grid: &Grid<HaloOverflowBoundGridNode<T>>,
    halo: &[Halo],
    (x, y): (usize, usize),
    bounds: &Rectangle,
    [min_x, min_y, max_x, max_y]: [f32; 4],
    stale: bool,
    callback: &mut impl FnMut(T),
) {
    // SAFETY: `(x, y)` is a node of `grid`, and `halo` holds one entry
    // per node in the same order. An edge node keeps the initial copies for
    // its missing neighbors, which fail every comparison, so the indices
    // below stay inside the grid.
    unsafe {
        let node = grid.get_unchecked(y, x);
        let halo = halo.get_unchecked(y * grid.cols() + x);

        // Whether each neighbor's items may reach the query, per the copies
        // in this node. All eight are computed up front; short-circuit
        // branches here measured search 50% slower on a scene where the
        // outcomes vary.
        let before = min_x < halo.left;
        let after = max_x > halo.right;
        let up = min_y < halo.top;
        let down = max_y > halo.bottom;
        let up_before = (min_x < halo.top_left[0]) & (min_y < halo.top_left[1]);
        let up_after = (max_x > halo.top_right[0]) & (min_y < halo.top_right[1]);
        let down_before = (min_x < halo.bottom_left[0]) & (max_y > halo.bottom_left[1]);
        let down_after = (max_x > halo.bottom_right[0]) & (max_y > halo.bottom_right[1]);
        debug_assert!(!(before | up_before | down_before) || x != 0);
        debug_assert!(!(after | up_after | down_after) || x + 1 != grid.cols());
        debug_assert!(!(up | up_before | up_after) || y != 0);
        debug_assert!(!(down | down_before | down_after) || y + 1 != grid.rows());

        // The copy of the node's own box is behind the box on a stale level,
        // where a store may have grown the box without any of the marks
        // around the node, which leave out the node itself. Where no store
        // came, the box can only have shrunk since the copy was taken.
        let own = (min_x < halo.own[2])
            & (halo.own[0] < max_x)
            & (min_y < halo.own[3])
            & (halo.own[1] < max_y);
        if own | stale {
            node.search_items(bounds, callback);
        }
        if up_before {
            let node = grid.get_unchecked(y - 1, x - 1);
            node.search_items(bounds, callback);
        }
        if up {
            let node = grid.get_unchecked(y - 1, x);
            node.search_items(bounds, callback);
        }
        if up_after {
            let node = grid.get_unchecked(y - 1, x + 1);
            node.search_items(bounds, callback);
        }
        if before {
            let node = grid.get_unchecked(y, x - 1);
            node.search_items(bounds, callback);
        }
        if after {
            let node = grid.get_unchecked(y, x + 1);
            node.search_items(bounds, callback);
        }
        if down_before {
            let node = grid.get_unchecked(y + 1, x - 1);
            node.search_items(bounds, callback);
        }
        if down {
            let node = grid.get_unchecked(y + 1, x);
            node.search_items(bounds, callback);
        }
        if down_after {
            let node = grid.get_unchecked(y + 1, x + 1);
            node.search_items(bounds, callback);
        }
    }
}

/// Search node `(x, y)` of `grid` and those of its eight neighbors whose
/// column and row spans reach the query, where the copies of row `y` may be
/// behind the boxes; [`SpanOverflowBoundGrid`] does this on every level.
///
/// [`SpanOverflowBoundGrid`]: crate::SpanOverflowBoundGrid
#[inline]
fn search_3x3_by_spans<T: Copy>(
    grid: &Grid<HaloOverflowBoundGridNode<T>>,
    lines: &LineSpan,
    (x, y): (usize, usize),
    bounds: &Rectangle,
    callback: &mut impl FnMut(T),
) {
    let grid_width = grid.cols();
    let grid_height = grid.rows();

    // SAFETY: `(x, y)` is a node of `grid`, and each neighbor is read only
    // after testing that it exists.
    unsafe {
        // Whether the span of each neighboring column and row reaches the
        // query. Clamped indices read an in-bounds line, whose result the
        // edge tests discard. Short-circuit branches here measured search
        // 50% slower on a scene where these outcomes vary.
        let above = lines.rows.get_unchecked(y.saturating_sub(1))[1];
        let below = lines.rows.get_unchecked((y + 1).min(grid_height - 1))[0];
        let left_of = lines.columns.get_unchecked(x.saturating_sub(1))[1];
        let right_of = lines.columns.get_unchecked((x + 1).min(grid_width - 1))[0];
        let up = (y != 0) & (bounds.min_y < above);
        let down = (y + 1 != grid_height) & (bounds.max_y > below);
        let before = (x != 0) & (bounds.min_x < left_of);
        let after = (x + 1 != grid_width) & (bounds.max_x > right_of);

        if up {
            if before {
                let node = grid.get_unchecked(y - 1, x - 1);
                node.search_items(bounds, callback);
            }
            {
                let node = grid.get_unchecked(y - 1, x);
                node.search_items(bounds, callback);
            }
            if after {
                let node = grid.get_unchecked(y - 1, x + 1);
                node.search_items(bounds, callback);
            }
        }
        {
            if before {
                let node = grid.get_unchecked(y, x - 1);
                node.search_items(bounds, callback);
            }
            {
                let node = grid.get_unchecked(y, x);
                node.search_items(bounds, callback);
            }
            if after {
                let node = grid.get_unchecked(y, x + 1);
                node.search_items(bounds, callback);
            }
        }
        if down {
            if before {
                let node = grid.get_unchecked(y + 1, x - 1);
                node.search_items(bounds, callback);
            }
            {
                let node = grid.get_unchecked(y + 1, x);
                node.search_items(bounds, callback);
            }
            if after {
                let node = grid.get_unchecked(y + 1, x + 1);
                node.search_items(bounds, callback);
            }
        }
    }
}

/// Changes to a level since its last refresh.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Marks {
    /// An item was stored on, replaced on or taken from the level, so a box
    /// or span may be looser than its items and the refresh must visit it.
    touched: bool,
    /// An item was stored on or replaced on the level, so a box may have
    /// grown past the copies of its sides held by the neighbors: until the
    /// refresh, search judges the neighbors of a query cell next to a marked
    /// node by the spans instead. Also set where a refresh for pairs left the
    /// copies behind, as [`LineSpan::copies_behind`] records.
    stale: bool,
}

impl Marks {
    /// Marks of a level an item was just stored on.
    const STORED: Self = Self {
        touched: true,
        stale: true,
    };

    /// Marks of a level right after a refresh for pairs: the boxes and spans
    /// are up to date, the copies are not.
    const COPIES_BEHIND: Self = Self {
        touched: false,
        stale: true,
    };
}

/// Extent of the items of every column along x and every row along y of one
/// level since the last refresh, as `[min, max]`; `[+infinity, -infinity]`
/// where no item has been, which meets no query.
struct LineSpan {
    columns: Box<[[f64; 2]]>,
    rows: Box<[[f64; 2]]>,
    /// Whether an item was stored in or taken from each row since the last
    /// refresh. A refresh that sweeps the level row by row shrinks the boxes
    /// and rebuilds the spans of these rows and pulls the copies of them and
    /// their neighbor rows again; until the refresh, on a stale level with
    /// [`dense_marks`](Self::dense_marks), search avoids the copies of a
    /// touched row and of the rows next to it.
    row_touched: Box<[bool]>,
    /// The same per node, for a refresh that visits only those and for the
    /// walk of a search before a refresh.
    marks: NodeMarks,
    /// Whether the last refresh found the level's marks dense, by
    /// [`NodeMarks::is_dense`].
    dense_marks: bool,
    /// A refresh for pairs left the level's copies behind its boxes, wherever
    /// those changed: search goes by the spans throughout the level, and the
    /// next full refresh pulls the copies of every row. Here and not in
    /// [`Marks`], which every store overwrites as a whole.
    copies_behind: bool,
}

impl LineSpan {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            columns: vec![NOWHERE; cols].into_boxed_slice(),
            rows: vec![NOWHERE; rows].into_boxed_slice(),
            row_touched: vec![false; rows].into_boxed_slice(),
            marks: NodeMarks::new(cols, rows),
            dense_marks: false,
            copies_behind: false,
        }
    }

    /// Mark row `y` and node `(x, y)` touched, without bounds checks, as
    /// every store and take runs this.
    ///
    /// # Safety
    ///
    /// `x` and `y` must be a column and a row of the level.
    #[inline]
    unsafe fn touch(&mut self, (x, y): (usize, usize)) {
        debug_assert!(y < self.row_touched.len());
        // SAFETY: by the caller's guarantee.
        unsafe {
            *self.row_touched.get_unchecked_mut(y) = true;
            self.marks.mark((x, y));
        }
    }

    /// Clear the marks of the rows and the nodes.
    fn clear_marks(&mut self) {
        self.row_touched.fill(false);
        self.marks.clear();
    }

    /// Whether row `y` or a row next to it is touched, the condition under
    /// which the copies of row `y` may be behind the boxes they stand for.
    #[inline]
    fn near_touched(&self, y: usize) -> bool {
        let rows = &self.row_touched;
        let last = rows.len() - 1;
        debug_assert!(y <= last);
        // SAFETY: `y` is a row, and the clamped neighbors are rows as well.
        unsafe {
            *rows.get_unchecked(y)
                | *rows.get_unchecked(y.saturating_sub(1))
                | *rows.get_unchecked((y + 1).min(last))
        }
    }

    /// Whether the copies that a search from node `(x, y)` of a grid of `size`
    /// columns and rows reads may be behind the boxes they stand for.
    #[inline(always)]
    fn around_changed(&self, (x, y): (usize, usize), size: (usize, usize)) -> bool {
        if self.copies_behind {
            return true;
        }
        if self.dense_marks {
            return self.near_touched(y);
        }
        debug_assert!(x < size.0 && y < size.1);
        // SAFETY: the walks stay in the grid, so `(x, y)` is one of its nodes.
        unsafe { self.marks.any_around((x, y), size) }
    }

    /// Widen column `x` and row `y` to cover `bounds` stored there, and mark
    /// the row and the node touched.
    #[inline]
    fn grow(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.widen((x, y), bounds);
        // SAFETY: `widen` indexed column `x` and row `y` with bounds checks.
        unsafe { self.touch((x, y)) };
    }

    /// Widen column `x` and row `y` to cover `bounds` without marking the
    /// row; the refresh does this for every box of a touched row.
    #[inline]
    fn widen(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.widen_column(x, bounds);
        let row = &mut self.rows[y];
        row[0] = lower(row[0], bounds.min_y);
        row[1] = higher(row[1], bounds.max_y);
    }

    /// Widen column `x` alone to cover `bounds`.
    #[inline]
    fn widen_column(&mut self, x: usize, bounds: &Rectangle) {
        let column = &mut self.columns[x];
        column[0] = lower(column[0], bounds.min_x);
        column[1] = higher(column[1], bounds.max_x);
    }

    fn clear(&mut self) {
        self.columns.fill(NOWHERE);
        self.rows.fill(NOWHERE);
    }
}

/// Sides of the boxes of the eight neighbors of a node that face the node,
/// in world coordinates: how far each neighbor's items reach towards it, and
/// the node's own box. The initial values fail every comparison of the walk,
/// so the copies never select a node that held no items at the last refresh,
/// and the slots of an edge node for its missing neighbors never change.
///
/// Absolute coordinates rather than distances past the cell edges, so that
/// the refresh copies box sides as they are and the walk compares them with
/// the query directly, with no cell edges. They are `f32`, each rounded away
/// from the items of its box, and the walk compares them with a query rounded
/// outwards by [`outwards`], so that wherever the exact values would select a
/// node, so do these. Their precision follows the distance from the origin,
/// not the cell size, so a world whose cells are small against that distance
/// (metres from a far map origin with 1 m cells, where `f32` steps by half a
/// metre) loses pruning, never items; how much was not measured.
///
/// A record fills one cache line, so the walk reads one line per level and,
/// where the copies say so, nothing else there. Against the twelve `f64` it
/// held before, in 96 bytes and without the own box, a frame of region
/// queries after the refresh took 0.91 to 0.93 times as long with 10,000
/// balls and 0.91 to 0.94 with 160,000, 1% to all of them moving. Regions
/// queried before the refresh took 0.96 and 0.95 times with 1% moving, 1.01
/// and 1.00 with all moving. With all moving the conversions make the refresh
/// 1.5 times as long with 10,000 balls and 1.04 with 160,000, so a frame of
/// pair queries after [`refresh_bounds`] took 1.05 and 1.00 times as long;
/// [`refresh_for_pairs`] writes no copies. All on a Core Ultra 9 275HX with
/// rustc 1.99. In the experiment that chose this record, the twelve `f64`
/// plus the own box in two lines took 0.95 and 1.05 times, twelve `f32`
/// alone 1.00 and 0.95 times. On a Ryzen 7 9800X3D with rustc 1.99, whose
/// cache is larger, the frames of region queries after the refresh took 0.92
/// to 0.95 and 0.95 to 0.97 times, before it with all moving 1.03 and 1.02,
/// and the refresh with all moving 1.44 times at both sizes
/// (`bench-results/m4-r8-halo-record-2026-11-10.md`).
///
/// [`refresh_bounds`]: HaloOverflowBoundGrid::refresh_bounds
/// [`refresh_for_pairs`]: HaloOverflowBoundGrid::refresh_for_pairs
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(align(64))]
struct Halo {
    /// At least the largest `max_x` of the items of the left neighbor
    /// `(x - 1, y)`.
    left: f32,
    /// At most the smallest `min_x` of the items of the right neighbor
    /// `(x + 1, y)`.
    right: f32,
    /// At least the largest `max_y` of the items of the neighbor above,
    /// `(x, y - 1)`.
    top: f32,
    /// At most the smallest `min_y` of the items of the neighbor below,
    /// `(x, y + 1)`.
    bottom: f32,
    /// The largest `max_x` and `max_y` of the items of `(x - 1, y - 1)`,
    /// rounded up.
    top_left: [f32; 2],
    /// The smallest `min_x` of the items of `(x + 1, y - 1)` rounded down,
    /// and their largest `max_y` rounded up.
    top_right: [f32; 2],
    /// The largest `max_x` of the items of `(x - 1, y + 1)` rounded up, and
    /// their smallest `min_y` rounded down.
    bottom_left: [f32; 2],
    /// The smallest `min_x` and `min_y` of the items of `(x + 1, y + 1)`,
    /// rounded down.
    bottom_right: [f32; 2],
    /// The node's own box by [`outwards`].
    own: [f32; 4],
}

// A record is one cache line; one field more would make it two.
const _: () = assert!(std::mem::size_of::<Halo>() == 64);

/// Copies for eight neighbors without items and a node without any, that is,
/// with [`EMPTY`] boxes.
const NO_HALO: Halo = Halo::of_boxes([&EMPTY; 8], &EMPTY);

/// The smallest `f32` not below `x`. The conversion rounds to nearest, which
/// leaves at most one step up to take; it is taken without a branch, since
/// whether it is needed is a coin flip. An infinity stays as it is, so the
/// copies of empty boxes keep failing every comparison, and a finite `x`
/// beyond the range of `f32` becomes the infinity on its side when rounded
/// away from zero and the largest finite `f32` when rounded towards it.
#[inline(always)]
const fn f32_at_least(x: f64) -> f32 {
    let value = x as f32;
    let below = ((value as f64) < x) as i32;
    let bits = value.to_bits() as i32;
    // One step up is a larger magnitude when positive, a smaller one when
    // negative, in the integer that the bits of the magnitude form.
    let sign = bits >> 31;
    f32::from_bits(bits.wrapping_add((below ^ sign).wrapping_sub(sign)) as u32)
}

/// The largest `f32` not above `x`.
#[inline(always)]
const fn f32_at_most(x: f64) -> f32 {
    -f32_at_least(-x)
}

/// `[min_x, min_y, max_x, max_y]` of `bounds` as `f32`, each rounded away
/// from the rectangle's inside.
#[inline(always)]
const fn outwards(bounds: &Rectangle) -> [f32; 4] {
    [
        f32_at_most(bounds.min_x),
        f32_at_most(bounds.min_y),
        f32_at_least(bounds.max_x),
        f32_at_least(bounds.max_y),
    ]
}

impl Halo {
    /// Copies of the facing sides of the eight neighbors' boxes, given in
    /// the order left, right, top, bottom, top left, top right, bottom left,
    /// bottom right, and of the node's own box.
    #[inline(always)]
    const fn of_boxes(
        [
            left,
            right,
            top,
            bottom,
            top_left,
            top_right,
            bottom_left,
            bottom_right,
        ]: [&Rectangle; 8],
        own: &Rectangle,
    ) -> Self {
        Self {
            left: f32_at_least(left.max_x),
            right: f32_at_most(right.min_x),
            top: f32_at_least(top.max_y),
            bottom: f32_at_most(bottom.min_y),
            top_left: [f32_at_least(top_left.max_x), f32_at_least(top_left.max_y)],
            top_right: [f32_at_most(top_right.min_x), f32_at_least(top_right.max_y)],
            bottom_left: [
                f32_at_least(bottom_left.max_x),
                f32_at_most(bottom_left.min_y),
            ],
            bottom_right: [
                f32_at_most(bottom_right.min_x),
                f32_at_most(bottom_right.min_y),
            ],
            own: outwards(own),
        }
    }

    /// Copies of the facing sides of the current boxes of the eight
    /// neighbors of `(x, y)` in `grid`, with [`EMPTY`] standing in for the
    /// missing neighbors of an edge node, as given by the flags, and of the
    /// box of the node itself.
    ///
    /// # Safety
    ///
    /// `(x, y)` must be a node of `grid`, and each flag must be false if the
    /// grid has no node on that side of it.
    #[inline(always)]
    unsafe fn of_neighbors<T: Copy>(
        grid: &Grid<HaloOverflowBoundGridNode<T>>,
        (x, y): (usize, usize),
        [has_left, has_right]: [bool; 2],
        [has_top, has_bottom]: [bool; 2],
    ) -> Self {
        // Wrapped indices are never read, since their flag is false.
        let (left, right) = (x.wrapping_sub(1), x + 1);
        let (top, bottom) = (y.wrapping_sub(1), y + 1);
        unsafe {
            Self::of_boxes(
                [
                    box_at(grid, has_left, left, y),
                    box_at(grid, has_right, right, y),
                    box_at(grid, has_top, x, top),
                    box_at(grid, has_bottom, x, bottom),
                    box_at(grid, has_left & has_top, left, top),
                    box_at(grid, has_right & has_top, right, top),
                    box_at(grid, has_left & has_bottom, left, bottom),
                    box_at(grid, has_right & has_bottom, right, bottom),
                ],
                box_at(grid, true, x, y),
            )
        }
    }
}

/// Box of node `(x, y)` of `grid` if `present`, else [`EMPTY`].
///
/// # Safety
///
/// `(x, y)` must be a node of `grid` when `present`.
#[inline(always)]
unsafe fn box_at<T: Copy>(
    grid: &Grid<HaloOverflowBoundGridNode<T>>,
    present: bool,
    x: usize,
    y: usize,
) -> &Rectangle {
    if present {
        unsafe { &grid.get_unchecked(y, x).bounds }
    } else {
        &EMPTY
    }
}

/// Refresh one level by sweeping it row by row, as described in
/// [`HaloOverflowBoundGrid::refresh_bounds`], and clear its marks. Without
/// `COPIES` the sweep leaves out the copies.
fn refresh_rows<const COPIES: bool, T: Copy>(
    grid: &mut Grid<HaloOverflowBoundGridNode<T>>,
    lines: &mut LineSpan,
    halo: &mut [Halo],
) {
    let rows = grid.rows();
    // A column span runs across every row, so it is rebuilt from the
    // boxes of all rows; a row's own span and boxes are rebuilt only
    // if the row is touched.
    lines.columns.fill(NOWHERE);
    let shrink_row =
        |grid: &mut Grid<HaloOverflowBoundGridNode<T>>, lines: &mut LineSpan, y: usize| {
            if lines.row_touched[y] {
                lines.rows[y] = NOWHERE;
                let row_marks = lines.marks.row_ptr(y);
                for (x, node) in grid[y].iter_mut().enumerate() {
                    // SAFETY: `x` is a column of row `y`, and the sweep stores
                    // no marks.
                    if unsafe { *row_marks.add(x) } == 0 {
                        // The box still equals what the items give, since
                        // every store and take marks. A path of its own, so
                        // that the shrink below keeps the new box in registers
                        // for the widening.
                        if !node.items.is_empty() {
                            lines.widen((x, y), &node.bounds);
                        }
                        continue;
                    }
                    node.refresh();
                    if !node.items.is_empty() {
                        lines.widen((x, y), &node.bounds);
                    }
                }
            } else {
                for (x, node) in grid[y].iter().enumerate() {
                    if !node.items.is_empty() {
                        lines.widen_column(x, &node.bounds);
                    }
                }
            }
        };
    // The copies of a row read the boxes of that row and the rows on
    // both sides, so they are pulled one row behind the shrinking,
    // and only if one of these three rows is touched.
    shrink_row(grid, lines, 0);
    for y in 0..rows {
        if y + 1 < rows {
            shrink_row(grid, lines, y + 1);
        }
        if COPIES && lines.near_touched(y) {
            pull_halo_row(grid, halo, y);
        }
    }
    lines.clear_marks();
}

/// Set the copies that the eight neighbors of node `(x, y)` hold of its box
/// to the facing sides of `bounds`, its box now, and the copy it holds
/// itself: the pull of [`pull_halo_row`] reversed, for one node. Neighbors
/// outside the grid, whose copies the node has no slot in, are skipped.
fn push_halo(
    halo: &mut [Halo],
    (cols, rows): (usize, usize),
    (x, y): (usize, usize),
    bounds: &Rectangle,
) {
    let index = y * cols + x;
    let (has_left, has_right) = (x > 0, x + 1 < cols);
    let (has_top, has_bottom) = (y > 0, y + 1 < rows);
    let own = outwards(bounds);
    let [min_x, min_y, max_x, max_y] = own;
    halo[index].own = own;
    if has_right {
        halo[index + 1].left = max_x;
    }
    if has_left {
        halo[index - 1].right = min_x;
    }
    if has_bottom {
        halo[index + cols].top = max_y;
    }
    if has_top {
        halo[index - cols].bottom = min_y;
    }
    if has_right && has_bottom {
        halo[index + cols + 1].top_left = [max_x, max_y];
    }
    if has_left && has_bottom {
        halo[index + cols - 1].top_right = [min_x, max_y];
    }
    if has_right && has_top {
        halo[index - cols + 1].bottom_left = [max_x, min_y];
    }
    if has_left && has_top {
        halo[index - cols - 1].bottom_right = [min_x, min_y];
    }
}

/// Set the copies in `halo` of every node of row `y` of `grid` to the
/// facing sides of its neighbors' boxes, which must already be shrunk in
/// rows `y - 1` to `y + 1`. Interior nodes of the row have both horizontal
/// neighbors, so only the two end nodes test for the left and right edges.
fn pull_halo_row<T: Copy>(grid: &Grid<HaloOverflowBoundGridNode<T>>, halo: &mut [Halo], y: usize) {
    let (cols, rows) = (grid.cols(), grid.rows());
    let vertical = [y != 0, y + 1 != rows];
    let row = &mut halo[y * cols..(y + 1) * cols];
    let mut pull = |x: usize, horizontal: [bool; 2]| {
        // SAFETY: `y` is a row and `x` a column of `grid`, and the flags
        // state which neighbors the node lacks.
        unsafe {
            *row.get_unchecked_mut(x) = Halo::of_neighbors(grid, (x, y), horizontal, vertical);
        }
    };
    if cols == 1 {
        pull(0, [false, false]);
        return;
    }
    pull(0, [false, true]);
    for x in 1..cols - 1 {
        pull(x, [true, true]);
    }
    pull(cols - 1, [true, false]);
}

/// The box comes first, as every test of the node reads it; the node's
/// copies live in the grid's `halo` array.
#[repr(C)]
struct HaloOverflowBoundGridNode<T: Copy> {
    /// Covers the bounds of every item in `items`; [`EMPTY`] while there are
    /// none.
    bounds: Rectangle,
    items: Vec<Entry<T>>,
}

impl<T: Copy> HaloOverflowBoundGridNode<T> {
    fn new() -> Self {
        Self {
            bounds: EMPTY,
            items: Vec::new(),
        }
    }

    /// Grow the box to cover `bounds`.
    #[inline]
    fn grow(&mut self, bounds: &Rectangle) {
        cover(&mut self.bounds, bounds);
    }

    /// Shrink the box to the bounds of the items stored now.
    fn refresh(&mut self) {
        let Self { bounds, items } = self;
        *bounds = EMPTY;
        for entry in items.iter() {
            cover(bounds, &entry.bounds);
        }
    }

    fn search_items(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        if !self.bounds.intersects(bounds) {
            return;
        }
        self.items
            .iter()
            .filter(|entry| entry.bounds.intersects(bounds))
            .for_each(|entry| callback(entry.item));
    }

    /// [`search_items`](Self::search_items) for the pair queries, a copy so
    /// that they share no code with the searches.
    fn pair_with(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        if !self.bounds.intersects(bounds) {
            return;
        }
        self.items
            .iter()
            .filter(|entry| entry.bounds.intersects(bounds))
            .for_each(|entry| callback(entry.item));
    }

    /// Test the items after `index` against `bounds`, without the box, which
    /// covers the item at `index` and thus meets any query it makes.
    fn pair_with_after(&self, index: usize, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        self.items[index + 1..]
            .iter()
            .filter(|entry| entry.bounds.intersects(bounds))
            .for_each(|entry| callback(entry.item));
    }

    /// [`iter_items`](Self::iter_items) for the pair queries, a copy so that
    /// they share no code with the searches.
    fn pair_all(&self, callback: &mut impl FnMut(T)) {
        self.items.iter().for_each(|entry| callback(entry.item));
    }

    fn iter_items(&self, callback: &mut impl FnMut(T)) {
        self.items.iter().for_each(|entry| callback(entry.item));
    }
}

impl<T: Copy> BoxedNode for HaloOverflowBoundGridNode<T> {
    #[inline]
    fn bounds(&self) -> &Rectangle {
        &self.bounds
    }

    #[inline]
    fn refresh(&mut self) {
        HaloOverflowBoundGridNode::refresh(self);
    }
}

/// Grow `target` to cover `bounds`. A NaN coordinate leaves its side
/// unchanged, which is harmless since an item with one intersects no query.
#[inline]
fn cover(target: &mut Rectangle, bounds: &Rectangle) {
    target.min_x = lower(target.min_x, bounds.min_x);
    target.min_y = lower(target.min_y, bounds.min_y);
    target.max_x = higher(target.max_x, bounds.max_x);
    target.max_y = higher(target.max_y, bounds.max_y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    const MAX_LEVEL: u8 = 7;
    type Tree = HaloOverflowBoundGrid<usize, MAX_LEVEL>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    crate::grid_tests::span_tests!(HaloOverflowBoundGridNode);

    /// For the shared refresh tests: whether `level` holds no level marks.
    fn level_marks_clear(tree: &Tree, level: usize) -> bool {
        tree.marks[level] == Marks::default()
    }

    /// For the shared refresh tests: the copies of `level` must equal the
    /// boxes they stand for, rounded outwards, right after a refresh.
    fn assert_refreshed_extra(tree: &Tree, level: usize, context: &str) {
        assert_eq!(
            &tree.halo[level][..],
            &exact_halo(tree, level)[..],
            "{context}"
        );
    }

    /// The same moves go to two trees, one refreshed for pairs, one fully.
    /// After every refresh the boxes and spans of the two agree, also when a
    /// second refresh for pairs follows with nothing changed, and searches
    /// find the same items in both. A full refresh
    /// of the first, with or without moves since its last refresh for pairs,
    /// then leaves it as if it had never skipped the copies. Bursts of moves
    /// alternate with a few, so that refreshes both sweep the rows and visit
    /// the marked nodes.
    #[test]
    fn a_full_refresh_restores_the_copies_after_refreshes_for_pairs() {
        let mut rng = StdRng::seed_from_u64(31);
        for world in WORLDS {
            let (mut pairs, mut full) = (Tree::new(world.clone()), Tree::new(world.clone()));
            let root = pairs.get_bounds().clone();
            let mut items: Vec<[ItemHandle; 2]> = Vec::new();
            for id in 0..500 {
                let bounds = random_bounds(&mut rng, &root);
                items.push([pairs.insert(bounds.clone(), id), full.insert(bounds, id)]);
            }
            pairs.refresh_bounds();
            full.refresh_bounds();
            let moves = |rng: &mut StdRng, pairs: &mut Tree, full: &mut Tree, count: usize| {
                for _ in 0..count {
                    let [a, b] = items[rng.gen_range(0..items.len())];
                    let bounds = random_bounds(rng, &root);
                    pairs.update(bounds.clone(), a);
                    full.update(bounds, b);
                }
            };
            for round in 0..40 {
                let context = format!("{world:?}, round {round}");
                let count = if round % 3 == 0 { 300 } else { 3 };
                moves(&mut rng, &mut pairs, &mut full, count);
                pairs.refresh_for_pairs();
                if round % 2 == 1 {
                    pairs.refresh_for_pairs();
                }
                full.refresh_bounds();
                for level in 1..=MAX_LEVEL as usize {
                    let boxes = |tree: &Tree| -> Vec<Rectangle> {
                        tree.layers[level]
                            .iter()
                            .map(|node| node.bounds.clone())
                            .collect()
                    };
                    assert_eq!(boxes(&pairs), boxes(&full), "{context}, level {level}");
                    let (a, b) = (&pairs.lines[level], &full.lines[level]);
                    assert_eq!(a.columns, b.columns, "{context}, level {level}");
                    assert_eq!(a.rows, b.rows, "{context}, level {level}");
                }
                let found = |tree: &Tree, query: &Rectangle| {
                    let mut found = Vec::new();
                    tree.search(query, |id| found.push(id));
                    found.sort_unstable();
                    found
                };
                for _ in 0..100 {
                    let query = random_bounds(&mut rng, &root);
                    assert_eq!(
                        found(&pairs, &query),
                        found(&full, &query),
                        "{context}, {query:?}"
                    );
                }
                if round % 5 == 4 {
                    if round % 10 == 9 {
                        moves(&mut rng, &mut pairs, &mut full, 3);
                        full.refresh_bounds();
                    }
                    pairs.refresh_bounds();
                    assert_refreshed(&pairs, &context);
                }
            }
        }
    }

    /// Expected copies of every node of `level` right after a refresh,
    /// computed item by item from the neighbors, in storage order.
    fn exact_halo(tree: &Tree, level: usize) -> Vec<Halo> {
        let grid = &tree.layers[level];
        let (cols, rows) = (grid.cols(), grid.rows());
        let unions: Vec<Rectangle> = grid.iter().map(union).collect();
        let union_at = |x: isize, y: isize| -> Rectangle {
            if x < 0 || y < 0 || x as usize >= cols || y as usize >= rows {
                EMPTY
            } else {
                unions[y as usize * cols + x as usize].clone()
            }
        };
        let mut exact = Vec::with_capacity(cols * rows);
        for y in 0..rows as isize {
            for x in 0..cols as isize {
                let left = union_at(x - 1, y);
                let right = union_at(x + 1, y);
                let top = union_at(x, y - 1);
                let bottom = union_at(x, y + 1);
                let top_left = union_at(x - 1, y - 1);
                let top_right = union_at(x + 1, y - 1);
                let bottom_left = union_at(x - 1, y + 1);
                let bottom_right = union_at(x + 1, y + 1);
                let own = union_at(x, y);
                // Spelled out instead of `Halo::of_boxes`, so that the
                // directions of rounding are checked against this.
                let (up, down) = (f32_at_least, f32_at_most);
                exact.push(Halo {
                    left: up(left.max_x),
                    right: down(right.min_x),
                    top: up(top.max_y),
                    bottom: down(bottom.min_y),
                    top_left: [up(top_left.max_x), up(top_left.max_y)],
                    top_right: [down(top_right.min_x), up(top_right.max_y)],
                    bottom_left: [up(bottom_left.max_x), down(bottom_left.min_y)],
                    bottom_right: [down(bottom_right.min_x), down(bottom_right.min_y)],
                    own: [
                        down(own.min_x),
                        down(own.min_y),
                        up(own.max_x),
                        up(own.max_y),
                    ],
                });
            }
        }
        exact
    }

    /// Whether `copies` reach at least as far as `exact` on every slot.
    fn covers(copies: &Halo, exact: &Halo) -> bool {
        copies.own[0] <= exact.own[0]
            && copies.own[1] <= exact.own[1]
            && copies.own[2] >= exact.own[2]
            && copies.own[3] >= exact.own[3]
            && copies.left >= exact.left
            && copies.right <= exact.right
            && copies.top >= exact.top
            && copies.bottom <= exact.bottom
            && copies.top_left[0] >= exact.top_left[0]
            && copies.top_left[1] >= exact.top_left[1]
            && copies.top_right[0] <= exact.top_right[0]
            && copies.top_right[1] >= exact.top_right[1]
            && copies.bottom_left[0] >= exact.bottom_left[0]
            && copies.bottom_left[1] <= exact.bottom_left[1]
            && copies.bottom_right[0] <= exact.bottom_right[0]
            && copies.bottom_right[1] <= exact.bottom_right[1]
    }

    /// The copies in every node must cover the items of its neighbors
    /// whenever neither its row nor a row next to it is touched, equal their
    /// boxes right after a refresh, which clears the level and row marks, and
    /// never change for the missing neighbors of an edge node.
    #[test]
    fn halo_covers_neighbors_until_refreshed() {
        exercise(5, |tree, step, refreshed| {
            for (level, grid) in tree.layers.iter().enumerate().skip(1) {
                let (cols, rows) = (grid.cols(), grid.rows());
                let marks = tree.marks[level];
                let lines = &tree.lines[level];
                if refreshed {
                    assert_eq!(marks, Marks::default(), "step {step}, level {level}");
                    assert!(
                        lines.row_touched.iter().all(|&touched| !touched),
                        "step {step}, level {level}: {:?}",
                        lines.row_touched
                    );
                }
                let exact = exact_halo(tree, level);
                let copies = &tree.halo[level];
                assert_eq!(copies.len(), cols * rows);
                for (index, (copies, exact)) in copies.iter().zip(&exact).enumerate() {
                    let (x, y) = (index % cols, index / cols);
                    let context = format!(
                        "step {step}, level {level}, node {index}: {copies:?} for {exact:?}"
                    );
                    if refreshed {
                        assert_eq!(copies, exact, "{context}");
                    } else if !lines.near_touched(y) {
                        assert!(covers(copies, exact), "{context}");
                    }
                    // An edge node keeps the initial copies of the neighbors it lacks.
                    type Side = fn(&Halo) -> (f32, [f32; 2], [f32; 2]);
                    let edges: [(bool, Side); 4] = [
                        (x == 0, |h| (h.left, h.top_left, h.bottom_left)),
                        (x + 1 == cols, |h| (h.right, h.top_right, h.bottom_right)),
                        (y == 0, |h| (h.top, h.top_left, h.top_right)),
                        (y + 1 == rows, |h| (h.bottom, h.bottom_left, h.bottom_right)),
                    ];
                    for (at_edge, side) in edges {
                        if at_edge {
                            assert_eq!(side(copies), side(&NO_HALO), "{context}");
                        }
                    }
                }
            }
        });
    }

    /// Storing an item marks its level touched and stale, taking one marks
    /// it touched, and a refresh clears both; an unchanged level stays clear.
    /// The spans and copies of a clear level are exact.
    #[test]
    fn marks_follow_stores_and_takes() {
        const TAKEN: Marks = Marks {
            touched: true,
            stale: false,
        };
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        let finest = MAX_LEVEL as usize;
        let small = Rectangle::center_rect(100.0, 100.0, 4.0, 4.0);
        let big = Rectangle::center_rect(100.0, 100.0, 200.0, 200.0);
        let coarser = tree.position(&big).0;
        assert!(coarser < finest);
        let clear = |tree: &Tree| tree.marks.iter().all(|&marks| marks == Marks::default());

        assert!(clear(&tree));
        let handle = tree.insert(small.clone(), 0);
        assert_eq!(tree.marks[finest], Marks::STORED);
        tree.refresh_bounds();
        assert!(clear(&tree));

        // Replaced in its node.
        tree.update(Rectangle::center_rect(101.0, 100.0, 4.0, 4.0), handle);
        assert_eq!(tree.marks[finest], Marks::STORED);
        tree.refresh_bounds();
        assert!(clear(&tree));

        // Moved to another level: stored there, taken here.
        tree.update(big.clone(), handle);
        assert_eq!(tree.marks[coarser], Marks::STORED);
        assert_eq!(tree.marks[finest], TAKEN);
        tree.refresh_bounds();
        assert!(clear(&tree));

        tree.remove(handle);
        assert_eq!(tree.marks[coarser], TAKEN);
        assert_eq!(tree.marks[finest], Marks::default());
        tree.refresh_bounds();
        assert!(clear(&tree));
    }

    /// Storing an item in a row or taking one from it marks the row; the
    /// rows next to a marked row, within the grid, count as near it; a
    /// refresh clears the marks of every row.
    #[test]
    fn row_marks_follow_stores_and_takes() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        let finest = MAX_LEVEL as usize;
        // Finest nodes are 16 units wide; this one is (6, 6).
        let item = Rectangle::center_rect(100.0, 100.0, 4.0, 4.0);
        assert_eq!(tree.position(&item), (finest, 6, 6));
        let marked = |tree: &Tree| -> Vec<usize> {
            (0..tree.lines[finest].row_touched.len())
                .filter(|&y| tree.lines[finest].row_touched[y])
                .collect()
        };
        let near = |tree: &Tree| -> Vec<usize> {
            (0..tree.lines[finest].row_touched.len())
                .filter(|&y| tree.lines[finest].near_touched(y))
                .collect()
        };

        let handle = tree.insert(item.clone(), 0);
        assert_eq!(marked(&tree), [6]);
        assert_eq!(near(&tree), [5, 6, 7]);
        tree.refresh_bounds();
        assert_eq!(marked(&tree), []);
        assert_eq!(near(&tree), []);

        // Moved to the row below: stored there, taken here.
        tree.update(Rectangle::center_rect(100.0, 116.0, 4.0, 4.0), handle);
        assert_eq!(marked(&tree), [6, 7]);
        assert_eq!(near(&tree), [5, 6, 7, 8]);
        tree.refresh_bounds();

        tree.remove(handle);
        assert_eq!(marked(&tree), [7]);
        tree.refresh_bounds();
        assert_eq!(marked(&tree), []);

        // At the edge of the grid, the neighborhood ends with the grid.
        tree.insert(Rectangle::center_rect(100.0, 4.0, 4.0, 4.0), 1);
        assert_eq!(marked(&tree), [0]);
        assert_eq!(near(&tree), [0, 1]);
    }

    /// An item moved inside its node, away from the box its node had at the
    /// refresh, must be found before the next refresh by a query at its new
    /// place. The walk reads the copies there, since no neighbor of the node
    /// is marked and the level's marks are sparse, and the copy of the node's
    /// own box does not reflect the move.
    #[test]
    fn item_moved_within_its_node_is_found_before_refresh() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        let finest = MAX_LEVEL as usize;
        // Finest nodes are 16 units wide; (6, 6) spans 96..112.
        let before = Rectangle::center_rect(98.0, 98.0, 2.0, 2.0);
        let after = Rectangle::center_rect(110.0, 110.0, 2.0, 2.0);
        let query = Rectangle::center_rect(110.0, 110.0, 1.0, 1.0);
        assert_eq!(tree.position(&before), (finest, 6, 6));
        assert_eq!(tree.position(&after), (finest, 6, 6));
        assert_eq!(tree.position(&query), (finest, 6, 6));
        let handle = tree.insert(before, 0);
        // Nine more items in nodes far from (6, 6), so that the refresh after
        // one of them moved finds one marked node of ten occupied, and the
        // level does not go by the touched rows until the next refresh.
        let others: Vec<ItemHandle> = (0..9)
            .map(|i| {
                let bounds = Rectangle::center_rect(500.0 + 20.0 * i as f64, 500.0, 2.0, 2.0);
                tree.insert(bounds, 1 + i)
            })
            .collect();
        tree.refresh_bounds();
        tree.update(Rectangle::center_rect(501.0, 500.0, 2.0, 2.0), others[0]);
        tree.refresh_bounds();
        assert!(!tree.lines[finest].dense_marks);
        tree.update(after, handle);
        let mut found = Vec::new();
        tree.search(&query, |id| found.push(id));
        assert_eq!(found, [0]);
        tree.refresh_bounds();
        found.clear();
        tree.search(&query, |id| found.push(id));
        assert_eq!(found, [0]);
    }

    /// Worlds far from the origin or beyond the range of `f32`, where the
    /// copies are loose or infinite: searches before and after refreshes of
    /// both kinds find what a brute force finds.
    #[test]
    fn far_and_huge_worlds_match_brute_force() {
        let mut rng = StdRng::seed_from_u64(97);
        for world in [
            Rectangle::new(1e7, 1e7, 1e7 + 4096.0, 1e7 + 4096.0),
            Rectangle::new(-3e9, 5e9, -3e9 + 1e6, 5e9 + 1e6),
            Rectangle::new(1e15, 1e15, 1e15 + 1e9, 1e15 + 1e9),
            Rectangle::new(0.0, 0.0, 1e39, 1e39),
            Rectangle::new(-1e39, -1e39, 1e39, 1e39),
            Rectangle::new(1e40, 1e40, 1e41, 1e41),
        ] {
            let mut tree = Tree::new(world.clone());
            let root = tree.get_bounds().clone();
            let mut items: Vec<(Rectangle, ItemHandle)> = Vec::new();
            for id in 0..300 {
                let bounds = random_bounds(&mut rng, &root);
                let handle = tree.insert(bounds.clone(), id);
                items.push((bounds, handle));
            }
            let check = |tree: &Tree, items: &[(Rectangle, ItemHandle)], rng: &mut StdRng| {
                for _ in 0..200 {
                    let query = random_bounds(rng, &root);
                    let mut found = Vec::new();
                    tree.search(&query, |id| found.push(id));
                    found.sort_unstable();
                    let expected: Vec<usize> = (0..items.len())
                        .filter(|&id| items[id].0.intersects(&query))
                        .collect();
                    assert_eq!(found, expected, "{world:?}, {query:?}");
                }
            };
            for round in 0..12 {
                if round % 3 == 2 {
                    tree.refresh_for_pairs();
                } else {
                    tree.refresh_bounds();
                }
                check(&tree, &items, &mut rng);
                let moves = if round % 2 == 0 { 3 } else { 150 };
                for _ in 0..moves {
                    let id = rng.gen_range(0..items.len());
                    let bounds = random_bounds(&mut rng, &root);
                    tree.update(bounds.clone(), items[id].1);
                    items[id].0 = bounds;
                }
                check(&tree, &items, &mut rng);
            }
        }
    }

    /// One axis of a case of `near_ties_survive_the_rounding`, in a world
    /// where `f32` steps by 1: the extents of the item and of the query, the
    /// query in the node starting at `low`, the item `delta` nodes further.
    /// The query's facing side lies within the item by half a step, in the
    /// same step as the item's side, so that rounding either of them the
    /// wrong way separates them. With `delta` 0, `tie` picks the side of the
    /// item that is close (1 low, 2 high) or none (0).
    fn near_tie_axis(low: f64, node: f64, delta: isize, tie: u8) -> ([f64; 2], [f64; 2]) {
        let high = low + node;
        let center = low + node / 2.0;
        match (delta, tie) {
            (1, _) => {
                let side = high - 10.0 + 0.25;
                ([side, side + 40.0], [side + 0.5 - 4.0, side + 0.5])
            }
            (-1, _) => {
                let side = low + 10.0 + 0.75;
                ([side - 40.0, side], [side - 0.5, side - 0.5 + 4.0])
            }
            (0, 1) => {
                let side = center - 3.0 + 0.25;
                ([side, side + 20.0], [side + 0.5 - 4.0, side + 0.5])
            }
            (0, 2) => {
                let side = center + 3.0 + 0.75;
                ([side - 20.0, side], [side - 0.5, side - 0.5 + 4.0])
            }
            _ => ([center - 10.0, center + 10.0], [center - 2.0, center + 2.0]),
        }
    }

    /// A query that overlaps an item by less than a step of `f32`, the item
    /// in the query's node or in any of its eight neighbors, must find it
    /// after either way of refreshing: every copy and every side of the query
    /// has to be rounded away from the other.
    #[test]
    fn near_ties_survive_the_rounding() {
        const NODE: f64 = 64.0;
        let axes = [(1, 0), (-1, 0), (0, 0), (0, 1), (0, 2)];
        let finest = MAX_LEVEL as usize;
        let (cell_x, cell_y) = (30usize, 31usize);
        for origin in [1e7, -1e7 - 4096.0] {
            let world = Rectangle::new(origin, origin, origin + 4096.0, origin + 4096.0);
            let mut tree = Tree::new(world);
            let low_x = origin + cell_x as f64 * NODE;
            let low_y = origin + cell_y as f64 * NODE;
            for sweep in [false, true] {
                for (delta_x, tie_x) in axes {
                    for (delta_y, tie_y) in axes {
                        let (item_x, query_x) = near_tie_axis(low_x, NODE, delta_x, tie_x);
                        let (item_y, query_y) = near_tie_axis(low_y, NODE, delta_y, tie_y);
                        let item = Rectangle::new(item_x[0], item_y[0], item_x[1], item_y[1]);
                        let query = Rectangle::new(query_x[0], query_y[0], query_x[1], query_y[1]);
                        let context =
                            format!("{delta_x} {tie_x} {delta_y} {tie_y} {sweep} {origin}");
                        let item_node = (
                            finest,
                            cell_x.wrapping_add_signed(delta_x),
                            cell_y.wrapping_add_signed(delta_y),
                        );
                        assert_eq!(tree.position(&item), item_node, "{context}");
                        assert_eq!(tree.position(&query), (finest, cell_x, cell_y), "{context}");
                        assert!(item.intersects(&query), "{context}");
                        let handle = tree.insert(item, 0);
                        if sweep {
                            tree.refresh_with(|_, _, _| false);
                        } else {
                            tree.refresh_bounds();
                        }
                        let mut found = Vec::new();
                        tree.search(&query, |id| found.push(id));
                        assert_eq!(found, [0], "{context}");
                        tree.remove(handle);
                        tree.refresh_bounds();
                    }
                }
            }
        }
    }

    /// The outward rounding: at or beyond the value, by at most one step
    /// where `f32` can hold the value, and infinities unchanged.
    #[test]
    fn outward_rounding_brackets_the_value() {
        let mut rng = StdRng::seed_from_u64(5);
        let check = |x: f64| {
            let (up, down) = (f32_at_least(x), f32_at_most(x));
            assert!(f64::from(up) >= x, "{x:e}: {up:e}");
            assert!(f64::from(down) <= x, "{x:e}: {down:e}");
            if x.abs() <= f64::from(f32::MAX) {
                assert!(
                    f64::from(up.next_down()) < x,
                    "{x:e}: {up:e} is not the smallest above"
                );
                assert!(
                    f64::from(down.next_up()) > x,
                    "{x:e}: {down:e} is not the largest below"
                );
            }
        };
        for x in [
            0.0,
            1.0,
            0.1,
            1e-46,
            1e-300,
            f64::MIN_POSITIVE,
            5e-324,
            f64::from(f32::MIN_POSITIVE),
            f64::from(f32::MAX),
            f64::from(f32::MAX) * 1.0000001,
            1e39,
            1e300,
            f64::MAX,
            80000.0,
            79999.99999,
        ] {
            check(x);
            check(-x);
        }
        for _ in 0..200_000 {
            let x = f64::from_bits(rng.r#gen::<u64>());
            if x.is_finite() {
                check(x);
            }
            let scale = rng.gen_range(-50.0..50.0_f64).exp2();
            check(rng.gen_range(-1.0..1.0) * scale);
        }
        assert_eq!(f32_at_least(f64::INFINITY), f32::INFINITY);
        assert_eq!(f32_at_most(f64::INFINITY), f32::INFINITY);
        assert_eq!(f32_at_least(f64::NEG_INFINITY), f32::NEG_INFINITY);
        assert_eq!(f32_at_most(f64::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(f32_at_least(f64::NAN).is_nan() && f32_at_most(f64::NAN).is_nan());
    }

    /// An item that grows or moves towards any of the eight neighbors of its
    /// node must be found before the next refresh by a query centered in
    /// that neighbor, whose copies do not reflect it yet.
    #[test]
    fn item_reaching_into_a_neighbor_is_found_before_refresh() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        // Finest nodes are 16 units wide; (6, 6) spans 96..112.
        let home = Rectangle::center_rect(100.0, 100.0, 4.0, 4.0);
        let far = Rectangle::center_rect(500.0, 500.0, 4.0, 4.0);
        assert_eq!(tree.position(&home), (MAX_LEVEL as usize, 6, 6));
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                if (dx, dy) == (0, 0) {
                    continue;
                }
                // As large as the node, centered 7 units from its center
                // towards the neighbor, and so reaching 1 unit short of the
                // neighbor's center: 103..119 for dx = 1, against 112..128.
                let reaching = Rectangle::center_rect(
                    104.0 + 7.0 * dx as f64,
                    104.0 + 7.0 * dy as f64,
                    16.0,
                    16.0,
                );
                let query = Rectangle::center_rect(
                    104.0 + 12.0 * dx as f64,
                    104.0 + 12.0 * dy as f64,
                    2.0,
                    2.0,
                );
                let context = format!("towards ({dx}, {dy})");
                assert_eq!(
                    tree.position(&reaching),
                    (MAX_LEVEL as usize, 6, 6),
                    "{context}"
                );
                assert_eq!(
                    tree.position(&query),
                    (MAX_LEVEL as usize, (6 + dx) as usize, (6 + dy) as usize),
                    "{context}"
                );

                // Grown within its node, or moved in from another node.
                for before in [&home, &far] {
                    let mut found = Vec::new();
                    let handle = tree.insert(before.clone(), 0);
                    tree.refresh_bounds();
                    tree.search(&query, |id| found.push(id));
                    assert_eq!(found, [], "{context}, {before:?}");
                    tree.update(reaching.clone(), handle);
                    tree.search(&query, |id| found.push(id));
                    assert_eq!(found, [0], "{context}, from {before:?}");
                    tree.refresh_bounds();
                    found.clear();
                    tree.search(&query, |id| found.push(id));
                    assert_eq!(found, [0], "{context}, from {before:?}, refreshed");
                    tree.remove(handle);
                    tree.refresh_bounds();
                }
            }
        }
    }
}
