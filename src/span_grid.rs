//! A single-level loose grid with column and row spans:
//! [`SpanOverflowBoundGrid`] reduced
//! to its finest level, for scenes whose items all fit a cell. It serves as a
//! comparison with the uniform grid, which instead registers an item in every
//! cell it overlaps.
//!
//! Every item is stored in the cell that contains its center, so an item no
//! larger than a cell extends at most half a cell beyond it (the fixed
//! reach). A query no larger than a cell reads the cell of its center and the
//! eight around it; a larger query scans the cells within half a cell of it.
//! Each cell keeps a box covering its items, and each column and row the span
//! of its items along its own axis; neighbor reads and the edge columns and
//! rows of a scan are skipped where no span meets the query. Boxes and spans
//! only widen, except that an emptied node resets its box, until
//! [`refresh_bounds`] shrinks the boxes and row spans of the touched rows and
//! rebuilds every column span. Items larger than a
//! cell, degenerate (zero or NaN size) or centered outside the world go to
//! one overflow node that every search reads.
//!
//! [`SpanOverflowBoundGrid`]: crate::SpanOverflowBoundGrid
//! [`refresh_bounds`]: SpanGrid::refresh_bounds

use crate::cell_order::for_each_cell_in_morton_order;
use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::layer_geometry::{
    LayerGeometry, first_reaching_node, last_reaching_node, reach_margin, within_reach,
};
use crate::node_marks::{BoxedNode, NodeMarks};
use crate::overflow_bound::{higher, lower};
use crate::rect::{EMPTY, NOWHERE, Rectangle};
use grid::*;

/// `level` of a [`GridLocation`] in the overflow node; cells are level 1.
const OVERFLOW: usize = 0;

pub struct SpanGrid<T: Copy> {
    cells: Grid<Node<T>>,
    geometry: LayerGeometry,
    /// Reciprocals of the cell width and height. Position and search multiply
    /// by them instead of dividing by the cell size.
    inverse_node_size: (f64, f64),
    /// Cell size less the margin of [`reach_margin`]: an item or query no larger
    /// stays within half a cell of its cell whatever the rounding of its bounds.
    size_limit: (f64, f64),
    lines: LineSpan,
    /// Items larger than a cell, degenerate or centered outside the world.
    overflow: Node<T>,
    overflow_touched: bool,
    /// Square of whole cells anchored at the world's min corner, from which
    /// the geometry is derived, as in the hierarchical grids.
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    slots: Slots<GridLocation>,
    /// Which cells changed since the last refresh, for a refresh that visits
    /// only those. A field of its own rather than part of `lines`, whose
    /// growth by these three slices shifted the fields that the pair queries
    /// read and measured them 4-6% slower at 160,000 items in `SpanGrid`.
    marks: NodeMarks,
}

impl<T: Copy> SpanGrid<T> {
    /// Create a grid of cells of about `cell_size` covering `world_bounds`.
    pub fn new(world_bounds: Rectangle, cell_size: f64) -> Self {
        assert!(
            cell_size > 0.0 && cell_size.is_finite(),
            "invalid cell size"
        );
        let cols = (world_bounds.get_width() / cell_size).ceil().max(1.0) as usize;
        let rows = (world_bounds.get_height() / cell_size).ceil().max(1.0) as usize;
        let side = cols.max(rows) as f64 * cell_size;
        let (min_x, min_y) = world_bounds.get_min();
        let root_bounds = Rectangle::new(min_x, min_y, min_x + side, min_y + side);
        let mut nodes = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            nodes.push(Node::new());
        }
        let (node_width, node_height) = LayerGeometry::node_size(&root_bounds, cols, rows);
        let inverse_node_size = (1.0 / node_width, 1.0 / node_height);
        let margin = reach_margin(&root_bounds);
        Self {
            cells: Grid::from_vec(nodes, cols),
            // Level 2, as every level but the root gets the update shortcut.
            geometry: LayerGeometry::new(&root_bounds, 2, cols, rows),
            inverse_node_size,
            size_limit: (node_width - margin, node_height - margin),
            lines: LineSpan::new(cols, rows),
            overflow: Node::new(),
            overflow_touched: false,
            root_bounds,
            world_bounds,
            slots: Slots::new(),
            marks: NodeMarks::new(cols, rows),
        }
    }

    /// Square of whole cells anchored at the world's min corner.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root_bounds
    }

    /// Size of a cell along each axis.
    pub fn cell_size(&self) -> (f64, f64) {
        (self.geometry.node_width, self.geometry.node_height)
    }

    /// Whether an item of this size belongs in a cell, not the overflow node,
    /// unless rounding takes it out of reach of the scans.
    #[inline]
    fn fits_cell(&self, bounds: &Rectangle) -> bool {
        self.geometry
            .matches_layer(bounds.get_width(), bounds.get_height(), true)
    }

    /// Whether a query of this size can read its cell and the neighbors: below
    /// [`size_limit`](Self::size_limit), or within reach despite its size, so
    /// that it cannot meet an item two cells off. A larger one, or a degenerate
    /// one, gives the same results from the scan.
    #[inline]
    fn fits_walk(&self, bounds: &Rectangle) -> bool {
        let (width, height) = (bounds.get_width(), bounds.get_height());
        (width > 0.0 && height > 0.0 && width <= self.size_limit.0 && height <= self.size_limit.1)
            || self.fits_walk_past_limit(bounds)
    }

    /// [`fits_walk`](Self::fits_walk) of a query above the limit, kept out of
    /// `search`, as few queries lie within the margin of the cell size and those
    /// above it go on to the scan.
    #[cold]
    #[inline(never)]
    fn fits_walk_past_limit(&self, bounds: &Rectangle) -> bool {
        self.fits_cell(bounds)
            && !self.out_of_reach(bounds, self.cell_containing(bounds.get_center()))
    }

    /// Cell whose area contains `point`, clamped into the grid.
    #[inline]
    fn cell_containing(&self, point: (f64, f64)) -> (usize, usize) {
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let x =
            (((point.0 - origin_x) * self.inverse_node_size.0) as usize).min(self.cells.cols() - 1);
        let y =
            (((point.1 - origin_y) * self.inverse_node_size.1) as usize).min(self.cells.rows() - 1);
        (x, y)
    }

    /// `(level, x, y)` of the node an item with these bounds belongs in.
    pub fn position(&self, bounds: &Rectangle) -> (usize, usize, usize) {
        let center = bounds.get_center();
        if !self.fits_cell(bounds) || !self.world_bounds.contains_point(center) {
            return (OVERFLOW, 0, 0);
        }
        let (x, y) = self.cell_containing(center);
        if (bounds.get_width() > self.size_limit.0 || bounds.get_height() > self.size_limit.1)
            && self.out_of_reach(bounds, (x, y))
        {
            return (OVERFLOW, 0, 0);
        }
        (1, x, y)
    }

    /// Whether an item or query within the margin of the cell size, which
    /// rounding may take past the half cell, cannot count on cell `(x, y)`.
    #[cold]
    #[inline(never)]
    fn out_of_reach(&self, bounds: &Rectangle, cell: (usize, usize)) -> bool {
        !within_reach(
            bounds,
            0.0,
            self.root_bounds.get_min(),
            self.inverse_node_size,
            cell,
        )
    }

    /// Cheap check that `position` would still return `(level, x, y)` for
    /// `bounds`.
    fn stays_in_node(&self, bounds: &Rectangle, (level, x, y): (usize, usize, usize)) -> bool {
        if !self.geometry.has_update_shortcut() {
            return false;
        }
        let center = bounds.get_center();
        let in_a_cell = self.fits_cell(bounds) && self.world_bounds.contains_point(center);
        if level == OVERFLOW {
            return !in_a_cell;
        }
        let (origin_x, origin_y) = self.root_bounds.get_min();
        in_a_cell
            && self
                .geometry
                .matches_cell((center.0 - origin_x, center.1 - origin_y), (x, y))
    }

    #[inline]
    fn node_mut(&mut self, (level, x, y): (usize, usize, usize)) -> &mut Node<T> {
        if level == OVERFLOW {
            &mut self.overflow
        } else {
            &mut self.cells[y][x]
        }
    }

    /// Store `entry` in the node at `coord`, widening its box and, for a
    /// cell, the spans of its column and row.
    fn store(&mut self, coord: (usize, usize, usize), entry: Entry<T>) {
        let (level, x, y) = coord;
        if level == OVERFLOW {
            self.overflow_touched = true;
        } else {
            self.lines.grow((x, y), &entry.bounds);
            // SAFETY: `grow` indexed column `x` and row `y` with bounds
            // checks, and the marks match the cells.
            unsafe { self.marks.mark((x, y)) };
        }
        let handle = entry.handle;
        let node = self.node_mut(coord);
        node.grow(&entry.bounds);
        let index = node.items.len();
        node.items.push(entry);
        *self.slots.get_mut(handle) = GridLocation::new(coord, index);
        if index == 0 && level != OVERFLOW {
            self.marks.filled();
        }
    }

    /// Insert an item and return the handle that identifies it for
    /// [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord = self.position(&bounds);
        let handle = self.slots.allocate(GridLocation::default());
        self.store(
            coord,
            Entry {
                bounds,
                item,
                handle,
            },
        );
        handle
    }

    /// Remove an item.
    ///
    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let entry = self.take_entry(location.coord(), location.index as usize);
        debug_assert_eq!(entry.handle, handle);
        self.slots.free(handle);
    }

    /// Update the bounds of an item and, if necessary, its node.
    ///
    /// Call it only for items that moved or changed size. It does not compare
    /// the bounds with the old ones, so an unchanged item still marks its node
    /// and the next [`refresh_bounds`](Self::refresh_bounds) redoes that node.
    ///
    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let prev = location.coord();
        let index = location.index as usize;
        if self.stays_in_node(&bounds, prev) {
            self.replace_bounds(prev, index, bounds);
            return;
        }
        let next = self.position(&bounds);
        if next == prev {
            self.replace_bounds(prev, index, bounds);
            return;
        }
        let entry = self.take_entry(prev, index);
        self.store(next, Entry { bounds, ..entry });
    }

    /// Overwrite the bounds of an entry that stays in its node, widening its
    /// box and, for a cell, the spans of its column and row.
    fn replace_bounds(&mut self, coord: (usize, usize, usize), index: usize, bounds: Rectangle) {
        let (level, x, y) = coord;
        if level == OVERFLOW {
            self.overflow_touched = true;
        } else {
            self.lines.grow((x, y), &bounds);
            // SAFETY: as in `store`.
            unsafe { self.marks.mark((x, y)) };
        }
        let node = self.node_mut(coord);
        node.grow(&bounds);
        node.items[index].bounds = bounds;
    }

    /// Take the entry at `index` out of the node at `coord`. The entry that
    /// fills the gap has its slot updated; an emptied node resets its box.
    fn take_entry(&mut self, coord: (usize, usize, usize), index: usize) -> Entry<T> {
        let (level, x, y) = coord;
        let node = if level == OVERFLOW {
            &mut self.overflow
        } else {
            &mut self.cells[y][x]
        };
        let entry = node.items.swap_remove(index);
        if let Some(moved) = node.items.get(index) {
            self.slots.get_mut(moved.handle).index = index as u32;
        }
        let emptied = node.items.is_empty();
        if emptied {
            node.bounds = EMPTY;
        }
        if level == OVERFLOW {
            self.overflow_touched = true;
        } else {
            // SAFETY: cell `(x, y)` was just indexed with bounds checks, and
            // the spans and marks match the cells.
            unsafe {
                self.lines.mark_taken(y);
                self.marks.mark((x, y));
                if emptied {
                    // The refresh can then no longer tell whether the box
                    // held an end of its column's or row's span, so both are
                    // rebuilt.
                    self.marks.emptied((x, y));
                }
            }
        }
        entry
    }

    /// Shrink the boxes of the cells that items were stored in, replaced in
    /// or taken from to their items, and the spans of their columns and rows,
    /// and shrink the overflow node's box if it was touched. Either way below
    /// leaves the boxes and spans computed from the items (a span end at zero
    /// may keep either sign, which no comparison tells apart).
    ///
    /// Where at most half the cells are marked, only the marked cells are
    /// visited, after reading the marks of the touched rows, and a column or row span is rebuilt
    /// only if a marked cell held one of its ends and no longer reaches it,
    /// or a cell in it emptied. Otherwise the boxes of the marked cells
    /// shrink and the spans of the touched rows and of every column are rebuilt.
    pub fn refresh_bounds(&mut self) {
        self.refresh_with(NodeMarks::is_sparse);
    }

    /// [`refresh_bounds`](Self::refresh_bounds), visiting only the marked
    /// cells where `sparse(marked, occupied, all cells)` holds and sweeping the rows
    /// otherwise.
    fn refresh_with(&mut self, sparse: impl Fn(usize, usize, usize) -> bool) {
        if std::mem::take(&mut self.overflow_touched) {
            self.overflow.refresh();
        }
        if !std::mem::take(&mut self.lines.touched) {
            return;
        }
        let lines = &mut self.lines;
        let cells = &mut self.cells;
        let marks = &mut self.marks;
        if sparse(
            marks.count(&lines.row_touched),
            marks.occupied(),
            cells.cols() * cells.rows(),
        ) {
            let LineSpan {
                columns,
                rows,
                row_touched,
                ..
            } = lines;
            marks.refresh(cells, columns, rows, row_touched, |_, _| {});
            row_touched.fill(false);
            return;
        }
        lines.columns.fill(NOWHERE);
        for y in 0..cells.rows() {
            if std::mem::take(&mut lines.row_touched[y]) {
                lines.rows[y] = NOWHERE;
                let row_marks = marks.row_ptr(y);
                for (x, node) in cells[y].iter_mut().enumerate() {
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
                for (x, node) in cells[y].iter().enumerate() {
                    if !node.items.is_empty() {
                        lines.widen_column(x, &node.bounds);
                    }
                }
            }
        }
        marks.clear();
    }

    /// Call `callback` with every stored item and each stored item whose bounds
    /// intersect its own, the item itself included where its bounds meet
    /// themselves: what searching with the bounds of every item reports. The
    /// results come from the walk of [`for_each_pair`](Self::for_each_pair),
    /// each pair in both orders, so a result between a cell item and an item of
    /// the overflow node is found by the overflow item's scan of the cells; the
    /// results of one query are not reported together. The walk visits every cell, so where cells far outnumber
    /// items, searching item by item is faster. On the search bench's
    /// distribution with 312.5-unit cells, whose largest balls overflow, the
    /// `self_query` bench group measured 0.39 ms, against 2.44 ms for the
    /// previous implementation, which searched with the bounds of every item in
    /// turn, cell by cell.
    pub fn for_each_self_query(&self, callback: impl FnMut(T, T)) {
        self.pair_walk::<true>(callback);
    }

    /// Call `callback` once for every pair of distinct stored items whose
    /// bounds intersect, with the two items in either order. An item in a
    /// cell reads only the items after it in its cell and the cells after its
    /// own in row-major order, that is the neighbor to the right and those in
    /// the row below, each only if its box reaches the item. The
    /// items of the overflow node pair up with the overflow items after them
    /// and scan every cell within half a cell of their bounds, as a search with
    /// bounds larger than a cell does, so a pair of a cell item and an overflow
    /// item is found once, by the overflow item, and no cell item reads the
    /// overflow node. The cells are visited in Morton order, the overflow node
    /// last.
    ///
    /// Letting each cell item read the overflow node instead, as a search does,
    /// took 2.10 ms against 0.41 ms on the mixed distribution with 312.5-unit
    /// cells, whose largest balls overflow, since every cell item then tested
    /// every overflow item; on small balls, which leave the overflow node
    /// empty, 0.28 against 0.27 ms.
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
        let cells = &self.cells;
        for_each_cell_in_morton_order(cells.cols(), cells.rows(), |x, y| {
            if !cells[y][x].items.is_empty() {
                self.pair_cell::<SELF_QUERY>((x, y), &mut callback);
            }
        });
        for (index, entry) in self.overflow.items.iter().enumerate() {
            if SELF_QUERY && entry.bounds.intersects(&entry.bounds) {
                callback(entry.item, entry.item);
            }
            let mut report = |found| {
                callback(entry.item, found);
                if SELF_QUERY {
                    callback(found, entry.item);
                }
            };
            self.overflow
                .pair_with_after(index, &entry.bounds, &mut report);
            self.pair_scan(&entry.bounds, &mut report);
        }
    }

    /// The pair walk at cell `(x, y)`, which holds items: each item with the
    /// items after it in the cell and with those of the four neighbors after
    /// the cell in row-major order whose boxes reach it.
    ///
    /// The four neighbors are judged by their own boxes as they are now, read
    /// once for the cell, as in [`HaloGrid`](crate::HaloGrid): a box covers
    /// its items at all times, so the tests hold before a refresh as well.
    /// Judging each item by the spans of the neighbors' columns and row
    /// instead, as `search_3x3` does, with each box left to the read of its
    /// cell, the pair query took 1.20 to 1.21 times as long on the mixed
    /// distribution with 312.5-unit cells, 1.23 to 1.27 times with 32 balls
    /// to a cell, and 1.21 to 1.24 times with 500 to 2,000 balls on 128 by
    /// 128 cells, on a Core Ultra 9 275HX with rustc 1.99
    /// (`bench-results/r1-port-2026-11-10.md`).
    ///
    /// Not inlined: inlined into the walk it measured 3% slower on the mixed
    /// distribution, 11% slower with 32 balls to a cell, and took 1.3 to 1.8
    /// times as long with 500 to 2,000 balls on 128 by 128 cells.
    #[inline(never)]
    fn pair_cell<const SELF_QUERY: bool>(
        &self,
        (x, y): (usize, usize),
        callback: &mut impl FnMut(T, T),
    ) {
        let grid = &self.cells;
        let node = &grid[y][x];
        let (has_left, has_right) = (x > 0, x + 1 < grid.cols());
        let has_below = y + 1 < grid.rows();
        // SAFETY: `(x, y)` is a cell of `grid`, and a neighbor is read only
        // if the grid has it.
        let (right, below_left, below, below_right) = unsafe {
            (
                box_at(grid, has_right, x + 1, y),
                box_at(grid, has_left & has_below, x.wrapping_sub(1), y + 1),
                box_at(grid, has_below, x, y + 1),
                box_at(grid, has_right & has_below, x + 1, y + 1),
            )
        };
        // The sides that face the cell: how far each neighbor's items reach
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
            // All four are computed before the first neighbor is read.
            let after = bounds.max_x > right;
            let down = bounds.max_y > below;
            let down_before = (bounds.min_x < below_left[0]) & (bounds.max_y > below_left[1]);
            let down_after = (bounds.max_x > below_right[0]) & (bounds.max_y > below_right[1]);
            node.pair_with_after(index, bounds, &mut report);
            // SAFETY: the box of a missing neighbor is `EMPTY`, which fails
            // every comparison above, so each cell read here exists.
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
        }
    }

    /// Search the grid and execute the callback for each item whose bounds
    /// intersect `bounds`.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        // Every comparison with NaN is false, so a query with a NaN corner
        // meets nothing, while for a NaN minimum the scans would report the
        // items of the nodes they take as inside it untested.
        if bounds.has_nan() {
            return;
        }
        if self.fits_walk(bounds) {
            self.search_3x3(bounds, &mut callback);
        } else {
            self.scan(bounds, &mut callback);
        }
        self.overflow.search_items(bounds, &mut callback);
    }

    /// Search the cell of the query's center and each neighbor whose column
    /// and row, where they differ from the center's, have spans that meet the
    /// query. For queries no larger than a cell.
    fn search_3x3(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        let (x, y) = self.cell_containing(bounds.get_center());
        let grid = &self.cells;
        let lines = &self.lines;
        let (cols, rows) = (grid.cols(), grid.rows());
        // SAFETY: `(x, y)` is a cell, the clamped line indices are in bounds,
        // and each neighbor is read only after the test that it exists.
        unsafe {
            let above = lines.rows.get_unchecked(y.saturating_sub(1))[1];
            let below = lines.rows.get_unchecked((y + 1).min(rows - 1))[0];
            let left_of = lines.columns.get_unchecked(x.saturating_sub(1))[1];
            let right_of = lines.columns.get_unchecked((x + 1).min(cols - 1))[0];
            let up = (y != 0) & (bounds.min_y < above);
            let down = (y + 1 != rows) & (bounds.max_y > below);
            let before = (x != 0) & (bounds.min_x < left_of);
            let after = (x + 1 != cols) & (bounds.max_x > right_of);

            if up {
                if before {
                    grid.get_unchecked(y - 1, x - 1)
                        .search_items(bounds, callback);
                }
                grid.get_unchecked(y - 1, x).search_items(bounds, callback);
                if after {
                    grid.get_unchecked(y - 1, x + 1)
                        .search_items(bounds, callback);
                }
            }
            if before {
                grid.get_unchecked(y, x - 1).search_items(bounds, callback);
            }
            grid.get_unchecked(y, x).search_items(bounds, callback);
            if after {
                grid.get_unchecked(y, x + 1).search_items(bounds, callback);
            }
            if down {
                if before {
                    grid.get_unchecked(y + 1, x - 1)
                        .search_items(bounds, callback);
                }
                grid.get_unchecked(y + 1, x).search_items(bounds, callback);
                if after {
                    grid.get_unchecked(y + 1, x + 1)
                        .search_items(bounds, callback);
                }
            }
        }
    }

    /// Scan every cell within half a cell of a query larger than a cell,
    /// dropping each edge column and row whose span misses the query.
    fn scan(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        let grid = &self.cells;
        let lines = &self.lines;
        let (cols, rows) = (grid.cols(), grid.rows());
        let (offset_x, offset_y) = self.root_bounds.get_min();
        let (inverse_width, inverse_height) = self.inverse_node_size;
        let min_coord_x = first_reaching_node((bounds.min_x - offset_x) * inverse_width);
        let min_coord_y = first_reaching_node((bounds.min_y - offset_y) * inverse_height);
        let max_coord_x =
            last_reaching_node((bounds.max_x - offset_x) * inverse_width).min(cols - 1);
        let max_coord_y =
            last_reaching_node((bounds.max_y - offset_y) * inverse_height).min(rows - 1);

        let column_reaches = |x: usize| {
            let [low, high] = unsafe { *lines.columns.get_unchecked(x) };
            high > bounds.min_x && low < bounds.max_x
        };
        let row_reaches = |y: usize| {
            let [low, high] = unsafe { *lines.rows.get_unchecked(y) };
            high > bounds.min_y && low < bounds.max_y
        };
        let (mut first_x, mut end_x) = (min_coord_x.min(cols), max_coord_x + 1);
        if first_x < end_x && !column_reaches(first_x) {
            first_x += 1;
        }
        if end_x > first_x && !column_reaches(end_x - 1) {
            end_x -= 1;
        }
        let (mut first_y, mut end_y) = (min_coord_y.min(rows), max_coord_y + 1);
        if first_y < end_y && !row_reaches(first_y) {
            first_y += 1;
        }
        if end_y > first_y && !row_reaches(end_y - 1) {
            end_y -= 1;
        }

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
                    node.iter_items(callback);
                } else {
                    node.search_items(bounds, callback);
                }
            }
        }
    }

    /// Scan the cells for an item of the overflow node in
    /// [`for_each_pair`](Self::for_each_pair) as [`scan`](Self::scan) does
    /// for a search, a copy so that pair queries and searches share no code.
    fn pair_scan(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        // As in `search`: a NaN corner meets nothing, while for a NaN minimum
        // the scan would report the items of the cells it takes as inside
        // `bounds` untested.
        if bounds.has_nan() {
            return;
        }
        let grid = &self.cells;
        let lines = &self.lines;
        let (cols, rows) = (grid.cols(), grid.rows());
        let (offset_x, offset_y) = self.root_bounds.get_min();
        let (inverse_width, inverse_height) = self.inverse_node_size;
        let min_coord_x = first_reaching_node((bounds.min_x - offset_x) * inverse_width);
        let min_coord_y = first_reaching_node((bounds.min_y - offset_y) * inverse_height);
        let max_coord_x =
            last_reaching_node((bounds.max_x - offset_x) * inverse_width).min(cols - 1);
        let max_coord_y =
            last_reaching_node((bounds.max_y - offset_y) * inverse_height).min(rows - 1);

        let column_reaches = |x: usize| {
            let [low, high] = unsafe { *lines.columns.get_unchecked(x) };
            high > bounds.min_x && low < bounds.max_x
        };
        let row_reaches = |y: usize| {
            let [low, high] = unsafe { *lines.rows.get_unchecked(y) };
            high > bounds.min_y && low < bounds.max_y
        };
        let (mut first_x, mut end_x) = (min_coord_x.min(cols), max_coord_x + 1);
        if first_x < end_x && !column_reaches(first_x) {
            first_x += 1;
        }
        if end_x > first_x && !column_reaches(end_x - 1) {
            end_x -= 1;
        }
        let (mut first_y, mut end_y) = (min_coord_y.min(rows), max_coord_y + 1);
        if first_y < end_y && !row_reaches(first_y) {
            first_y += 1;
        }
        if end_y > first_y && !row_reaches(end_y - 1) {
            end_y -= 1;
        }

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
                    node.pair_all(callback);
                } else {
                    node.pair_with(bounds, callback);
                }
            }
        }
    }
}

/// Span of the items of every column along x and of every row along y since
/// the last refresh, as `[min, max]`, with the touched marks for the refresh.
struct LineSpan {
    columns: Box<[[f64; 2]]>,
    rows: Box<[[f64; 2]]>,
    touched: bool,
    row_touched: Box<[bool]>,
}

impl LineSpan {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            columns: vec![NOWHERE; cols].into_boxed_slice(),
            rows: vec![NOWHERE; rows].into_boxed_slice(),
            touched: false,
            row_touched: vec![false; rows].into_boxed_slice(),
        }
    }

    #[inline]
    fn grow(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.widen((x, y), bounds);
        // SAFETY: `widen` indexed row `y` with a bounds check.
        unsafe { self.mark_taken(y) };
    }

    /// Mark the grid and row `y` touched, without a bounds check, as every
    /// store and take runs this.
    ///
    /// # Safety
    ///
    /// `y` must be a row of the cells.
    #[inline]
    unsafe fn mark_taken(&mut self, y: usize) {
        debug_assert!(y < self.row_touched.len());
        self.touched = true;
        // SAFETY: by the caller's guarantee.
        unsafe {
            *self.row_touched.get_unchecked_mut(y) = true;
        }
    }

    #[inline]
    fn widen(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.widen_column(x, bounds);
        let row = &mut self.rows[y];
        row[0] = lower(row[0], bounds.min_y);
        row[1] = higher(row[1], bounds.max_y);
    }

    #[inline]
    fn widen_column(&mut self, x: usize, bounds: &Rectangle) {
        let column = &mut self.columns[x];
        column[0] = lower(column[0], bounds.min_x);
        column[1] = higher(column[1], bounds.max_x);
    }
}

/// Box of cell `(x, y)` of `grid` if `present`, else [`EMPTY`].
///
/// # Safety
///
/// `(x, y)` must be a cell of `grid` when `present`.
#[inline(always)]
unsafe fn box_at<T: Copy>(grid: &Grid<Node<T>>, present: bool, x: usize, y: usize) -> &Rectangle {
    if present {
        unsafe { &grid.get_unchecked(y, x).bounds }
    } else {
        &EMPTY
    }
}

impl<T: Copy> BoxedNode for Node<T> {
    #[inline]
    fn bounds(&self) -> &Rectangle {
        &self.bounds
    }

    #[inline]
    fn refresh(&mut self) {
        Node::refresh(self);
    }
}

struct Node<T: Copy> {
    /// Covers the bounds of every item in `items`; `EMPTY` while there are none.
    bounds: Rectangle,
    items: Vec<Entry<T>>,
}

impl<T: Copy> Node<T> {
    fn new() -> Self {
        Self {
            bounds: EMPTY,
            items: Vec::new(),
        }
    }

    #[inline]
    fn grow(&mut self, bounds: &Rectangle) {
        self.bounds.min_x = lower(self.bounds.min_x, bounds.min_x);
        self.bounds.min_y = lower(self.bounds.min_y, bounds.min_y);
        self.bounds.max_x = higher(self.bounds.max_x, bounds.max_x);
        self.bounds.max_y = higher(self.bounds.max_y, bounds.max_y);
    }

    fn refresh(&mut self) {
        self.bounds = EMPTY;
        let items = std::mem::take(&mut self.items);
        for entry in &items {
            self.grow(&entry.bounds);
        }
        self.items = items;
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

    fn iter_items(&self, callback: &mut impl FnMut(T)) {
        self.items.iter().for_each(|entry| callback(entry.item));
    }

    /// [`iter_items`](Self::iter_items) for the pair queries.
    fn pair_all(&self, callback: &mut impl FnMut(T)) {
        self.items.iter().for_each(|entry| callback(entry.item));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    type Tree = SpanGrid<usize>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    crate::grid_tests::single_level_tests!();

    /// For the shared refresh tests: whether the grid holds no marks.
    fn marks_clear(tree: &Tree) -> bool {
        !tree.lines.touched && !tree.overflow_touched
    }

    /// For the shared refresh tests: nothing beyond the boxes and spans.
    fn assert_refreshed_extra(_tree: &Tree, _context: &str) {}
}
