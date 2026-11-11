//! Marks of the nodes of one grid level that changed since the last refresh,
//! and the refresh that visits only those, for the grids that keep a box per
//! node and a span per column and row: `SpanOverflowBoundGrid`,
//! `HaloOverflowBoundGrid`, `SpanGrid` and `HaloGrid`.
//!
//! Every store and take marks its node. Where at most half the nodes that
//! hold items, or at most an eighth of all nodes, of a level are marked,
//! [`NodeMarks::refresh`] shrinks the boxes of the marked nodes alone, and rebuilds a column or row span only if a marked node held
//! one of its ends and no longer reaches it, or a node in it emptied; the
//! grids sweep the other levels row by row. Either way leaves the
//! boxes and spans computed from the items (a span end at zero may keep
//! either sign, which no comparison tells apart).
//!
//! With 1% of the balls moving per frame, the frame bench measured the four
//! grids' updates and refresh 83-87% faster, e.g. `HaloOverflowBoundGrid`'s
//! at 4.7 µs instead of 29.9 µs for 10,000 balls, and 0.09 ms instead of
//! 0.65 ms for 160,000; with every ball moving, their update benchmarks
//! measured 2-10% slower, reinsertion in the single-level grids the most
//! (`bench-results/cell-refresh-2026-11-01.md`).
//!
//! Every mark lies in a touched row, so the marks are counted and found in
//! those rows alone, which costs `mark` nothing. With 500 balls on 9 levels
//! and 1% of them moving, where the marks span 84 KB and five nodes change,
//! counting and finding them took about 16 of the 18 µs of a refresh, which
//! now takes 2 µs, on a Core Ultra 9 275HX with rustc 1.99
//! (`bench-results/n2-2026-11-09.md`).
//!
//! The row-by-row sweep of the other levels shrinks only the marked nodes of
//! a touched row too, reading its marks through [`NodeMarks::row_ptr`]: a
//! node no store or take reached still has the box its items give. With nine
//! in ten of 160,000 balls lying still, 32 to a cell, the refresh of
//! `HaloOverflowBoundGrid` took 0.70 times as long and its frame 0.97 times,
//! on the same machine and compiler; with every ball moving, so that no node
//! can be skipped, the refresh took 1.02 times as long and the frame did not
//! change (`bench-results/n1-rustc199-2026-11-09.md`).

use crate::overflow_bound::{higher, lower};
use crate::rect::{NOWHERE, Rectangle};
use grid::Grid;

/// Refresh a level node by node when at most one occupied node in this many
/// is marked. The sweep's work grows with the nodes that hold items, while
/// its visits to empty nodes cost little, so the share of those that changed
/// decides: on the frame bench's mixed distribution with a quarter of the
/// balls moving, about 45% of the finest nodes changed and visiting only
/// those measured 7-14% faster than the sweep, while with all of 10,000 small
/// balls moving on 156-unit cells, 45% of `HaloGrid`'s cells changed, all of
/// those holding items, and visiting them measured 8% slower.
pub(crate) const SPARSE_SHARE: usize = 2;

/// Refresh a level node by node also when at most one node of all in this
/// many is marked, as in a sparse world where the sweep would mostly visit
/// empty nodes.
pub(crate) const SPARSE_NODES: usize = 8;

/// A level whose last refresh found more than one occupied node in this many
/// marked looks for no marks around a query until the next refresh, and
/// goes by the touched rows: most nodes there have a marked neighbor, so
/// reading the marks would rarely spare the fallback.
pub(crate) const DENSE_MARKS: usize = 8;

/// A node whose box covers its items.
pub(crate) trait BoxedNode {
    /// The box; [`EMPTY`](crate::rect::EMPTY) while the node holds no items.
    fn bounds(&self) -> &Rectangle;
    /// Shrink the box to the items held now.
    fn refresh(&mut self);
}

/// Which nodes of one level changed since the last refresh.
pub(crate) struct NodeMarks {
    /// One byte per node, row after row, 1 if an item was stored in,
    /// replaced in or taken from the node since the last refresh, else 0.
    /// Zero bytes pad every row to whole eight-byte words, so that the refresh
    /// skips eight unmarked nodes at a time and finds a node's column and row
    /// without dividing.
    nodes: Box<[u8]>,
    /// Bytes per row of `nodes`: the columns rounded up to a multiple of 8.
    stride: usize,
    /// Number of nodes that hold items, kept by [`filled`](Self::filled) and
    /// [`emptied`](Self::emptied).
    occupied: usize,
    /// Columns and rows whose spans the refresh must rebuild from the boxes:
    /// those in which a node emptied since the last refresh, since an emptied
    /// node resets its box at once, and the refresh can then no longer tell
    /// from the box whether it held an end of the span; and, during a
    /// refresh, those whose end a shrunk box gave up.
    loose_columns: Box<[bool]>,
    loose_rows: Box<[bool]>,
}

/// The number of marks in `marks`. A mark is 0 or 1, so at most 255 of them
/// sum within a byte, and bytes sum in vectors, where widening each to
/// `usize` first took 8 µs for the 84 KB of marks of 9 levels.
#[inline]
fn count_marks(marks: &[u8]) -> usize {
    marks
        .chunks(255)
        .map(|chunk| chunk.iter().sum::<u8>() as usize)
        .sum()
}

impl NodeMarks {
    /// Marks for a grid of `cols` columns and `rows` rows, none set.
    pub(crate) fn new(cols: usize, rows: usize) -> Self {
        let stride = cols.next_multiple_of(8);
        Self {
            nodes: vec![0; stride * rows].into_boxed_slice(),
            stride,
            occupied: 0,
            loose_columns: vec![false; cols].into_boxed_slice(),
            loose_rows: vec![false; rows].into_boxed_slice(),
        }
    }

    /// Mark node `(x, y)` changed. Every store and take runs this, so it
    /// writes without a bounds check.
    ///
    /// # Safety
    ///
    /// `x` and `y` must be a column and a row of the grid the marks were made
    /// for.
    #[inline]
    pub(crate) unsafe fn mark(&mut self, (x, y): (usize, usize)) {
        debug_assert!(x < self.loose_columns.len() && y < self.loose_rows.len());
        // SAFETY: by the caller's guarantee, `y * stride + x` indexes a node.
        unsafe {
            *self.nodes.get_unchecked_mut(y * self.stride + x) = 1;
        }
    }

    /// The marks of row `y`, for a sweep that reads them while it changes
    /// everything else of its level, and need not load the array and the
    /// stride again for every node. The sweep must store no marks while it
    /// reads them, and may read only the row's `cols` first bytes.
    #[inline(always)]
    pub(crate) fn row_ptr(&self, y: usize) -> *const u8 {
        debug_assert!(y < self.loose_rows.len());
        self.nodes.as_ptr().wrapping_add(y * self.stride)
    }

    /// Count one more node holding items: the caller stored the first item
    /// in an empty node.
    #[inline]
    pub(crate) fn filled(&mut self) {
        self.occupied += 1;
    }

    /// Count one node fewer holding items, and mark column `x` and row `y`
    /// for a rebuild of their spans: node `(x, y)` emptied, and reset its box.
    ///
    /// # Safety
    ///
    /// As for [`mark`](Self::mark).
    #[inline]
    pub(crate) unsafe fn emptied(&mut self, (x, y): (usize, usize)) {
        debug_assert!(x < self.loose_columns.len() && y < self.loose_rows.len());
        debug_assert!(self.occupied > 0);
        self.occupied -= 1;
        // SAFETY: by the caller's guarantee, `x` is a column and `y` a row.
        unsafe {
            *self.loose_columns.get_unchecked_mut(x) = true;
            *self.loose_rows.get_unchecked_mut(y) = true;
        }
    }

    /// Number of nodes that hold items.
    pub(crate) fn occupied(&self) -> usize {
        self.occupied
    }

    /// Number of nodes marked since the last refresh; reads the marks of the
    /// rows flagged in `row_touched`, which must hold them all.
    pub(crate) fn count(&self, row_touched: &[bool]) -> usize {
        debug_assert!(row_touched.len() == self.loose_rows.len());
        debug_assert!(
            self.nodes
                .chunks_exact(self.stride)
                .zip(row_touched)
                .all(|(row, &touched)| touched || row.iter().all(|&marked| marked == 0)),
            "a mark outside the touched rows"
        );
        self.nodes
            .chunks_exact(self.stride)
            .zip(row_touched)
            .filter(|&(_, &touched)| touched)
            .map(|(row, _)| count_marks(row))
            .sum()
    }

    /// Whether to refresh a level node by node, by [`SPARSE_SHARE`] and
    /// [`SPARSE_NODES`], given its `marked` nodes, its `occupied` ones and
    /// its `nodes` in all.
    pub(crate) fn is_sparse(marked: usize, occupied: usize, nodes: usize) -> bool {
        marked * SPARSE_SHARE <= occupied || marked * SPARSE_NODES <= nodes
    }

    /// Whether a level with `marked` of its `occupied` nodes marked is to go
    /// by the touched rows, by [`DENSE_MARKS`].
    pub(crate) fn is_dense(marked: usize, occupied: usize) -> bool {
        marked * DENSE_MARKS > occupied
    }

    /// Whether one of the eight nodes around `(x, y)` of a grid of `cols` by
    /// `rows` nodes is marked. At the edges of the grid the clamped indices
    /// read the node itself, so that its own mark may answer true, which
    /// costs a fallback and nothing else.
    ///
    /// Not inlined: inlined into the walks, this code measured the region
    /// queries after a refresh, which never run it, 1-2% slower
    /// (`bench-results/stale-marks-2026-11-09.md`).
    ///
    /// # Safety
    ///
    /// `(x, y)` must be a node of the grid the marks were made for, which
    /// must have `cols` columns and `rows` rows.
    #[inline(never)]
    pub(crate) unsafe fn any_around(
        &self,
        (x, y): (usize, usize),
        (cols, rows): (usize, usize),
    ) -> bool {
        let (left, right) = (x.saturating_sub(1), (x + 1).min(cols - 1));
        let above = y.saturating_sub(1) * self.stride;
        let middle = y * self.stride;
        let below = (y + 1).min(rows - 1) * self.stride;
        // SAFETY: every clamped index is a node of the grid, by the caller's
        // guarantee.
        let at = |row: usize, column: usize| unsafe { *self.nodes.get_unchecked(row + column) };
        (at(above, left)
            | at(above, x)
            | at(above, right)
            | at(middle, left)
            | at(middle, right)
            | at(below, left)
            | at(below, x)
            | at(below, right))
            != 0
    }

    /// Clear every mark; the count of occupied nodes stays.
    pub(crate) fn clear(&mut self) {
        self.nodes.fill(0);
        self.loose_columns.fill(false);
        self.loose_rows.fill(false);
    }

    /// Indices of the marked nodes in row-major order (`y * cols + x`), and
    /// whether any column or row is loose.
    #[cfg(test)]
    pub(crate) fn marked(&self) -> (Vec<usize>, bool) {
        let cols = self.loose_columns.len();
        let nodes = (0..self.nodes.len())
            .filter(|&index| self.nodes[index] != 0)
            .map(|index| index / self.stride * cols + index % self.stride)
            .collect();
        let loose = self
            .loose_columns
            .iter()
            .chain(self.loose_rows.iter())
            .any(|&loose| loose);
        (nodes, loose)
    }

    /// Shrink the box of every marked node of `grid` to its items, call
    /// `refreshed` with the node and its new box, rebuild from the boxes the
    /// `columns` and `rows` spans that may have shrunk, and clear the marks.
    /// Only the rows flagged in `row_touched` are read, which must hold every
    /// mark.
    ///
    /// A span is the union of its boxes. Since the last refresh, every box
    /// only grew, except those that emptied, whose column and row
    /// [`emptied`](Self::emptied) marked; elsewhere a span can shrink only
    /// where a box held one of its ends, and only then is it rebuilt.
    pub(crate) fn refresh<N: BoxedNode>(
        &mut self,
        grid: &mut Grid<N>,
        columns: &mut [[f64; 2]],
        rows: &mut [[f64; 2]],
        row_touched: &[bool],
        mut refreshed: impl FnMut((usize, usize), &Rectangle),
    ) {
        let cols = grid.cols();
        debug_assert!(cols == columns.len() && cols == self.loose_columns.len());
        debug_assert!(grid.rows() == rows.len() && grid.rows() == self.loose_rows.len());
        let Self {
            nodes,
            stride,
            loose_columns,
            loose_rows,
            ..
        } = self;
        let gave_up = |before: f64, now: f64, end: f64| before == end && now != before;
        for (y, row_marks) in nodes.chunks_exact_mut(*stride).enumerate() {
            if !row_touched[y] {
                continue;
            }
            let (words, padding) = row_marks.as_chunks_mut::<8>();
            debug_assert!(padding.is_empty());
            // The search is a loop of its own: merged with the visits below,
            // whose registers it then had to share, it kept its position on
            // the stack and took about four times as long per word.
            let mut words = words.iter_mut().enumerate();
            while let Some((word, marks)) =
                words.find(|(_, marks)| u64::from_le_bytes(**marks) != 0)
            {
                let mut marked = u64::from_le_bytes(*marks);
                *marks = [0; 8];
                let row = &mut grid[y];
                let row_span = rows[y];
                while marked != 0 {
                    let byte = marked.trailing_zeros() as usize / 8;
                    marked &= !(0xff << (byte * 8));
                    let x = word * 8 + byte;
                    let node = &mut row[x];
                    let before = node.bounds().clone();
                    node.refresh();
                    let now = node.bounds();
                    loose_columns[x] |= gave_up(before.min_x, now.min_x, columns[x][0])
                        | gave_up(before.max_x, now.max_x, columns[x][1]);
                    loose_rows[y] |= gave_up(before.min_y, now.min_y, row_span[0])
                        | gave_up(before.max_y, now.max_y, row_span[1]);
                    refreshed((x, y), now);
                }
            }
        }
        for (y, loose) in loose_rows.iter_mut().enumerate() {
            if std::mem::take(loose) {
                let mut span = NOWHERE;
                for node in grid[y].iter() {
                    let bounds = node.bounds();
                    span = [lower(span[0], bounds.min_y), higher(span[1], bounds.max_y)];
                }
                rows[y] = span;
            }
        }
        for (x, loose) in loose_columns.iter_mut().enumerate() {
            if std::mem::take(loose) {
                let mut span = NOWHERE;
                for y in 0..grid.rows() {
                    let bounds = grid[y][x].bounds();
                    span = [lower(span[0], bounds.min_x), higher(span[1], bounds.max_x)];
                }
                columns[x] = span;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row of more marks than a byte can sum counts in full, and a node
    /// marked twice counts once.
    #[test]
    fn count_sums_the_touched_rows_however_long() {
        let cols = 600;
        let mut marks = NodeMarks::new(cols, 3);
        for x in 0..cols {
            // SAFETY: `x` is a column and 1 a row of the grid.
            unsafe { marks.mark((x, 1)) };
        }
        for _ in 0..2 {
            // SAFETY: 7 is a column and 2 a row of the grid.
            unsafe { marks.mark((7, 2)) };
        }
        assert_eq!(marks.count(&[false, true, true]), cols + 1);
    }
}
