//! Per-level geometry of the grid-based structures: exact size limits and
//! cell ranges with which `update` confirms that an item stays in its cell
//! without repeating the positioning arithmetic, and what keeps every item
//! and query within the half node that the walks and scans count on.
//!
//! An item no larger than a node and centered in it reaches at most half a
//! node past it, but only up to rounding: its width may round down to the node
//! size and its center onto a node edge while its bounds go a few floats
//! further. That takes it out of the 3x3 of a search and of the scan range of
//! another level. Only what is within [`reach_margin`] of both limits can do
//! so, a size within it of the node size and a center within it of a node
//! edge, so everything else passes as before at no cost:
//!
//! - Positioning finds the level with [`sizing_root`], which leaves out sizes
//!   within the margin of a level's limit. An item of such a size stays on
//!   the level only if [`within_reach`] says so, and otherwise goes one level
//!   up, where it is far below the limit.
//! - A search finds its level the same way, so a query that does not stay
//!   within reach starts one level up. Any coarser level gives the same
//!   results: it scans the finer ones.
//! - The update shortcut accepts only centers beyond the margin of a node
//!   edge, and leaves the rest to positioning.

use crate::overflow_bound::rounding_slack;
use crate::rect::Rectangle;
use grid::Grid;

/// How far within a limit rounding could take an item past the half node, by
/// a wide margin: four times [`rounding_slack`], of which the arithmetic needs
/// about `2^-48` of the largest root coordinate; [`within_reach`] may add one
/// slack for arithmetic it does not repeat.
pub(crate) fn reach_margin(root_bounds: &Rectangle) -> f64 {
    4.0 * rounding_slack(root_bounds)
}

/// Root size to give [`size_level`] for items and queries, for a finest level
/// of `finest_cells` nodes per axis: smaller by the margin on each of them, so
/// that a size it puts on a level is at least [`reach_margin`] below the node
/// size there. In a world so far from the origin that this leaves nothing, all
/// sizes come out on the root.
pub(crate) fn sizing_root(root_bounds: &Rectangle, finest_cells: f64) -> (f64, f64) {
    let shrink = reach_margin(root_bounds) * finest_cells;
    (
        root_bounds.get_width() - shrink,
        root_bounds.get_height() - shrink,
    )
}

/// Whether an item with `bounds` in node `(x, y)` of a level with these
/// reciprocal node sizes stays, widened by `margin`, within reach of the walks
/// and scans. With corners `f(v) = (v - origin) * 2 * inverse`, in half nodes,
/// the conditions are `f(min) >= 2x - 1` and `f(max) < 2x + 3`, except for no
/// lower one on the first node, and they do three things:
///
/// - The scans of [`scan_range`], [`first_reaching_node`] and
///   [`last_reaching_node`] compute the same `f` and reach node `x` from a
///   query ending at `f >= 2x - 1` or starting at `f <= 2x + 3`, and `f` only
///   grows with its argument, so a query that meets the item reaches its node.
/// - Two items two or more nodes apart cannot meet, nor can an item and a query
///   within reach of the node they walk from: `f(max)` of the one is below
///   `f(min)` of the other, so `max` is too. The lower limit may be met exactly
///   so that an item of exactly the node size, centered exactly on a node edge,
///   stays; the strict upper limit then decides.
/// - On a coarser level `f` halves and the cell index halves, rounded down,
///   so the conditions hold for the ancestors of the node too.
///
/// A `margin` covers the center lines of the 3x3 of the grids that compare with
/// those, which come from yet other arithmetic.
pub(crate) fn within_reach(
    bounds: &Rectangle,
    margin: f64,
    origin: (f64, f64),
    inverse: (f64, f64),
    (x, y): (usize, usize),
) -> bool {
    let axis = |min: f64, max: f64, origin: f64, inverse: f64, node: usize| {
        let half_nodes = 2.0 * inverse;
        (node == 0 || (min - margin - origin) * half_nodes >= (2 * node - 1) as f64)
            && (max + margin - origin) * half_nodes < (2 * node + 3) as f64
    };
    axis(bounds.min_x, bounds.max_x, origin.0, inverse.0, x)
        && axis(bounds.min_y, bounds.max_y, origin.1, inverse.1, y)
}

/// Per-level constants derived from the root bounds, plus the number of
/// objects currently stored on the level.
///
/// Everything except `object_count` is fixed for the lifetime of the tree.
/// The node size agrees bit for bit with the positioning arithmetic of the
/// owning structure, such as
/// [`HierarchicalLooseGrid::position`](crate::HierarchicalLooseGrid::position),
/// and the cell ranges stay clear of its rounding, so `update` can confirm that
/// an item stays in its node without repeating that computation.
#[derive(Default)]
pub(crate) struct LayerGeometry {
    // `search` reads only the first three fields of each level it visits.
    pub(crate) object_count: usize,
    /// Size of a cell: the root size divided by the larger of the grid's
    /// column and row counts, `2^(level - 1)` on the levels of a hierarchy.
    ///
    /// Also the largest object size (inclusive) that `position` assigns to
    /// this level; `matches_layer` explains why the comparison is exact.
    pub(crate) node_width: f64,
    pub(crate) node_height: f64,
    /// Half the node size: the largest object size of the next finer level.
    /// Cached; multiplying in `matches_layer` instead measured `update` 28%
    /// slower.
    pub(crate) half_node_width: f64,
    pub(crate) half_node_height: f64,
    /// Centers relative to the root origin that the update shortcut accepts
    /// for each cell along each axis: cell `i` takes the half-open interval
    /// `ranges[2 * i]..ranges[2 * i + 1]`, its own without [`reach_margin`] at
    /// both ends, except for the outer end of the first. Empty on the root,
    /// which disables the update shortcut.
    ///
    /// Boxed slices keep the struct at 72 bytes; `Vec`s make it 88 bytes,
    /// which measured `search` a few percent slower.
    pub(crate) cell_x_ranges: Box<[f64]>,
    pub(crate) cell_y_ranges: Box<[f64]>,
}

impl LayerGeometry {
    /// Geometry of `level`, whose grid has `cols` by `rows` cells.
    pub(crate) fn new(root_bounds: &Rectangle, level: usize, cols: usize, rows: usize) -> Self {
        let (node_width, node_height) = Self::node_size(root_bounds, cols, rows);

        // Root items may come from outside the world and keep full
        // positioning.
        if level == 1 {
            return Self {
                node_width,
                node_height,
                ..Self::default()
            };
        }

        let margin = reach_margin(root_bounds);
        Self {
            object_count: 0,
            node_width,
            node_height,
            half_node_width: node_width * 0.5,
            half_node_height: node_height * 0.5,
            cell_x_ranges: Self::cell_ranges(node_width, cols, margin),
            cell_y_ranges: Self::cell_ranges(node_height, rows, margin),
        }
    }

    /// Node size of a level whose grid has `cols` by `rows` cells: the root
    /// size divided by the larger count, as a grid trimmed to a non-square
    /// world keeps the cell size of the full square grid.
    ///
    /// Panics unless both are normal floats, which also rules out the zero of
    /// a root too far from the origin to keep any size. Below them the
    /// reciprocal, or twice it, which [`half_node_corners`] and
    /// [`within_reach`] multiply by, comes out infinite: positions and scans
    /// collapse onto the first and last nodes, and queries both miss items and
    /// report ones they do not meet.
    pub(crate) fn node_size(root_bounds: &Rectangle, cols: usize, rows: usize) -> (f64, f64) {
        let cells_per_axis = cols.max(rows) as f64;
        let width = root_bounds.get_width() / cells_per_axis;
        let height = root_bounds.get_height() / cells_per_axis;
        assert!(
            width.is_normal() && height.is_normal(),
            "Nodes need a normal size; the world or cell is too small, or too far from the origin."
        );
        (width, height)
    }

    // The predicates below are not generic but run inside generic tree
    // methods, which other crates instantiate; `#[inline]` keeps them
    // inlinable there. Without it, `update` measured a few percent slower.

    /// Whether `matches_layer` and `matches_cell` are usable on this level.
    #[inline]
    pub(crate) fn has_update_shortcut(&self) -> bool {
        !self.cell_x_ranges.is_empty()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.object_count == 0
    }

    /// Whether an object of this size belongs on this level, as `position`
    /// sees it for all but objects that rounding takes out of reach.
    ///
    /// That is when `(root / size) as usize` lies in `2^(level - 1)..2^level`.
    /// Comparing with the node size reproduces this exactly: `node_width` is
    /// `root / 2^(level - 1)` without rounding, so dividing the root by it
    /// yields exactly `2^(level - 1)`, and dividing by the next float above it
    /// yields less. The halved node size, also exact, bounds the next finer
    /// level in the same way.
    #[inline]
    pub(crate) fn matches_layer(&self, width: f64, height: f64, is_finest_layer: bool) -> bool {
        // Degenerate sizes fall through to full positioning.
        if !(width > 0.0 && height > 0.0) {
            return false;
        }
        let fits_this_level = width <= self.node_width && height <= self.node_height;
        if !fits_this_level {
            return false;
        }
        // The finest level also takes everything smaller; any other level
        // requires the object to be too large for the next finer level.
        is_finest_layer || width > self.half_node_width || height > self.half_node_height
    }

    /// Whether an object of at most this level's size limit, centered at this
    /// root-relative offset, goes in `cell` and stays within reach: a center
    /// beyond [`reach_margin`] of the cell's edges, where rounding can neither
    /// move it across one nor take the object past the half node. A center
    /// within the margin is left to `position`, however small the object.
    #[inline]
    pub(crate) fn matches_cell(&self, relative_center: (f64, f64), cell: (usize, usize)) -> bool {
        let (center_x, center_y) = relative_center;
        let (x, y) = cell;
        (self.cell_x_ranges[2 * x]..self.cell_x_ranges[2 * x + 1]).contains(&center_x)
            && (self.cell_y_ranges[2 * y]..self.cell_y_ranges[2 * y + 1]).contains(&center_y)
    }

    /// Whether an object of this size fits a node of this level, zero sizes
    /// included and inverted ones not. Nearly always false, so all comparisons
    /// are made up front: which axis fits varies from one object to the next,
    /// and short-circuit branches on that measured queries of 400 to 1500 units
    /// about 3% slower, with the code alignment forced.
    #[inline]
    pub(crate) fn fits_node(&self, width: f64, height: f64) -> bool {
        (width <= self.node_width) & (height <= self.node_height) & (width >= 0.0) & (height >= 0.0)
    }

    /// The ranges of [`cell_x_ranges`](Self::cell_x_ranges) along one axis. The
    /// first cell has no lower limit: negative casts saturate to it, and it
    /// has no neighbor on that side to overreach to.
    fn cell_ranges(cell_size: f64, cell_count: usize, margin: f64) -> Box<[f64]> {
        (0..cell_count)
            .flat_map(|index| {
                let low = if index == 0 {
                    f64::NEG_INFINITY
                } else {
                    cell_size * index as f64 + margin
                };
                [low, cell_size * (index + 1) as f64 - margin]
            })
            .collect()
    }
}

/// The levels of a multi-level grid, from level zero, which is unused and has
/// no cells, to `max_level`; level `L` has at most `2^(L - 1)` cells per axis.
pub(crate) struct Levels<N> {
    pub(crate) layers: Vec<Grid<N>>,
    pub(crate) root_bounds: Rectangle,
    pub(crate) geometry: Vec<LayerGeometry>,
    /// Reciprocal of the node width and height of each level, which `position`
    /// multiplies by.
    pub(crate) inverse_node_size: Vec<(f64, f64)>,
    /// Root size to find the level of an item or query from, by [`sizing_root`].
    pub(crate) size_root: (f64, f64),
}

/// The levels of a multi-level grid over `world_bounds`, with nodes made by
/// `node`. Unless `FIT`, the root is a square from the world's minimum corner,
/// and each level keeps only the rows or columns that cover the world: the
/// finest level those that the world's shorter side spans, and every coarser
/// level half as many, rounded up, so that every node has its parent. The
/// walks step from a node to its parent and read it unchecked.
///
/// Panics if `max_level` is zero, if the finest level has more cells than
/// `usize` counts, unless the world's width and height are positive and
/// finite, or if [`LayerGeometry::node_size`] rejects the nodes of a level.
pub(crate) fn build_levels<N, const FIT: bool>(
    world_bounds: &Rectangle,
    max_level: u8,
    node: impl Fn() -> N,
) -> Levels<N> {
    assert!(max_level as usize > 0, "`MAX_LEVEL` cannot be zero.");
    let n = 1usize
        .checked_shl((max_level - 1) as u32)
        .expect("`MAX_LEVEL` is too large and will cause overflow.");
    assert!(
        n.checked_mul(n).is_some(),
        "`MAX_LEVEL` is too large and will cause overflow."
    );

    let world_width = world_bounds.get_width();
    let world_height = world_bounds.get_height();
    // A world without width or height would leave the trimmed levels without
    // columns or rows, and the walks and scans, which clamp to the last node
    // of a level and read it unchecked, without a node to read.
    assert!(
        world_width > 0.0
            && world_height > 0.0
            && world_width.is_finite()
            && world_height.is_finite(),
        "The world must have a positive, finite width and height."
    );

    let (mut finest_rows, mut finest_cols) = (n, n);
    if !FIT {
        if world_width > world_height {
            finest_rows = trimmed_lines(n as f64 * world_height / world_width, n);
        } else if world_height > world_width {
            finest_cols = trimmed_lines(n as f64 / world_height * world_width, n);
        }
    }

    let mut layers = Vec::with_capacity(max_level as usize + 1);
    for level in 0..=max_level {
        // Level zero has no nodes. A node of any other level stands for
        // `span` rows and columns of the finest level, the last node of a row
        // or column for those left over, so every node has a parent. The
        // quotient of each level's own size gives the same counts as long as
        // nothing overflows or underflows on the way; a product that
        // overflows only from some level on would leave the finer levels
        // with rows or columns that have no parent.
        let (rows, cols) = if level == 0 {
            (0, 0)
        } else {
            let span = 1usize << (max_level - level);
            (finest_rows.div_ceil(span), finest_cols.div_ceil(span))
        };
        layers.push(Grid::from_vec(
            (0..rows * cols).map(|_| node()).collect(),
            cols,
        ));
    }

    let root_bounds = if !FIT {
        let len = world_width.max(world_height);
        let (min_x, min_y) = world_bounds.get_min();
        Rectangle::center_rect(min_x + len * 0.5, min_y + len * 0.5, len, len)
    } else {
        world_bounds.clone()
    };

    let mut geometry = Vec::with_capacity(layers.len());
    let mut inverse_node_size = Vec::with_capacity(layers.len());
    // Level zero is unused, like `layers[0]`.
    geometry.push(LayerGeometry::default());
    inverse_node_size.push((0.0, 0.0));
    for (level, grid) in layers.iter().enumerate().skip(1) {
        let (node_width, node_height) =
            LayerGeometry::node_size(&root_bounds, grid.cols(), grid.rows());
        geometry.push(LayerGeometry::new(
            &root_bounds,
            level,
            grid.cols(),
            grid.rows(),
        ));
        inverse_node_size.push((1.0 / node_width, 1.0 / node_height));
    }
    let size_root = sizing_root(&root_bounds, n as f64);

    Levels {
        layers,
        root_bounds,
        geometry,
        inverse_node_size,
        size_root,
    }
}

/// Rows or columns that the finest level, of `size` nodes per axis, keeps
/// along the shorter side of the world, which spans `lines` of them: that
/// rounded up, and at least one and at most all. The limits apply only
/// where the arithmetic gave out: a quotient that underflowed to zero, as
/// for a world `1e300` wide and `1e-300` high, would leave the level without
/// nodes, and one whose product overflowed, as for a world `2^1023` wide and
/// `2^1022` high, comes out infinite.
fn trimmed_lines(lines: f64, size: usize) -> usize {
    (lines.ceil() as usize).clamp(1, size)
}

/// The level an item of extent `size` along one axis of a root of extent
/// `root` belongs on, before clamping to the levels of the structure:
/// `usize::BITS - ((root / size) as usize).leading_zeros()`, the bit length of
/// how many times the item fits across the root, which every hierarchical
/// structure's `position` computes on both axes.
///
/// For a normal positive `size` and `root` it follows from the exponents
/// without the division. The rounded quotient reaches `2^k` exactly when
/// `size * 2^k <= root`, as `matches_layer` explains for `k` a level, and that
/// holds when the exponent of `size` plus `k` is below that of `root`, or
/// equal to it with a mantissa no larger. The largest such `k`, plus one, is
/// the bit length; as `size` is normal, `root / 2^k` stays normal up to that
/// `k`, which the argument of `matches_layer` needs. The result may fall
/// below zero or exceed 64 where the division's bit length stops at those
/// bounds; clamped to the levels, the two agree. Zero, subnormal, negative,
/// infinite and NaN sizes and roots keep the division.
#[inline]
pub(crate) fn size_level(root: f64, size: f64) -> i64 {
    const MANTISSA: u64 = (1 << 52) - 1;
    let (root_bits, size_bits) = (root.to_bits(), size.to_bits());
    // With the sign bit, which is clear for positive numbers; normal positive
    // numbers have a biased exponent of 1 to 2046.
    let (root_exponent, size_exponent) = ((root_bits >> 52) as i64, (size_bits >> 52) as i64);
    if (1..=2046).contains(&size_exponent) && (1..=2046).contains(&root_exponent) {
        let larger_mantissa = (size_bits & MANTISSA) > (root_bits & MANTISSA);
        root_exponent - size_exponent - larger_mantissa as i64 + 1
    } else {
        (usize::BITS - ((root / size) as usize).leading_zeros()) as i64
    }
}

/// Corners `[min_x, min_y, max_x, max_y]` of a query, given relative to the
/// root origin, in half nodes of the finest level: the minima rounded up and
/// the maxima truncated toward zero. [`scan_range`] derives the nodes to scan
/// on every level from them. Rounding each level's coordinates instead
/// measured large aligned queries 27-34% slower in the hierarchical loose,
/// column-and-row and neighbor grids, for the scans before they included
/// their ends, which took `round` and `ceil` on every level; both compile to
/// C runtime calls on x86-64 without SSE4.1. Truncation, which saves the call
/// `floor` would make, rounds the maxima down except below zero, where
/// [`scan_range`] ends at node zero for either. The minima take their
/// ceilings from truncation too, by [`ceil_from_truncation`], which saves the
/// two calls to `ceil`. Against `ceil` itself, this measured large aligned
/// queries 2-6% faster in the hierarchical loose and neighbor grids, with and
/// without code alignment forced. The mixed distribution of the hierarchical
/// loose grid measured 2-5% slower in default builds, in all six rounds of
/// two runs, as an earlier version had by 4.5%, but about 3% faster with
/// alignment forced, which points to code layout; the other queries moved
/// within the few percent by which binaries of the same code differ.
#[inline]
pub(crate) fn half_node_corners(relative: [f64; 4], finest_inverse: (f64, f64)) -> [i64; 4] {
    let (half_x, half_y) = (2.0 * finest_inverse.0, 2.0 * finest_inverse.1);
    let (min_x, min_y) = (relative[0] * half_x, relative[1] * half_y);
    [
        ceil_from_truncation(min_x, min_x as i64),
        ceil_from_truncation(min_y, min_y as i64),
        (relative[2] * half_x) as i64,
        (relative[3] * half_y) as i64,
    ]
}

/// Nodes `[min_x, min_y, max_x, max_y]` that the scan of a level `shift`
/// levels coarser than the finest visits for a query with the given
/// [`half_node_corners`], the maxima clamped to `last`, the last column and
/// row. The items of a node reach half a node past it, so for a corner `n`
/// in nodes of that level these are `ceil(n - 1.5)` and `floor(n + 0.5)`:
/// the nodes whose items reach the query, and also a node that the query
/// starts exactly half a node past or ends exactly half a node before, as an
/// item may reach exactly there and still meet such a query through rounding
/// of its corner. The scan used to stop short of that node, at `round(n) - 1`
/// and `ceil(n - 0.5)`. An item stays within these bounds by [`within_reach`],
/// which positioning checks, even where rounding takes its bounds past the
/// half node, as when its width rounds to the node size. Both bounds follow by
/// shifts alone: the reciprocal node sizes of the levels differ by exact powers of
/// two, so `n` on that level is the finest-level value scaled by `2^-shift`
/// without rounding, and floor and ceil of a quotient by a power of two
/// follow from those of the dividend. Only a minimum beyond the last node
/// may differ, for a corner at infinity or `2^63` half nodes and more, which
/// leaves the range empty either way.
///
/// A node at least two nodes in from both ends of the range lies, with the
/// half node its items may reach on each side, inside the query, ends
/// included. Its items, centered in it and no larger, then meet the query
/// without a test, points included: half a node separates their centers from
/// the query's edges. That takes corners that compare; the searches return
/// before scanning for a query with a NaN corner, which meets nothing.
#[inline]
pub(crate) fn scan_range(half: [i64; 4], shift: u32, last: (usize, usize)) -> [usize; 4] {
    // ceil(n - 1.5) is ceil(2n) / 2, rounded down, less one, and ceil(2n) on
    // this level is the finest ceil(2n) divided by 2^shift, rounded up;
    // saturating keeps a query at infinity in range.
    let min =
        |half: i64| ((half.saturating_add((1 << shift) - 1) >> (shift + 1)) - 1).max(0) as usize;
    // floor(n + 0.5) is floor(2n + 1) / 2, rounded down.
    let max = |half: i64, last: usize| {
        (((half >> shift).saturating_add(1)) >> 1).clamp(0, last as i64) as usize
    };
    [
        min(half[0]),
        min(half[1]),
        max(half[2], last.0),
        max(half[3], last.1),
    ]
}

/// First node whose items may reach a query, for `n` the query's minimum in
/// nodes of the grid or level scanned, plus half the shrink of the level's
/// bound, if any: `ceil(n - 1.5)`, zero at least, as in [`scan_range`]. The
/// items of a node reach half a node past it, and one whose bounds round a
/// float further still meets a query that starts exactly there, which the
/// scans used to leave out by starting at `round(n) - 1`. `ceil(n - 1.5)` is
/// `ceil(2n)` halved, rounded down, less one; the ceiling comes from the
/// truncation by [`ceil_from_truncation`], without a call to `ceil`. From
/// `2^62` nodes on this gives `2^62 - 2`, beyond every node.
#[inline]
pub(crate) fn first_reaching_node(n: f64) -> usize {
    let twice = 2.0 * n;
    ((ceil_from_truncation(twice, twice as i64) >> 1) - 1).max(0) as usize
}

/// Last node whose items may reach a query, for `n` the query's maximum in
/// nodes of the grid or level scanned, less half the shrink of the level's
/// bound, if any: `floor(n + 0.5)`, zero at least, as in [`scan_range`],
/// which includes a node that the query ends exactly half a node before.
/// `floor(n + 0.5)` is `floor(2n) + 1` halved and rounded down; the cast,
/// which truncates, stands in for `floor`, as the two differ only below zero,
/// where both give zero. From `2^62` nodes on this gives `2^62 - 1` where
/// the cast to `usize` would saturate higher; the scans clamp both to their
/// last node.
#[inline]
pub(crate) fn last_reaching_node(n: f64) -> usize {
    let twice = (2.0 * n) as i64;
    (twice.saturating_add(1) >> 1).max(0) as usize
}

/// `value.ceil() as i64`, from `truncated`, which is `value as i64`: one more
/// where the truncation fell short, which happens only above zero. `ceil`
/// compiles to a C runtime call on x86-64 without SSE4.1. Beyond `2^53` the
/// floats are whole and the comparison exact; the addition saturates where
/// the cast did, at `+infinity` and from `2^63` on.
#[inline]
pub(crate) fn ceil_from_truncation(value: f64, truncated: i64) -> i64 {
    truncated.saturating_add(((truncated as f64) < value) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// The definition `size_level` must reproduce.
    fn by_division(root: f64, size: f64) -> i64 {
        (usize::BITS - ((root / size) as usize).leading_zeros()) as i64
    }

    /// `(rows, columns)` of every level of the unfitted grid over `world`.
    fn shapes(world: &Rectangle, max_level: u8) -> Vec<(usize, usize)> {
        build_levels::<u8, false>(world, max_level, || 0)
            .layers
            .iter()
            .map(|grid| (grid.rows(), grid.cols()))
            .collect()
    }

    /// The trimmed levels have the rows and columns that the quotient of each
    /// level's own size gives, as before they followed from the finest level:
    /// for the worlds of the tests and benches, for sides whose quotients
    /// are whole or half numbers of nodes on some level, a float to either
    /// side of those, and for random sides.
    #[test]
    fn levels_keep_the_rows_and_columns_of_their_own_quotient() {
        let mut rng = StdRng::seed_from_u64(41);
        let mut sides = vec![
            (20000.0, 20000.0),
            (80000.0, 80000.0),
            (20000.0, 80000.0),
            (20000.0, 320000.0),
            (1024.0, 512.0),
            (999.8, 334.0),
            (300.0, 1300.0),
            (1000.0, 300.0),
        ];
        for _ in 0..20_000 {
            let long = 2f64.powf(rng.gen_range(-30.0..30.0));
            let short = match rng.gen_range(0..3) {
                // A whole number of sixty-fourths of the longer side.
                0 => long * rng.gen_range(1..64) as f64 / 64.0,
                1 => long * rng.gen_range(1..1024) as f64 / 1024.0,
                _ => long * 2f64.powf(rng.gen_range(-12.0..0.0)),
            };
            let short = match rng.gen_range(0..4) {
                0 => short.next_up(),
                1 => short.next_down(),
                _ => short,
            };
            sides.push(if rng.gen_bool(0.5) {
                (long, short)
            } else {
                (short, long)
            });
        }
        for (index, &(width, height)) in sides.iter().enumerate() {
            // Eleven levels, whose finest has a million nodes, for a few.
            let most = if index < 300 { 11 } else { 7 };
            for max_level in [1u8, 2, 7, most] {
                let world = Rectangle::new(0.0, 0.0, width, height);
                let levels = shapes(&world, max_level);
                assert_eq!(levels[0], (0, 0));
                for level in 1..=max_level as usize {
                    let size = 1usize << (level - 1);
                    let (mut rows, mut cols) = (size, size);
                    if width > height {
                        rows = (size as f64 * height / width).ceil() as usize;
                    } else if height > width {
                        cols = (size as f64 / height * width).ceil() as usize;
                    }
                    assert_eq!(
                        levels[level],
                        (rows, cols),
                        "level {level} of {max_level}, {width:e} by {height:e}"
                    );
                }
            }
        }
    }

    /// Every level below level zero has rows and columns, at most those of
    /// the full square and all of them along one axis, and half as many as
    /// the next finer level, rounded up, so that every node has a parent:
    /// also over worlds whose shorter side spans a number of rows or columns
    /// that underflows to zero, or whose product overflows from some level on.
    #[test]
    fn every_level_has_nodes_and_every_node_a_parent() {
        let huge = 2f64.powi(1023);
        let sides = [
            (1000.0, 300.0),
            (300.0, 1300.0),
            (512.0, 512.0),
            // The quotient underflows to zero.
            (1e200, 1e-200),
            (1e-200, 1e200),
            (f64::MAX, f64::MIN_POSITIVE),
            (5e-324, f64::MAX),
            // `size * height` overflows from the third level on.
            (huge, huge / 2.0),
            (huge / 2.0, huge),
            (f64::MAX, f64::MAX / 3.0),
        ];
        for max_level in [1u8, 2, 3, 7, 10] {
            for (width, height) in sides {
                let world = Rectangle::new(0.0, 0.0, width, height);
                let levels = shapes(&world, max_level);
                let context = format!("{max_level} levels, {width:e} by {height:e}");
                assert_eq!(levels[0], (0, 0), "{context}");
                for level in 1..=max_level as usize {
                    let size = 1usize << (level - 1);
                    let (rows, cols) = levels[level];
                    assert!(
                        rows >= 1 && cols >= 1 && rows.max(cols) == size,
                        "{context}: level {level} is {cols} by {rows}"
                    );
                    if level > 1 {
                        assert_eq!(
                            levels[level - 1],
                            (rows.div_ceil(2), cols.div_ceil(2)),
                            "{context}: the parents of level {level}, {cols} by {rows}"
                        );
                    }
                }
            }
        }
    }

    /// Up to `count` floats on either side of `value`, and `value` itself.
    fn around(value: f64, count: usize) -> Vec<f64> {
        let mut values = vec![value];
        let (mut up, mut down) = (value, value);
        for _ in 0..count {
            up = up.next_up();
            down = down.next_down();
            values.extend([up, down]);
        }
        values
    }

    /// `ceil(n - 1.5)` and `floor(n + 0.5)`, saturated at zero, computed
    /// exactly. `n - 1.5` and `n + 0.5` may round onto a whole number, and
    /// from `2^52` on cannot even hold the half; the whole part and the
    /// fraction, both exact, give the rounding instead, the sum taken in
    /// integers that neither round nor overflow.
    fn reaching_by_rounding(n: f64) -> (usize, usize) {
        let floor = n.floor();
        let (whole, fraction) = (floor as i128, n - floor);
        let clamp = |v: i128| v.clamp(0, u64::MAX as i128) as usize;
        (
            clamp(
                whole
                    .saturating_sub(1)
                    .saturating_add((fraction > 0.5) as i128),
            ),
            clamp(whole.saturating_add((fraction >= 0.5) as i128)),
        )
    }

    /// `scan_range` must give exactly what rounding the product with each
    /// level's own reciprocal node size gives, `ceil(n - 1.5)` and
    /// `floor(n + 0.5)`, for reciprocals built as the grids build them, and
    /// for corners on and next to node and half-node edges, outside the root,
    /// infinite and NaN. Minima beyond the last node all leave the range
    /// empty, so they are compared capped at `last + 1`.
    #[test]
    fn scan_range_agrees_with_the_rounding_of_each_level() {
        let calc_min = |n: f64| reaching_by_rounding(n).0;
        let calc_max = |n: f64, last: usize| reaching_by_rounding(n).1.min(last);
        let mut rng = StdRng::seed_from_u64(9);
        let roots = [
            Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
            Rectangle::new(-128.0, -128.0, 896.0, 384.0),
            Rectangle::new(0.3, -0.7, 1000.1, 333.3),
            Rectangle::new(-3.7e5, 1.1e4, 2.9e6, 1.7e6),
        ];
        for max_level in [1usize, 2, 7, 10] {
            for root in &roots {
                // Reciprocals of every level as the grids compute them.
                let inverses: Vec<(f64, f64)> = (1..=max_level)
                    .map(|level| {
                        let cells = 1usize << (level - 1);
                        let (width, height) = LayerGeometry::node_size(root, cells, cells);
                        (1.0 / width, 1.0 / height)
                    })
                    .collect();
                let finest = inverses[max_level - 1];
                for _ in 0..20_000 {
                    // A coordinate in finest half nodes, often on or a few
                    // floats away from an edge, sometimes far outside.
                    let mut coordinate = |inverse: f64| -> f64 {
                        let half = 0.5 / inverse;
                        let value = match rng.gen_range(0..10) {
                            0 => rng.gen_range(-3.0..3.0) * half,
                            1 => f64::INFINITY,
                            2 => f64::NEG_INFINITY,
                            3 => f64::NAN,
                            4..=6 => rng.gen_range(-4i64..(4 << max_level)) as f64 * half,
                            _ => rng.gen_range(-0.1..1.1) * half * (2 << max_level) as f64,
                        };
                        let mut value = value;
                        for _ in 0..rng.gen_range(0..3) {
                            value = if rng.gen_bool(0.5) {
                                value.next_up()
                            } else {
                                value.next_down()
                            };
                        }
                        value
                    };
                    let relative = [
                        coordinate(finest.0),
                        coordinate(finest.1),
                        coordinate(finest.0),
                        coordinate(finest.1),
                    ];
                    let half = half_node_corners(relative, finest);
                    for level in 1..=max_level {
                        let (inverse_x, inverse_y) = inverses[level - 1];
                        let last = (1usize << (level - 1)) - 1;
                        let expected = [
                            calc_min(relative[0] * inverse_x).min(last + 1),
                            calc_min(relative[1] * inverse_y).min(last + 1),
                            calc_max(relative[2] * inverse_x, last),
                            calc_max(relative[3] * inverse_y, last),
                        ];
                        let shift = (max_level - level) as u32;
                        let [min_x, min_y, max_x, max_y] = scan_range(half, shift, (last, last));
                        assert_eq!(
                            [min_x.min(last + 1), min_y.min(last + 1), max_x, max_y],
                            expected,
                            "level {level} of {max_level}, root {root:?}, corners {relative:?}"
                        );
                    }
                }
            }
        }
    }

    /// `first_reaching_node` and `last_reaching_node` must give what
    /// `ceil(n - 1.5)` and `floor(n + 0.5)`, saturated at zero, give: for `n`
    /// on, next to and halfway between whole nodes, negative, near the powers
    /// of two where floats stop holding halves and where the integer casts
    /// saturate, random over the whole range of floats, and the special
    /// values. Results from `2^61` nodes on, beyond the last node of any grid,
    /// are compared capped there, as the scans clamp them to their last node
    /// or leave the range empty.
    #[test]
    fn reaching_nodes_agree_with_rounding() {
        let first = |n: f64| reaching_by_rounding(n).0;
        let last = |n: f64| reaching_by_rounding(n).1;
        let cap = 1usize << 61;
        for value in special_values(31) {
            for n in around(value, 3).into_iter().flat_map(|n| [n, -n]) {
                assert_eq!(
                    first_reaching_node(n).min(cap),
                    first(n).min(cap),
                    "first, n = {n:e}"
                );
                assert_eq!(
                    last_reaching_node(n).min(cap),
                    last(n).min(cap),
                    "last, n = {n:e}"
                );
            }
        }
    }

    /// The special values, whole and half numbers around zero, the powers of
    /// two up to where the saturating cast stops and their neighbors, and
    /// random values of every magnitude drawn with `seed`.
    fn special_values(seed: u64) -> Vec<f64> {
        let mut values = vec![
            0.0,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
        ];
        for whole in -10..=10 {
            values.extend([whole as f64, whole as f64 + 0.5]);
        }
        for power in 0..70 {
            let value = 2f64.powi(power);
            values.extend([value, value + 0.5, value - 0.5, value * 1.5]);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..20000 {
            values.push(rng.gen_range(-3.0..300.0));
            values.push(rng.gen_range(-3i32..300) as f64 * 0.5);
            values.push(2f64.powf(rng.gen_range(-1074.0..1023.0)) * rng.gen_range(1.0..2.0));
        }
        values
    }

    /// `ceil_from_truncation` must give what `ceil` and the saturating cast
    /// give: for values on, next to and halfway between whole numbers, on both
    /// sides of zero, near the powers of two where floats stop holding
    /// fractions and where the cast saturates, random over the whole range of
    /// floats, and the special values.
    #[test]
    fn ceil_from_truncation_agrees_with_ceil() {
        for value in special_values(29) {
            for v in around(value, 3).into_iter().flat_map(|v| [v, -v]) {
                assert_eq!(
                    ceil_from_truncation(v, v as i64),
                    v.ceil() as i64,
                    "v = {v:e}"
                );
            }
        }
    }

    /// Every level threshold `root / 2^k`, a few floats around it, random
    /// sizes over the whole range of floats and the special values all give
    /// the bit length of the division, up to the bounds where it stops.
    #[test]
    fn size_level_agrees_with_the_division() {
        let roots = [
            1024.0,
            20000.0,
            999.8,
            334.0,
            512.0,
            1e-10,
            3.7e5,
            1e300,
            // Roots whose thresholds `root / 2^k` underflow for large `k`.
            2f64.powi(-958),
            1e-300,
            f64::MIN_POSITIVE,
        ];
        let specials = [
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE / 2.0,
            5e-324,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -1.0,
            -1e-300,
        ];
        let mut rng = StdRng::seed_from_u64(17);
        for root in roots {
            let mut sizes: Vec<f64> = specials.to_vec();
            sizes.extend(around(root, 4));
            sizes.extend(around(root * 2.0, 4));
            for k in 0..70 {
                sizes.extend(around(root / 2f64.powi(k), 4));
            }
            for _ in 0..20000 {
                let exponent = rng.gen_range(-1074.0..1023.0);
                sizes.push(2f64.powf(exponent) * rng.gen_range(1.0..2.0));
            }
            for size in sizes {
                assert_eq!(
                    size_level(root, size).clamp(0, 64),
                    by_division(root, size),
                    "root {root:e}, size {size:e}"
                );
            }
        }
    }
}
