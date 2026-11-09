# Overflow-Based Pruning

Pruning region queries in hierarchical loose grids by padding each search with measured per-direction bounds on how far stored items extend beyond their cells, instead of the fixed half cell, kept per level, per column and row, and per neighboring cell. Implemented in Rust for moving 2D axis-aligned boxes.

## Motivation

A hierarchical loose grid assigns each item to the level matching its size and, within that level, to the cell containing its center. An item may thus extend up to half a cell beyond its cell, and every search pads each cell by that amount. The padding reflects the worst case rather than the stored items; measuring the actual maximum extent on each level yields a per-level looseness.

A single value per level, however, conflates four directions. An item near the right edge of its cell extends only rightward, while the search consults each direction separately: cells to the left of a query matter only as far as their items extend rightward, and cells to the right only as far as theirs extend leftward. Keeping the maximum extent of each direction apart yields four bounds per level, the **directional overflow bounds**, so that a large extent in one direction no longer widens the other three.

A per-level bound is set by the farthest-reaching item on the level and, on densely populated levels, approaches the worst case again. The bounds are therefore refined to ever smaller groups of items, each consulted where the search selects cells:

| Bounds kept | Structure |
|---|---|
| None: half a cell in every direction | `HierarchicalLooseGrid`; `BoxedLooseGrid` adds a box per cell |
| Per level | `ExactOverflowBoundGrid` (with an exact box per cell), `LazyOverflowBoundGrid` (with a box per cell) |
| Per column and per row | `SpanOverflowBoundGrid` |
| Per neighbor of every cell | `HaloOverflowBoundGrid` |

On 10,000 balls of mixed sizes, each queried with its own bounds, per-neighbor bounds halve the search time of the plain grid (1.00 ms vs. 1.98 ms) at twice its update cost (149 µs vs. 76 µs per 10,000 moves, including one refresh). Full measurements are given under [Results](#results).

## The hierarchical loose grid

`HierarchicalLooseGrid` is the base shared by all bounded grids.

**Positioning.** Level `L` (1 ≤ `L` ≤ `MAX_LEVEL`) divides the root into `2^(L-1)` cells per axis. Each item is assigned in O(1) to the finest level whose cells are no smaller than the item, and to the cell containing its center. Items smaller than the finest cells stay on the finest level; items centered outside the world or larger than a level-2 cell go to the root. Being no larger than its cell and centered in it, an item extends at most half a cell beyond each edge, so each cell acts as a loose cell of twice its size. Rounding can break this by a few floats, when an item's width rounds to the cell size and its center onto a cell edge: such an item goes one level coarser, and such a query starts one level coarser, which gives the same results.

**Search.** The query is positioned like an item, at level `q`.

- On each level finer than `q`, the cells within half a cell of the query are scanned. Cells at least two cells inside this range lie, loose bounds included, entirely within the query; their items are reported without intersection tests.
- On level `q` and above, the query spans at most one cell, so only the cell containing its center and the eight around it can hold intersecting items; these cells follow the ancestor path of the query's cell. The search walks this path and visits a neighbor only if the query crosses the cell's center line toward it, which selects the same cells as half-cell padding.
- The root is scanned in full.

No node bounds are tested, so nodes store only their items. As levels are visited from the finest upward, smaller items tend to be reported first, though items within a level are unordered.

**Handles.** `insert` returns an `ItemHandle` recording the item's location, so `update` and `remove` reach it in O(1) without hashing or scanning its node. An update within the same cell only rewrites the bounds. Items need only be `Copy`.

## Directional bounds per level

For an item `b` stored in cell `[x0, x1] × [y0, y1]`, with top denoting negative y:

```text
left = x0 - b.min_x      right  = b.max_x - x1
top  = y0 - b.min_y      bottom = b.max_y - y1

L, R, T, B = maxima of left, right, top and bottom over the items of a level
```

Values are negative where an item stays inside the edge. The left side of the search uses `R`, the right side `L`. With the origin at zero, cell width `s` and a query `[a, b]` in x:

```text
finer levels: scan columns floor((a - R) / s) through ceil((b + L) / s) - 1

ancestor walk, in the query's cell [x0, x1]:
    visit the left neighbors  only if a < x0 + R
    visit the right neighbors only if b > x1 - L
```

Rows use `B` above the query and `T` below. Each bound is widened by a rounding margin of `2^-40` times the largest root coordinate and capped at half a cell, where the rules reduce to those of the plain grid.

Relative to the plain grid, and before the second to fourth changes noted below the results tables, these bounds cut search time by 46% for balls well inside the finest cells, by 56% and 32% for small and large queries over items near their cell centers, and by 27% to 32% for the ball queries of the two-tier scenes. On the mixed distribution they had no effect, as some items on every level extend nearly half a cell. A single looseness per level, the largest of the four bounds applied in all directions, performed equally on scenes with symmetrically placed items and 3% to 8% worse on the two-tier scenes and the mixed distribution.

**Negative bounds.** If every item on a level keeps clear of a cell edge, the bound is negative, and the search crosses that edge only when the query exceeds it by more than the clearance. On the aligned scene, whose items stay at least 36 units inside their 312.5-unit cells, admitting negative bounds made small queries 17% to 20% faster than clamping them at zero.

**Maintenance.** The exact and the lazy bounds share the same search.

- `ExactOverflowBoundGrid` keeps the bounds exact. Each level and direction holds a counted multiset of all item distances: a compressed radix-64 bitmap over order-preserving 64-bit keys ([`bitmap_max_multiset.rs`](src/bitmap_max_multiset.rs)), with at most 11 branches per path, giving O(1) insertion, removal and maximum for the fixed key width. Each item contributes four entries, and a move replaces those that change, usually all four; updating 10,000 balls takes 4.36 ms.
- `LazyOverflowBoundGrid` keeps lazy bounds. It raises a bound on insertion and update, with one comparison per direction, and leaves it unchanged on removal; `refresh_bounds` recomputes it from the items, e.g. once per frame. A raised bound remains valid as items move, shrink or leave, so results are always exact and only pruning degrades until the next refresh. Until 2026-10-26 `LazyOverflowBoundGrid` kept these bounds without cell boxes and updated 10,000 balls in 126 µs, refresh included; the grid with cell boxes, then `BoxedOverflowBoundGrid`, replaced it and took its name.

**Cell boxes.** `LazyOverflowBoundGrid` adds to the lazy per-level bounds a box per node covering its items, grown on insertion and update and shrunk by `refresh_bounds`. `BoxedLooseGrid` keeps the same boxes without level bounds, on the fixed reach of `HierarchicalLooseGrid`. `ExactOverflowBoundGrid` keeps its boxes exact instead: removal, moving out and an update within the node recompute the box from the node's items whenever the old bounds touched one of its sides. The level bounds select the nodes to visit; a visited node whose box misses the query is skipped without reading its items. The ancestor walk also stops at the coarsest non-root level that may hold items. Relative to the lazy grid, which did not stop there at the time, search time fell by 19% on the mixed distribution, 15% on small balls, 26% on small aligned queries and 33% for the rectangle queries of the adjacent scene.

## Directional bounds per column and row

`SpanOverflowBoundGrid` maintains for every column of a level the span from the smallest to the largest x of its items, and likewise in y for every row. The ends of a column's span are its own `L` and `R` in absolute coordinates; rows carry `T` and `B`. The ancestor walk visits a neighboring column or row only if its span meets the query, the scan of finer levels drops boundary columns and rows whose spans miss the query, and the cell boxes of the boxed grid skip the remaining nodes that miss it.

Spans consist of item coordinates, so comparisons need neither cell edges nor a rounding margin, and maintenance needs no arithmetic. On sparse levels most columns and rows are empty and never visited; on dense levels a column still spans many items, but no longer stands for the entire level.

Insertion and update widen the node's box and the spans of its column and row, and mark the level, the row and the node as touched. `refresh_bounds` skips untouched levels. Where at most half the nodes of a level are marked, it shrinks only the boxes of the marked nodes, and rebuilds a column or row span only where a marked node held one of its ends and no longer reaches it, or a node in it emptied. Elsewhere it shrinks the boxes and rebuilds the spans of touched rows only, reading the boxes of other rows solely for the column spans; when moving items stay within a band of rows, or a large static population occupies its own level, that costs about a quarter of a full sweep. Both ways leave the boxes and spans computed from the items.

Relative to the lazy per-level bounds, and before the second change noted below the results tables, search time fell by 38% on the mixed distribution, 24% on small balls, 34% on small aligned queries, 23% and 26% for the ball queries of the two-tier scenes and 24% for the rectangle queries of the adjacent scene, and rose by 9% on large aligned queries. Updating 10,000 balls takes 140 µs, vs. 126 µs for the lazy grid.

## Directional bounds per neighboring cell

With per-column and per-row bounds, nearly all remaining waste lies in the ancestor walk: in a counting model of the mixed distribution, a query visits 12.0 nodes, 11.4 of them in the walk and 9.5 of those to no effect, since the span of a dense column or row almost always meets the query.

`HaloOverflowBoundGrid` stores at every node the bounds of its eight neighbors facing it: the largest `max_x` among the items of its left neighbor, the smallest `min_x` of its right neighbor, their counterparts in y, and two values per diagonal neighbor, twelve in total. Since the walk reads the query's own node on every level anyway, these values alone decide which neighbors to visit. They are kept in an array beside the grid rather than in the nodes, so that nodes remain as compact as in the column-and-row grid for scans and updates.

Insertion and update maintain boxes and spans as in the column-and-row grid and mark the level stale and the row touched; the copies held by neighbors are left as they are. On a stale level, queries in or next to a touched row judge neighbors by the spans, all others by the copies. `refresh_bounds` shrinks the boxes of touched rows and recopies the facing sides into those rows and their neighbors, or, where at most half the nodes of a level are marked, shrinks only the marked nodes' boxes and writes their sides into the copies their eight neighbors hold. Before a refresh, search therefore runs at the speed of the column-and-row grid (rows "before a refresh" below).

**Refreshing only the changed nodes** (2026-11-01). The four grids with spans, `SpanOverflowBoundGrid`, `HaloOverflowBoundGrid`, `SpanGrid` and `HaloGrid`, also mark every node that an item is stored in, replaced in or taken from, and count the nodes that hold items ([`src/node_marks.rs`](src/node_marks.rs)). Where at most half the occupied nodes of a level, or an eighth of all its nodes, are marked, `refresh_bounds` visits only the marked nodes; elsewhere it sweeps the touched rows as before. On the frame bench (pairs mode, mixed distribution), the updates and refresh of a frame with 1% of the balls moving took 83–85% less time with 10,000 balls and 86–87% less with 160,000 (`HaloOverflowBoundGrid`: 29.9 → 4.7 µs and 0.65 → 0.09 ms), and 5–12% less with a quarter moving. With every ball moving they took 1–4% more, and the update benchmarks 2–10% more, reinsertion in the single-level grids the most. Details in `bench-results/cell-refresh-2026-11-01.md`.

Relative to the per-column and per-row bounds, search time fell by 20% on the mixed distribution, 39% for the rectangle queries of the adjacent scene, 16% and 9% for the ball queries of the two-tier scenes and 4% on small balls, and rose by 4% on small aligned queries, where the spans already exclude everything. Updates cost 1.03 to 1.12 times those of the column-and-row grid, as the refresh copies the facing sides of eight neighbors per node. The structure suits frames that update, refresh and then query.

## Results

Measured in a single session on an AMD Ryzen 7 9800X3D (rustc 1.99.0, Windows 11) with Criterion defaults. World 20,000 × 20,000, 7 levels, finest cells 312.5 units wide. Columns: `HierarchicalLooseGrid` (none), `LazyOverflowBoundGrid` as it was before 2026-10-26, without cell boxes, and `ExactOverflowBoundGrid` (per level, the latter also measured before it kept cell boxes), `BoxedOverflowBoundGrid` (now `LazyOverflowBoundGrid`), `SpanOverflowBoundGrid` (per column and row) and `HaloOverflowBoundGrid` (per neighbor).

Search time in µs:

| Scene and queries | None | Level, lazy | Level, exact | Boxed | Column and row | Neighbor |
|---|---:|---:|---:|---:|---:|---:|
| Mixed distribution, 10,000 | 1978 | 2029 | 2040 | 1643 | 1254 | 1003 |
| Small balls, 10,000 | 1256 | 684 | 683 | 579 | 522 | 500 |
| Aligned, 10,000 small | 719 | 319 | 318 | 237 | 212 | 220 |
| Aligned, 2,000 large | 327 | 223 | 217 | 220 | 242 | 233 |
| Gap, 10,000 balls | 1567 | 1146 | 1131 | 1036 | 886 | 810 |
| Adjacent, 10,000 balls | 1759 | 1191 | 1195 | 1006 | 878 | 739 |
| Adjacent, 1,024 rectangles | 292 | 216 | 220 | 145 | 164 | 100 |
| Gap, balls before a refresh | | | | 580 | 499 | 501 |
| Adjacent, balls before a refresh | | | | 571 | 512 | 517 |

The 64 rectangle queries of the gap scene take 27 to 31 µs in every structure.

Item management in µs. `update` moves all 10,000 balls one step and refreshes once; `reinsert` removes and reinserts every ball; the last three rows apply 20,000 precomputed updates (small moves, one-cell moves, moves with resizing) with two refreshes.

| Operation | None | Level, lazy | Level, exact | Boxed | Column and row | Neighbor |
|---|---:|---:|---:|---:|---:|---:|
| `update` | 76 | 126 | 4359 | 147 | 140 | 149 |
| `reinsert` | 271 | 327 | 4231 | 368 | 385 | 398 |
| Small moves | 119 | 200 | 8547 | 237 | 223 | 250 |
| One-cell moves | 385 | 490 | 9414 | 554 | 557 | 578 |
| Moves with resizing | 188 | 298 | 8874 | 322 | 318 | 333 |

These tables predate seven changes. First, every grid now finds an item's level from the float exponents of its size and of the root, and its cell by multiplying with the reciprocal of the cell size, where most of them divided before. In later runs (three interleaved rounds, 2 s measurements), `reinsert` took 174 µs without bounds (`HierarchicalLooseGrid`) and 212, 250, 256 and 279 µs for the lazy, boxed, column-and-row and neighbor grids, 30% to 36% less than above. `update` and the precomputed moves stayed within 5%, and search became up to 17% faster on aligned queries. `SpanGrid` reinserts in 215 µs and `HaloGrid` in 223 µs. Second, the grid without bounds and the grids with exact and lazy level bounds now also stop the ancestor walk at the coarsest level below the root that holds items (for the lazy grid, that may hold items until the next refresh), as the other three overflow-bound grids already did. Over two runs of three interleaved rounds, this cut search time on small aligned queries by 40-44% without bounds, 21-24% with exact and 21-31% with lazy bounds. Without bounds it also cut search time on small balls by 23-24%, on large aligned queries by 18-21% and in the adjacent scene by 9-14%, and with lazy bounds on the rectangle queries of the adjacent scene by 7-13%. Every other difference was under 5% in at least one of the runs, within the noise of the unchanged boxed grid; this includes the mixed distribution, where only level 2 is empty, the gap scene and the self query. Without bounds the gain is larger because there the step to the parent cell compiles to a branch on the parity of the cell coordinates, which random queries cannot predict, while the bounded grids compute it without a branch; making only that step branch-free gained as much in a separate measurement. Updates stayed within 3%, except that one-cell moves without bounds were about 2% slower in every run. Third, the grid without bounds now skips that step after the coarsest level, where its result is unused. In three more runs this cut its search time by a further 6-7% on small balls, 13-16% on small aligned queries and 4-6% for the ball queries of the adjacent scene; the other differences stayed within the 3% by which the same source varies between binaries. A branch-free step with the same skip was a trade-off: 8-9% faster for the ball queries of the gap scene, which cross three empty levels, but 2% slower on the mixed distribution and 3% slower in the batched self query. In the bounded grids, the same skip turned their branch-free step into a branch and made scenes with empty levels 8-37% slower, so they keep the step. Fourth, the grid without bounds and the column-and-row and neighbor grids now compute the query's corners once, in half cells of the finest level, and derive the scan range of every finer level by shifts, bit for bit what rounding on each level gave. In three runs this cut search time on large aligned queries by 32-34%, 27% and 31-32%, and by 12-15% for the rectangles of the gap scene without bounds; the mixed distribution became 2-4% faster. On x86-64 without SSE4.1, `round`, `floor` and `ceil` compile to C runtime calls. Called in isolation, `round` takes about 6 ns against 1.3 ns for `floor`, but in a scan, where its result bounds the loop, it measured about 16 ns: the Windows C runtime's `round` clears the fraction with 2-byte stores and reads the result back in one 8-byte load, which store forwarding cannot serve, and branches on whether the fraction is one half, which random queries mispredict. Fifth, the grids with exact, lazy and boxed bounds and `SpanGrid` and `HaloGrid`, which still called `round` once per axis for every scanned level or query, now take the first node of the scan from `floor`, with the same result. In three interleaved rounds, confirmed by a reviewer's runs, this cut search time on large aligned queries by 23-29% in all five, and on the rectangle queries of the adjacent scene by 8-15% with exact and lazy bounds and by about 17% with boxes, measured in one process, as criterion's figures for the boxes swing between 29% and 47% in this bimodal scene. The batched self query of `SpanGrid` became 1-3% slower, from a change in register allocation; the other differences stayed within the noise of the unchanged grids. Sixth, every grid now truncates where it took `floor`: for the first node of the scans above and for the minima of the corners from which the grid without bounds and the column-and-row and neighbor grids derive their scans. Truncation differs from `floor` only below zero, where every scan range clamps to its first node alike, so results are unchanged. In three to five interleaved rounds, and in five with code alignment forced, this cut search time on large aligned queries by a further 4-11% in the grids with bounds and in `SpanGrid` and `HaloGrid` and by 2% without bounds. Small aligned queries, which do not scan and run none of the changed code, moved by -6% to +10% depending on the grid, from changes in the generated code; with alignment forced, the boxed grid stayed about 5% slower there. Seventh, the column-and-row and neighbor grids now grow their cell boxes and their column and row spans with a comparison and a select instead of `f64::min` and `f64::max`, which on x86-64 spend six more instructions each on handling a NaN receiver; these extents, grown from infinity, never hold one. In three interleaved rounds, with and without code alignment forced, and in six rounds by a reviewer, this cut update times by 3-13% in both grids, with search unchanged. The column-and-row grid needed its span update forced inline, which the shorter code had otherwise turned into a call. The comparisons in the sections above were measured before the second to seventh changes. In a run after the fourth change, the exact bounds cut search time relative to the grid without bounds by 29% instead of 46% on small balls, by 34% instead of 56% on small aligned queries and by 25% and 17% instead of 27% to 32% for the ball queries of the gap and adjacent scenes, but were 9% slower instead of 32% faster on large aligned queries, whose scan the exact grid then still rounded with `round` on every level; after the fifth change they were 14-19% faster there, and after the sixth about 23%. Relative to the lazy bounds, the boxes cut search time on small aligned queries by 1-17% instead of 26%, depending on the run, and the column and row bounds by 15-18% instead of 34%.

**Selection.** All structures return exact results at all times, and refreshing only restores pruning. The loose grids search a fixed reach of half a cell without a rounding margin; positioning keeps every item within it, by the same arithmetic as the scans, so that an item that overlaps a query by a few units in the last place, right at that reach, is still found.

- Update, refresh, then query each frame: `HaloOverflowBoundGrid`.
- Queries interleaved with updates, without a natural refresh point: `SpanOverflowBoundGrid`, refreshed when convenient, or `HierarchicalLooseGrid` when updates dominate.
- Exact per-level extents at any time (`layer_overflow`): `ExactOverflowBoundGrid`, at a high update cost.
- Items of uniform size: `UniformGrid` or `SpanGrid`, with the cell size tuned to the items.

## Usage

The library target is named `lib`.

```rust
use lib::rect::Rectangle;
use lib::*;

const MAX_LEVEL: u8 = 7;

let world = Rectangle::new(0.0, 0.0, 20000.0, 20000.0);
let mut grid = HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(world);

let handle = grid.insert(Rectangle::center_rect(100.0, 100.0, 50.0, 50.0), 0);
grid.update(Rectangle::center_rect(103.0, 101.0, 50.0, 50.0), handle);
grid.refresh_bounds(); // Once per frame, after the updates.

grid.search(&Rectangle::center_rect(120.0, 90.0, 80.0, 80.0), |id| println!("{id}"));
let bound = grid.overflow_bound(MAX_LEVEL as usize); // Left, right, top, bottom.
grid.remove(handle);
```

- `new` uses square cells spanning the larger side of the world; `new_fit` fits the cells to the world.
- Using a handle after `remove` panics, even if its slot has been reused.
- `refresh_bounds` is provided by the lazy structures: `BoxedLooseGrid`, `LazyOverflowBoundGrid`, `SpanOverflowBoundGrid`, `HaloOverflowBoundGrid`, `SpanGrid` and `HaloGrid`.
- Every structure except `LooseQuadTree` also provides `for_each_self_query(|item, found| ...)`, whose results are those of searching with the bounds of every stored item, the item itself included. It runs the walk of `for_each_pair` and reports every pair in both orders and every item with itself where its bounds meet themselves, so a result is found by the item that the pair rules let find it (the coarser one in the hierarchical grids, the overflow item in `SpanGrid` and `HaloGrid`, the smaller one in `MultiGrid`), the results of one query are not reported together. The walk visits every cell (in the multi-level grids, on every level down to the finest one holding items), so it pays off only when most cells hold items. On the mixed distribution the `self_query` bench group (10,000 items) measured 0.27 ms for `HaloOverflowBoundGrid`, 0.33-0.38 ms for the other overflow-bound grids and `BoxedLooseGrid`, 0.53 ms for `HierarchicalLooseGrid`, 0.45 ms for `UniformGrid` and 0.31 ms for `TwoLayerGrid` with 312.5-unit cells, and 0.39 and 0.33 ms for `SpanGrid` and `HaloGrid`: 53-86% less than when every item searched with its bounds in turn, cell by cell (0.78-2.44 ms). `MultiGrid` gains least, 27% (1.49 to 1.09 ms in a probe); its self-query now takes about as long as its pair walk. Searching item by item in random order (`self_query/loop`) takes 0.92 ms in `HaloOverflowBoundGrid` and 1.02 ms in `UniformGrid`. The `pairs` group below calls back through `&mut dyn FnMut` and the `self_query` group through a closure, so their times do not compare.
- Every structure provides `for_each_pair(|a, b| ...)`, which reports every intersecting pair of distinct items once, through functions of its own that share no code with the searches. The hierarchical grids let each item look for partners only on its own level and below: on its own level only in the half stencil after its node (the items after it, the neighbor to the right and the row below), and on the finer levels by a copy of the search's scan, so each pair across levels is found by its coarser item; `UniformGrid` tests a pair only in the first cell both ranges share, `TwoLayerGrid` joins only the classes A-A, A-B, A-C, A-D and B-C within a cell, `LooseQuadTree` searches only the nodes after the item's own in preorder, `SpanGrid` and `HaloGrid` use the half stencil in the cells and let each item of the overflow node scan the cells as a search does, and `MultiGrid` follows LAMMPS, with a half stencil within a collection and a one-way search from smaller to larger collections. On the mixed distribution (10,131 pairs) `HaloOverflowBoundGrid` takes 0.28 ms (69% less than searching item by item), `TwoLayerGrid` 0.32 ms, `SpanOverflowBoundGrid` 0.34 ms, the grids with boxes or per-level bounds 0.38-0.39 ms, `UniformGrid` 0.41 ms with 312.5-unit cells and `HierarchicalLooseGrid` 0.55 ms. Letting the finer item walk up to the coarser levels instead, as a search does, took 2.0-2.8 times as long, since nearly all items lie on the finest level. In `SpanGrid` and `HaloGrid` with 312.5-unit cells, where the largest balls overflow, letting every cell item read the overflow node took 5.1 and 5.7 times as long (2.10 against 0.41 ms, 1.98 against 0.35 ms).
- In the three lazy overflow-bound grids, `overflow_bound(level)` returns a level's current bounds, for levels 2 to `MAX_LEVEL`; `ExactOverflowBoundGrid::layer_overflow(level)` returns the exact maxima, for levels 1 to `MAX_LEVEL`. Empty levels report negative infinity.

## Other structures

- [`LooseQuadTree`](src/loose_quadtree.rs): a loose quadtree with conventional top-down search over node bounds. Mixed distribution 3.54 ms; update 77 µs; reinsertion 413 µs.
- [`UniformGrid`](src/uniform_grid.rs): a single level registering each item in every cell it overlaps. With 312.5-unit cells it takes 1.03 ms on the mixed distribution and is fastest on large aligned queries (138 µs), but it depends on the cell size, and one-cell moves and resizing cost 628 and 701 µs, vs. 578 and 333 µs with per-neighbor bounds.
Structures of other research live in `lib::external` ([`src/external`](src/external/mod.rs)): ports of published methods, and adapters of external libraries behind cargo features.

- [`TwoLayerGrid`](src/external/two_layer_grid.rs): the two-layer partitioning of Tsitsigkos, Bouros, Mamoulis et al. (ICDE 2021, TKDE), as in `UniformGrid` a single level registering each item in every cell it overlaps, with the entries of every cell split into four classes by whether the item starts in the cell or before it along each axis. A search reads only the classes not met in an earlier cell of its range, so it reports each item once without a deduplication test, and tests only the query sides that a cell's position in the range leaves open; cells strictly inside the range report their entries untested. Entries hold the bounds, so an update within the same cell range rewrites every copy. Not yet measured.
- [`MultiGrid`](src/external/multi_grid.rs): size-class grids after the `multi` neighbor style of LAMMPS (Shire, Hanley and Stratford 2021; Monti et al. 2022). Items go by their larger side to one of a few collections, each a uniform grid whose cell size is the collection's size limit, and are registered once by center; a search scans every collection over the query padded by half a cell. With three log-spaced collections (cells 200, 800 and 3200) the mixed distribution takes 1.81 ms and 86 µs per update, with two (400 and 3200) 1.79 ms and 73 µs: 6-7% faster to search than `HierarchicalLooseGrid` and about twice the time of `HaloOverflowBoundGrid`. The pair-once rules of LAMMPS serve pair lists and have no counterpart in region queries.
- [`SpatialJoinSweep`](src/external/spatial_join_sweep.rs): the candidate phase of the sweep of Freiburg's spatialjoin (SIGSPATIAL 2025), in memory: x events, and an active set indexed in y by `IntervalIdx` (spans in decimal buckets, each sorted by left end). Pairs only; 2.0-3.3 times as long per frame as `SortAndSweep` in the square scenes, faster than it in a world 16 times as tall (`bench-results/spatial-join-sweep-2026-10-31.md`).
- [`ParryBvh`](src/external/parry_bvh.rs) (feature `parry`, on by default): the BVH of parry2d-f64 0.31.1, Rapier's broad-phase tree, rebuilt (binned SAH or PLOC), refitted, refitted and incrementally optimized in either order, or reinserted at each refresh, with optional fat margins; exact between refreshes. In the frame benchmark its best setting (incremental optimization, then a refit) takes 1.40-2.24 times as long per frame as `HaloOverflowBoundGrid`, and parry driven directly, without the adapter, 1.25-2.03 times (`bench-results/frame-parry-audit-2026-10-31.md`).
- [`PackedRTree`](src/external/packed_rtree.rs) (feature `packed`, on by default): packed R-trees of static_aabb2d_index 2.1.0 (Hilbert-sorted, as flatbush) or geo-index 0.4.0 (Hilbert or STR), rebuilt whole at each refresh after a change, since neither library can change a built tree; exact between refreshes. Options: add the items in the order of the last tree, so that the sort finds them nearly sorted; rebuild only once a share of the entries has changed; pairs by a self-join written on static_aabb2d_index's public node arrays. Adding in the last tree's order, with the self-join for pairs, when every ball moves it takes 1.04-1.32 times as long per frame as `HaloOverflowBoundGrid` (160k balls on an idle host; 1.48-3.14 times as the libraries are meant to be used), faster than parry in the same batch; when only 1% move, the rebuild still costs 86% as much and the frame 1.36-1.59 times as long, while the maintenance of `HaloOverflowBoundGrid`, `UniformGrid` and parry with partial refits falls with the changes. Not suited to incremental updates; the pairs of a static scene are as fast as `HaloOverflowBoundGrid`'s or slightly faster (`bench-results/packed-rtree-2026-10-31.md`).
- [`SpanGrid`](src/span_grid.rs): per-column and per-row bounds on a single level of chosen cell size, for scenes whose items all fit a cell; larger items go to one overflow node read by every search.
- [`HaloGrid`](src/halo_grid.rs): `SpanGrid` with the per-neighbor copies of `HaloOverflowBoundGrid`, so that its 3 × 3 walk reads a neighbor only if that neighbor's own box may reach the query. Small balls take 225 µs on 156-unit cells, vs. 372 µs in `SpanGrid` (1.29 instead of 4.01 nodes read per query), and 441 vs. 465 µs on 312-unit cells. Aligned small queries are slower, 185 vs. 144 µs, because the spans already prune nearly as well and the copies only add a read. Updates are 2-8% slower.

## Benchmarks

```bash
python benches/run_baseline.py NAME       # everything, in under 30 minutes; then: python benches/summarize_baseline.py NAME
cargo bench --bench search
cargo bench --bench update
bash benches/run_frames.sh frames.jsonl   # then: python benches/summarize_frames.py frames.jsonl
```

[`benches/run_baseline.py`](benches/run_baseline.py) runs all of them on one commit, compiler and pinned core, and continues where it stopped if interrupted. The criterion benches take about 1.7 s per id, and arguments such as `-- --measurement-time 5 --sample-size 100` lengthen them; the noise lies mostly between processes, so another round says more than a longer one.

[`benches/frame.rs`](benches/frame.rs) times whole frames on one evolving state (move and update every ball, end the frame, then search with every ball or enumerate the pairs), one process per structure, mode, scene and seed, after checking the results of selected frames against a reference. `--moving SHARE` moves only that share of the balls; the warmup runs the queries only in its last 5 frames (`--warmup-queries`), and `--frames` sets the timed frames, by default 100, or 40 in the 160k and tall16 scenes.

Some mixed-distribution benchmarks are commented out in `criterion_group!` in [`benches/search.rs`](benches/search.rs); uncomment them to reproduce the tables. Scenes:

- **Mixed distribution:** 10,000 balls of radius 25 to 50, each doubled up to five times with probability 0.25 per step, populating several levels; each ball's bounds serve as a query.
- **Small balls:** as above without doubling; every ball lies well inside a finest cell.
- **Aligned:** two items per finest cell, within 30 units of its center, 60 to 180 units per side; queries of 50 to 300 units (small) and 400 to 1,500 units (large) per side.
- **Gap:** small balls on the finest level and 64 rectangles with sides of 2,600 to 4,900 units on level 3, with the levels in between empty.
- **Adjacent:** small balls and 1,024 rectangles with sides of 320 to 620 units on level 6, directly above them.
- **Before a refresh:** the two-tier scenes after the rectangles have shrunk onto the finest level, queried without a refresh.

The single per-level looseness compared above is not part of the crate; it was measured with `LazyOverflowBoundGrid`, then without cell boxes, modified to apply the largest of its four bounds in all directions.

`SpanOverflowBoundGrid`, `HaloOverflowBoundGrid`, `SpanGrid` and `HaloGrid` were called `StripedOverflowBoundGrid`, `IncomingOverflowBoundGrid`, `StripedGrid` and `IncomingGrid` until 2026-10-21. `ExactOverflowBoundGrid` was called `OverflowBoundGrid` until 2026-10-27, and its benchmark IDs changed with it.

## License

[MIT](LICENSE).
