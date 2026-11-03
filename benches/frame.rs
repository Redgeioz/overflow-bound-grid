//! Frames of moving balls on one evolving state, to compare structures by
//! what a frame of a simulation costs. Every frame moves every ball and
//! updates its bounds, ends the frame (refresh, rebuild or refit), then runs
//! the frame's queries: every ball searching with its own bounds (`search`),
//! or one query for every intersecting pair (`pairs`). Unlike the criterion
//! benches, which time updates and queries on separate copies, the queries
//! here see the state that the frame's updates left, which matters for the
//! structures whose quality drifts between rebuilds.
//!
//! One run is one process for one structure, mode, scene and seed, so that
//! runs share no heap state; `benches/run_frames.sh` runs them all. A run
//! prints one JSON line: the update and query time of each timed frame, the
//! items the structure maintained per frame where it counts them, the heap
//! bytes it holds, and a checksum of the results, which must agree across the
//! structures of a mode, scene and seed.
//!
//! ```text
//! cargo bench --bench frame -- <structure> <search|pairs> <10k|160k>
//!     [--seed N] [--warmup N] [--warmup-queries N] [--frames N] [--verify]
//!     [--verify-every N] [--pool] [--moving SHARE [--rotate]]
//! ```
//!
//! The warmup moves and updates, and runs the queries only in its last
//! `--warmup-queries` frames (20 by default): the queries change no state.
//! `--moving` moves and updates only that share of the balls; with
//! `--rotate`, each frame's share starts where the previous one ended.
//!
//! With `--verify` nothing is timed: the results of selected frames are
//! checked against a reference computed outside the structure, every
//! intersecting pair or the hits of every ball's search, as sorted lists, so
//! that duplicates fail too.

use std::alloc::{GlobalAlloc, Layout, System};
#[cfg(feature = "parry")]
use std::cell::RefCell;
use std::process::exit;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use rand::*;

use lib::ball::*;
use lib::external::*;
use lib::rect::*;
use lib::*;
#[cfg(feature = "parry")]
use parry2d_f64::bounding_volume::Aabb;
#[cfg(feature = "parry")]
use parry2d_f64::math::Vector;
#[cfg(feature = "parry")]
use parry2d_f64::partitioning::{Bvh, BvhLeafUpdateStatus, BvhNode, BvhWorkspace};

#[allow(dead_code, unused_imports, unused_macros, reason = "only the scenes are used here")]
mod common;
use common::*;

/// The system allocator, counting the allocations, the bytes allocated and
/// their peak.
struct CountingAllocator;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    ALLOCS.fetch_add(1, Relaxed);
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
}

/// With `--pool`, freed blocks of at least `POOL_MIN` bytes are kept, up to
/// `POOL_SLOTS`, and handed out again for an allocation of the same size and
/// alignment, as a pooling allocator would. Without it, large blocks go back
/// to the system, whose fresh pages fault on their first touch. Kept blocks
/// do not count as live heap; once the slots are full, freed blocks go back
/// to the system. A pooled `realloc` allocates before it frees, so the peak
/// counts both blocks.
static POOL_ON: AtomicBool = AtomicBool::new(false);
const POOL_MIN: usize = 256 * 1024;
const POOL_SLOTS: usize = 32;
/// (address, size, alignment) of each kept block; a zero address is empty.
static POOL: Mutex<[(usize, usize, usize); POOL_SLOTS]> = Mutex::new([(0, 0, 0); POOL_SLOTS]);

fn pooled(layout: Layout) -> bool {
    layout.size() >= POOL_MIN && POOL_ON.load(Relaxed)
}

/// A kept block of `layout`'s size and alignment, if any.
fn take_pooled(layout: Layout) -> *mut u8 {
    let mut pool = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    for entry in pool.iter_mut() {
        if entry.0 != 0 && entry.1 == layout.size() && entry.2 == layout.align() {
            let ptr = entry.0 as *mut u8;
            *entry = (0, 0, 0);
            return ptr;
        }
    }
    std::ptr::null_mut()
}

/// Keep a freed block; false if the pool is full.
fn keep_pooled(ptr: *mut u8, layout: Layout) -> bool {
    let mut pool = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(entry) = pool.iter_mut().find(|entry| entry.0 == 0) {
        *entry = (ptr as usize, layout.size(), layout.align());
        return true;
    }
    false
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut ptr = if pooled(layout) { take_pooled(layout) } else { std::ptr::null_mut() };
        if ptr.is_null() {
            ptr = unsafe { System.alloc(layout) };
        }
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if pooled(layout) {
            let ptr = take_pooled(layout);
            if !ptr.is_null() {
                unsafe { ptr.write_bytes(0, layout.size()) };
                grew(layout.size());
                return ptr;
            }
        }
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !(pooled(layout) && keep_pooled(ptr, layout)) {
            unsafe { System.dealloc(ptr, layout) };
        }
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees a valid layout for `new_size`.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        if pooled(layout) || pooled(new_layout) {
            let new = unsafe { self.alloc(new_layout) };
            if !new.is_null() {
                unsafe { std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size)) };
                unsafe { self.dealloc(ptr, layout) };
            }
            return new;
        }
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            grew(new_size);
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// The process's page faults so far, soft and hard, where the platform
/// reports them.
#[cfg(windows)]
fn page_faults() -> Option<u64> {
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }
    let mut counters = ProcessMemoryCounters::default();
    let size = size_of::<ProcessMemoryCounters>() as u32;
    counters.cb = size;
    // SAFETY: the counters are a properly sized, writable struct.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) };
    (ok != 0).then_some(counters.page_fault_count as u64)
}

#[cfg(not(windows))]
fn page_faults() -> Option<u64> {
    None
}

/// A structure as one frame uses it.
trait Structure {
    type Handle: Copy;
    fn insert(&mut self, bounds: Rectangle, id: usize) -> Self::Handle;
    fn update(&mut self, bounds: Rectangle, handle: Self::Handle);
    fn end_frame(&mut self);
    fn search(&self, bounds: &Rectangle, callback: impl FnMut(usize));
    fn pairs(&self, callback: impl FnMut(usize, usize));
    /// Items that the last `end_frame` maintained, where the structure counts
    /// them.
    fn maintained(&self) -> Option<usize> {
        None
    }
}

/// Implements [`Structure`] for `$tree`, whose `end_frame` calls `$end_frame`.
macro_rules! structure {
    ($tree:ty $(, $end_frame:ident)?) => {
        impl Structure for $tree {
            type Handle = ItemHandle;
            fn insert(&mut self, bounds: Rectangle, id: usize) -> ItemHandle {
                <$tree>::insert(self, bounds, id)
            }
            fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
                <$tree>::update(self, bounds, handle);
            }
            fn end_frame(&mut self) {
                $(self.$end_frame();)?
            }
            #[inline]
            fn search(&self, bounds: &Rectangle, callback: impl FnMut(usize)) {
                <$tree>::search(self, bounds, callback);
            }
            #[inline]
            fn pairs(&self, callback: impl FnMut(usize, usize)) {
                self.for_each_pair(callback);
            }
        }
    };
}

structure!(HaloOverflowBoundGrid<usize, MAX_LEVEL>, refresh_bounds);
structure!(HaloOverflowBoundGrid<usize, LARGE_MAX_LEVEL>, refresh_bounds);
structure!(SpanOverflowBoundGrid<usize, MAX_LEVEL>, refresh_bounds);
structure!(SpanOverflowBoundGrid<usize, LARGE_MAX_LEVEL>, refresh_bounds);
structure!(HierarchicalLooseGrid<usize, MAX_LEVEL>);
structure!(HierarchicalLooseGrid<usize, LARGE_MAX_LEVEL>);
structure!(UniformGrid<usize>);
structure!(SpanGrid<usize>, refresh_bounds);
structure!(HaloGrid<usize>, refresh_bounds);
structure!(TwoLayerGrid<usize>);
structure!(SortAndSweep<usize>, refresh_order);
structure!(SpatialJoinSweep<usize>, refresh_order);
#[cfg(feature = "packed")]
structure!(PackedRTree<usize>, refresh_tree);

/// `SortAndSweep` with its scalar pair kernel, which tests one item at a time.
struct ScalarSweep(SortAndSweep<usize>);

impl Structure for ScalarSweep {
    type Handle = ItemHandle;
    fn insert(&mut self, bounds: Rectangle, id: usize) -> ItemHandle {
        self.0.insert(bounds, id)
    }
    fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.0.update(bounds, handle);
    }
    fn end_frame(&mut self) {
        self.0.refresh_order();
    }
    #[inline]
    fn search(&self, bounds: &Rectangle, callback: impl FnMut(usize)) {
        self.0.search(bounds, callback);
    }
    #[inline]
    fn pairs(&self, callback: impl FnMut(usize, usize)) {
        self.0.for_each_pair_scalar(callback);
    }
}

#[cfg(feature = "parry")]
impl Structure for ParryBvh<usize> {
    type Handle = ItemHandle;
    fn insert(&mut self, bounds: Rectangle, id: usize) -> ItemHandle {
        ParryBvh::insert(self, bounds, id)
    }
    fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        ParryBvh::update(self, bounds, handle);
    }
    fn end_frame(&mut self) {
        self.refresh_tree();
    }
    #[inline]
    fn search(&self, bounds: &Rectangle, callback: impl FnMut(usize)) {
        ParryBvh::search(self, bounds, callback);
    }
    #[inline]
    fn pairs(&self, callback: impl FnMut(usize, usize)) {
        self.for_each_pair(callback);
    }
    fn maintained(&self) -> Option<usize> {
        Some(self.last_maintained())
    }
}

/// parry's tree driven directly, as an engine's broad phase drives it: a
/// control for the costs of the adapter `ParryBvh`, not a structure of the
/// crate. The item is its leaf index; an update whose bounds leave the leaf
/// writes the leaf at once (or reinserts it), and `end_frame` refits,
/// optimizes or rebuilds. Searches read compact bounds, and with no margin
/// test the nodes strictly, so that the leaves found need no filter. Results
/// are exact only after `end_frame`, which the crate's structures do not
/// require.
#[cfg(feature = "parry")]
struct ParryRaw {
    tree: Bvh,
    workspace: BvhWorkspace,
    /// The traversal stack of the pair queries, kept between them.
    pair_workspace: RefCell<BvhWorkspace>,
    maintenance: Maintenance,
    margin: f64,
    bounds: Vec<Rectangle>,
    leaves: Vec<Rectangle>,
    maintained: usize,
    last_maintained: usize,
}

#[cfg(feature = "parry")]
fn to_aabb(bounds: &Rectangle) -> Aabb {
    Aabb::new(
        Vector::new(bounds.min_x, bounds.min_y),
        Vector::new(bounds.max_x, bounds.max_y),
    )
}

#[cfg(feature = "parry")]
fn grown(bounds: &Rectangle, margin: f64) -> Rectangle {
    Rectangle::new(
        bounds.min_x - margin,
        bounds.min_y - margin,
        bounds.max_x + margin,
        bounds.max_y + margin,
    )
}

#[cfg(feature = "parry")]
impl ParryRaw {
    fn new((maintenance, margin): (Maintenance, f64)) -> Self {
        Self {
            tree: Bvh::new(),
            workspace: BvhWorkspace::default(),
            pair_workspace: RefCell::new(BvhWorkspace::default()),
            maintenance,
            margin,
            bounds: Vec::new(),
            leaves: Vec::new(),
            maintained: 0,
            last_maintained: 0,
        }
    }
}

#[cfg(feature = "parry")]
impl Structure for ParryRaw {
    type Handle = usize;
    fn insert(&mut self, bounds: Rectangle, id: usize) -> usize {
        assert_eq!(id, self.bounds.len(), "Items are numbered in order of insertion.");
        let leaf = grown(&bounds, self.margin);
        self.tree.insert(to_aabb(&leaf), id as u32);
        self.bounds.push(bounds);
        self.leaves.push(leaf);
        self.maintained += 1;
        id
    }
    fn update(&mut self, bounds: Rectangle, id: usize) {
        let leaf = &self.leaves[id];
        let inside = leaf.min_x <= bounds.min_x
            && leaf.min_y <= bounds.min_y
            && bounds.max_x <= leaf.max_x
            && bounds.max_y <= leaf.max_y;
        if !inside {
            let leaf = grown(&bounds, self.margin);
            let aabb = to_aabb(&leaf);
            if self.maintenance == Maintenance::Reinsert {
                self.tree.remove(id as u32);
                self.tree.insert(aabb, id as u32);
            } else {
                self.tree.insert_or_update_partially(aabb, id as u32, 0.0);
            }
            self.leaves[id] = leaf;
            self.maintained += 1;
        }
        self.bounds[id] = bounds;
    }
    fn end_frame(&mut self) {
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
        self.last_maintained = self.maintained;
        self.maintained = 0;
    }
    #[inline]
    fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(usize)) {
        if self.margin == 0.0 {
            // The leaves hold the exact bounds, so a strict test of every
            // node finds exactly the intersecting items.
            let strict = |node: &BvhNode| {
                let (min, max) = (node.mins(), node.maxs());
                min.x < bounds.max_x && bounds.min_x < max.x && min.y < bounds.max_y && bounds.min_y < max.y
            };
            for leaf in self.tree.leaves(strict) {
                callback(leaf as usize);
            }
        } else {
            for leaf in self.tree.intersect_aabb(&to_aabb(bounds)) {
                if self.bounds[leaf as usize].intersects(bounds) {
                    callback(leaf as usize);
                }
            }
        }
    }
    #[inline]
    fn pairs(&self, mut callback: impl FnMut(usize, usize)) {
        let mut workspace = self.pair_workspace.borrow_mut();
        self.tree.traverse_bvtt_single_tree::<false>(&mut workspace, &mut |a, b| {
            let (a, b) = (a as usize, b as usize);
            if self.bounds[a].intersects(&self.bounds[b]) {
                callback(a, b);
            }
        });
    }
    fn maintained(&self) -> Option<usize> {
        Some(self.last_maintained)
    }
}

/// The leaf margins of [`ParryRapier`]: a fixed skin, or Rapier's adaptive
/// one, `clamp(min_extent / 8, 0.04, 0.25)` length units, with a length unit
/// of 100 for the balls of radius 25 to 1600.
#[cfg(feature = "parry")]
#[derive(Clone, Copy)]
enum Skin {
    Fixed(f64),
    Adaptive,
}

#[cfg(feature = "parry")]
impl Skin {
    fn of(self, bounds: &Rectangle) -> f64 {
        match self {
            Skin::Fixed(skin) => skin,
            Skin::Adaptive => (bounds.get_width().min(bounds.get_height()) * 0.125).clamp(4.0, 25.0),
        }
    }
}

/// parry's tree under the schedule of Rapier's broad phase
/// (`broad_phase_bvh/update.rs` and `mod.rs`, single-threaded), as a
/// control. Every moved item's leaf goes through
/// `update_partially_if_present` with its skin, which writes the leaf grown
/// by the skin only if the bounds left it. Rapier reinserts leaves instead
/// only when fewer than 1/16 of its colliders moved at all, which never
/// happens here, where every ball moves. The tree is then refitted, fully
/// unless this and the last frame changed fewer than 1/16 of the leaves
/// together, in which case only their paths are. The incremental optimizer
/// runs after the frame's queries, followed by a refit that keeps the change
/// flags: in every frame that changed at least 1/16 of the leaves, and every
/// 8th frame once the changed leaves since its last run reach 1/64. Rapier
/// then traverses only the pairs of changed leaves to update persistent pairs;
/// here every frame enumerates all pairs, as for the other structures.
#[cfg(feature = "parry")]
struct ParryRapier {
    raw: ParryRaw,
    skin: Skin,
    /// Leaves written this frame and the last.
    changed: Vec<u32>,
    last_changed: Vec<u32>,
    /// Leaf writes since the optimizer last ran.
    debt: usize,
    frame: usize,
    /// Whether the optimizer is due before the next frame's updates.
    optimize: bool,
}

#[cfg(feature = "parry")]
impl ParryRapier {
    fn new(skin: Skin) -> Self {
        // A margin other than zero makes the searches filter the leaves found.
        let mut raw = ParryRaw::new((Maintenance::OptimizeRefit, 1.0));
        raw.margin = f64::NAN;
        Self {
            raw,
            skin,
            changed: Vec::new(),
            last_changed: Vec::new(),
            debt: 0,
            frame: 0,
            optimize: false,
        }
    }
}

#[cfg(feature = "parry")]
impl Structure for ParryRapier {
    type Handle = usize;
    fn insert(&mut self, bounds: Rectangle, id: usize) -> usize {
        // Rapier inserts the exact box; the skin comes with the first write.
        assert_eq!(id, self.raw.bounds.len(), "Items are numbered in order of insertion.");
        self.raw.tree.insert(to_aabb(&bounds), id as u32);
        self.raw.bounds.push(bounds);
        self.changed.push(id as u32);
        id
    }
    fn update(&mut self, bounds: Rectangle, id: usize) {
        let raw = &mut self.raw;
        if self.optimize {
            raw.tree.optimize_incremental(&mut raw.workspace);
            raw.tree.refit_without_resolve(&mut raw.workspace);
            self.optimize = false;
        }
        let status = raw
            .tree
            .update_partially_if_present(to_aabb(&bounds), id as u32, self.skin.of(&bounds));
        if status == Some(BvhLeafUpdateStatus::UpdatedInPlace) {
            self.changed.push(id as u32);
            self.debt += 1;
        }
        raw.bounds[id] = bounds;
    }
    fn end_frame(&mut self) {
        let raw = &mut self.raw;
        let leaves = raw.bounds.len();
        let changed = self.changed.len();
        let optimize = self.debt > 0
            && (changed * 16 >= leaves || (self.frame % 8 == 0 && self.debt * 64 >= leaves));
        if optimize {
            self.debt = 0;
            self.optimize = true;
        }
        if self.frame == 0 || (changed + self.last_changed.len()) * 16 >= leaves {
            raw.tree.refit(&mut raw.workspace);
        } else {
            raw.tree.refit_partial(&self.last_changed, &self.changed);
        }
        std::mem::swap(&mut self.last_changed, &mut self.changed);
        self.changed.clear();
        raw.last_maintained = changed;
        self.frame += 1;
    }
    #[inline]
    fn search(&self, bounds: &Rectangle, callback: impl FnMut(usize)) {
        self.raw.search(bounds, callback);
    }
    #[inline]
    fn pairs(&self, callback: impl FnMut(usize, usize)) {
        self.raw.pairs(callback);
    }
    fn maintained(&self) -> Option<usize> {
        Some(self.raw.last_maintained)
    }
}

/// No structure: what moving the balls and calling a structure cost by
/// themselves, to subtract from the update times. Its queries find nothing,
/// so its checksums differ from the others'.
struct Null;

impl Structure for Null {
    type Handle = ();
    fn insert(&mut self, _bounds: Rectangle, _id: usize) {}
    fn update(&mut self, bounds: Rectangle, _handle: ()) {
        std::hint::black_box(bounds);
    }
    fn end_frame(&mut self) {}
    fn search(&self, bounds: &Rectangle, _callback: impl FnMut(usize)) {
        std::hint::black_box(bounds);
    }
    fn pairs(&self, _callback: impl FnMut(usize, usize)) {}
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Search,
    Pairs,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Search => "search",
            Mode::Pairs => "pairs",
        }
    }
}

/// The scenes of the benches: 10,000 balls in `BOUNDS`, or sixteen times as
/// many at the same density in `LARGE_BOUNDS`; or, at the same density and
/// width as `BOUNDS`, worlds 4 and 16 times as tall (`tall4`, `tall16`), whose
/// columns hold more items, for the sweeps along x.
struct Scene {
    name: &'static str,
    world: Rectangle,
    count: usize,
    large: bool,
}

struct Options {
    seed: u64,
    warmup: usize,
    frames: usize,
    verify: bool,
    /// Frames between checks of `--verify`, after the first three.
    verify_every: usize,
    /// Warmup frames at the end of the warmup that run the queries; the
    /// others only move and update, which is all that changes the state.
    warmup_queries: usize,
    /// Share of the balls that move and are updated each frame, the first
    /// ones by id, which lie at random; the others stay where they are and
    /// are never updated.
    moving: f64,
    /// Whether each frame moves the share of balls after the previous
    /// frame's, wrapping around, rather than always the first ones.
    rotate: bool,
}

/// The moving balls of `benches/update.rs`, drawn in the same order: radii
/// doubled up to five times, each with a chance of a quarter.
fn moving_balls(scene: &Scene, seed: u64) -> Vec<Ball> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..scene.count)
        .map(|_| {
            let (x, y) = scene.world.get_random_point(&mut rng);
            let mut r = rng.gen_range(MIN_RADIUS..MAX_RADIUS);
            for _ in 0..5 {
                if rng.gen_bool(0.25) {
                    r *= 2.0;
                } else {
                    break;
                }
            }
            let v = rng.gen_range(MIN_VELOCITY..MAX_VELOCITY);
            let angle = rng.gen_range(0.0..std::f64::consts::TAU);
            BallBuilder::new()
                .set_position(x, y)
                .set_radius(r)
                .set_velocity(v * angle.cos(), v * angle.sin())
                .build()
        })
        .collect()
}

/// Move `moving` balls one step, the first ones or, if `first` is not zero,
/// those from `first` on, wrapping around; update their bounds and end the
/// frame.
fn step<S: Structure>(
    tree: &mut S,
    balls: &mut [Ball],
    handles: &[S::Handle],
    world: &Rectangle,
    moving: usize,
    first: usize,
) {
    if first != 0 {
        for i in 0..moving {
            let id = (first + i) % balls.len();
            balls[id].update(world);
            tree.update(balls[id].get_bounds(), handles[id]);
        }
        tree.end_frame();
        return;
    }
    for (ball, &handle) in balls[..moving].iter_mut().zip(handles) {
        ball.update(world);
        tree.update(ball.get_bounds(), handle);
    }
    tree.end_frame();
}

/// SplitMix64's finalizer, so that the sum of the results' hashes changes with
/// any result that changes.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Run the frame's queries; returns the number of results and the sum of a
/// hash of every (query, item) or (smaller id, larger id), which every exact
/// structure reproduces in any order.
fn query(tree: &impl Structure, mode: Mode, balls: &[Ball]) -> (u64, u64) {
    let (mut count, mut sum) = (0u64, 0u64);
    match mode {
        Mode::Search => {
            for (query, ball) in balls.iter().enumerate() {
                let query = (query as u64) << 32;
                tree.search(&ball.get_bounds(), |id| {
                    count += 1;
                    sum = sum.wrapping_add(mix(query | id as u64));
                });
            }
        }
        Mode::Pairs => tree.pairs(|a, b| {
            count += 1;
            sum = sum.wrapping_add(mix(((a.min(b) as u64) << 32) | a.max(b) as u64));
        }),
    }
    (count, sum)
}

/// Every intersecting pair of distinct balls, by a forward scan in order of
/// min x, smaller id first, sorted.
fn reference_pairs(bounds: &[Rectangle]) -> Vec<(usize, usize)> {
    let mut order: Vec<usize> = (0..bounds.len()).collect();
    order.sort_unstable_by(|&a, &b| bounds[a].min_x.total_cmp(&bounds[b].min_x));
    let mut pairs = Vec::new();
    for (position, &a) in order.iter().enumerate() {
        for &b in &order[position + 1..] {
            if bounds[b].min_x >= bounds[a].max_x {
                break;
            }
            if bounds[a].intersects(&bounds[b]) {
                pairs.push((a.min(b), a.max(b)));
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

/// Check the results of the current frame against the references: every
/// pair, or every ball's search. A search is with the ball's own bounds, so
/// its hits are the ball and the partners of its reference pairs.
fn verify(tree: &impl Structure, mode: Mode, balls: &[Ball]) -> Result<(), String> {
    let bounds: Vec<Rectangle> = balls.iter().map(Ball::get_bounds).collect();
    match mode {
        Mode::Pairs => {
            let mut reported = Vec::new();
            tree.pairs(|a, b| reported.push((a.min(b), a.max(b))));
            reported.sort_unstable();
            let expected = reference_pairs(&bounds);
            if reported != expected {
                return Err(format!(
                    "{} pairs reported, {} expected",
                    reported.len(),
                    expected.len()
                ));
            }
        }
        Mode::Search => {
            let mut expected: Vec<Vec<usize>> = bounds
                .iter()
                .enumerate()
                .map(|(id, own)| if own.intersects(own) { vec![id] } else { Vec::new() })
                .collect();
            for (a, b) in reference_pairs(&bounds) {
                expected[a].push(b);
                expected[b].push(a);
            }
            let mut found = Vec::new();
            for (query, expected) in bounds.iter().zip(&mut expected) {
                found.clear();
                tree.search(query, |id| found.push(id));
                found.sort_unstable();
                expected.sort_unstable();
                if found != *expected {
                    return Err(format!("search {query:?}: {found:?}, expected {expected:?}"));
                }
            }
        }
    }
    Ok(())
}

/// Median, mean and 10th and 90th percentiles of `values`.
fn summary(values: &[f64]) -> String {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    format!(
        "{{\"median\":{:.3},\"mean\":{:.3},\"p10\":{:.3},\"p90\":{:.3}}}",
        at(0.5),
        mean,
        at(0.1),
        at(0.9)
    )
}

fn run<S: Structure>(name: &str, mode: Mode, scene: &Scene, options: &Options, create: impl FnOnce() -> S) {
    let mut balls = moving_balls(scene, options.seed);
    let mut handles: Vec<S::Handle> = Vec::with_capacity(balls.len());
    let before = LIVE.load(Relaxed);
    let mut tree = create();
    for (id, ball) in balls.iter().enumerate() {
        handles.push(tree.insert(ball.get_bounds(), id));
    }
    tree.end_frame();
    let setup_bytes = LIVE.load(Relaxed) - before;
    let total = options.warmup + options.frames;
    let moving = (balls.len() as f64 * options.moving).round() as usize;
    let count = balls.len().max(1);
    let mut first = 0;
    let advance = |first: &mut usize| {
        if options.rotate {
            *first = (*first + moving) % count;
        }
    };

    if options.verify {
        let mut checked = 0;
        for frame in 0..total {
            step(&mut tree, &mut balls, &handles, &scene.world, moving, first);
            advance(&mut first);
            if frame < 3 || frame % options.verify_every == 0 || frame + 1 == total {
                if let Err(message) = verify(&tree, mode, &balls) {
                    eprintln!("{name} {} {} seed {}: frame {frame}: {message}", mode.name(), scene.name, options.seed);
                    exit(1);
                }
                checked += 1;
            }
        }
        println!(
            "{{\"verify\":true,\"structure\":\"{name}\",\"mode\":\"{}\",\"scene\":\"{}\",\"seed\":{},\"frames_checked\":{checked}}}",
            mode.name(),
            scene.name,
            options.seed
        );
        return;
    }

    let mut update_us = Vec::with_capacity(options.frames);
    let mut query_us = Vec::with_capacity(options.frames);
    let mut frame_us = Vec::with_capacity(options.frames);
    let mut maintained = Vec::with_capacity(options.frames);
    let (mut count, mut sum) = (0u64, 0u64);
    PEAK.store(LIVE.load(Relaxed), Relaxed);
    let mut allocs = 0;
    let mut faults = None;
    for frame in 0..total {
        if frame == options.warmup {
            allocs = ALLOCS.load(Relaxed);
            faults = page_faults();
        }
        let start = Instant::now();
        step(&mut tree, &mut balls, &handles, &scene.world, moving, first);
        let updated = Instant::now();
        advance(&mut first);
        if frame.saturating_add(options.warmup_queries) < options.warmup {
            continue;
        }
        let (frame_count, frame_sum) = query(&tree, mode, &balls);
        let done = Instant::now();
        if frame >= options.warmup {
            update_us.push((updated - start).as_secs_f64() * 1e6);
            query_us.push((done - updated).as_secs_f64() * 1e6);
            frame_us.push((done - start).as_secs_f64() * 1e6);
            if let Some(items) = tree.maintained() {
                maintained.push(items as f64 / balls.len() as f64);
            }
            count += frame_count;
            sum = sum.wrapping_add(frame_sum);
        }
    }
    let allocs_per_frame = (ALLOCS.load(Relaxed) - allocs) as f64 / options.frames as f64;
    let faults_per_frame = match (faults, page_faults()) {
        (Some(start), Some(end)) => format!("{:.1}", (end - start) as f64 / options.frames as f64),
        _ => "null".to_string(),
    };
    let end_bytes = LIVE.load(Relaxed) - before;
    let peak_bytes = PEAK.load(Relaxed) - before;
    let maintained = if maintained.is_empty() {
        "null".to_string()
    } else {
        summary(&maintained)
    };
    println!(
        "{{\"structure\":\"{name}\",\"mode\":\"{}\",\"scene\":\"{}\",\"seed\":{},\"warmup\":{},\"warmup_queries\":{},\"frames\":{},\
         \"update_us\":{},\"query_us\":{},\"frame_us\":{},\"maintained\":{maintained},\
         \"setup_bytes\":{setup_bytes},\"end_bytes\":{end_bytes},\"peak_bytes\":{peak_bytes},\
         \"allocs_per_frame\":{allocs_per_frame:.2},\"faults_per_frame\":{faults_per_frame},\"pool\":{},\"moving\":{},\"rotate\":{},\
         \"results\":{count},\"checksum\":\"{count}:{sum}\"}}",
        mode.name(),
        scene.name,
        options.seed,
        options.warmup,
        options.warmup_queries.min(options.warmup),
        options.frames,
        summary(&update_us),
        summary(&query_us),
        summary(&frame_us),
        POOL_ON.load(Relaxed),
        options.moving,
        options.rotate,
    );
}

/// `<maintenance>[-m<margin>]` after `ParryBvh/` or `ParryRaw/`:
/// `rebuild-binned`, `rebuild-ploc`, `refit`, `refit-opt` or `reinsert`, with
/// leaves grown by the margin; a rebuilt tree takes none.
#[cfg(feature = "parry")]
fn parry_config(variant: &str) -> Option<(Maintenance, f64)> {
    let (maintenance, margin) = match variant.rsplit_once("-m") {
        Some((maintenance, margin)) => (maintenance, margin.parse().ok()?),
        None => (variant, 0.0),
    };
    let maintenance = match maintenance {
        "rebuild-binned" => Maintenance::Rebuild(BvhBuildStrategy::Binned),
        "rebuild-ploc" => Maintenance::Rebuild(BvhBuildStrategy::Ploc),
        "refit" => Maintenance::Refit,
        "refit-opt" => Maintenance::RefitOptimize,
        "opt-refit" => Maintenance::OptimizeRefit,
        "reinsert" => Maintenance::Reinsert,
        _ => return None,
    };
    if !(margin >= 0.0) || (margin > 0.0 && matches!(maintenance, Maintenance::Rebuild(_))) {
        return None;
    }
    Some((maintenance, margin))
}

fn usage() -> ! {
    eprintln!(
        "usage: frame <structure> <search|pairs> <10k|160k|tall4|tall16> [--seed N] [--warmup N] [--warmup-queries N] [--frames N]\n\
         [--verify [--verify-every N]] [--pool] [--moving SHARE [--rotate]]\n\
         structures: HaloOverflowBoundGrid, SpanOverflowBoundGrid, HierarchicalLooseGrid, UniformGrid,\n\
         TwoLayerGrid, SortAndSweep[/scalar], SpatialJoinSweep/<<sorted|kept>-<btree|vec>|kept-cells<height>>,\n\
         PackedRTree/<static-hilbert[-join]|geo-hilbert[-walk]|geo-str>[-coherent][-defer<percent>][-n<node size>], Null,\n\
         ParryBvh/<variant>, ParryRaw/<variant>,\n\
         ParryRapier/<m<margin>|adaptive>; variants: <rebuild-binned|rebuild-ploc|refit|refit-opt|opt-refit|reinsert>[-m<margin>]"
    );
    exit(2);
}

fn main() {
    // `cargo bench` adds `--bench`.
    let args: Vec<String> = std::env::args().skip(1).filter(|arg| arg != "--bench").collect();
    if args.len() < 3 {
        usage();
    }
    let name = args[0].as_str();
    let mode = match args[1].as_str() {
        "search" => Mode::Search,
        "pairs" => Mode::Pairs,
        _ => usage(),
    };
    let tall = |name, times: usize| Scene {
        name,
        world: Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT * times as f64),
        count: AMOUNT * times,
        large: false,
    };
    let scene = match args[2].as_str() {
        "10k" => Scene { name: "10k", world: BOUNDS, count: AMOUNT, large: false },
        "160k" => Scene { name: "160k", world: LARGE_BOUNDS, count: LARGE_AMOUNT, large: true },
        "tall4" => tall("tall4", 4),
        "tall16" => tall("tall16", 16),
        _ => usage(),
    };
    let (warmup, frames) = match scene.name {
        "160k" => (300, 200),
        "tall16" => (100, 100),
        _ => (300, 300),
    };
    let mut options =
        Options { seed: SEED, warmup, frames, verify: false, verify_every: 25, warmup_queries: 20, moving: 1.0, rotate: false };
    let mut rest = args[3..].iter();
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().and_then(|value| value.parse().ok()).unwrap_or_else(|| usage());
        match flag.as_str() {
            "--seed" => options.seed = value() as u64,
            "--warmup" => options.warmup = value(),
            "--warmup-queries" => options.warmup_queries = value(),
            "--frames" => options.frames = value(),
            "--verify" => options.verify = true,
            "--pool" => POOL_ON.store(true, Relaxed),
            "--rotate" => options.rotate = true,
            "--moving" => {
                let share: f64 = rest.next().and_then(|value| value.parse().ok()).unwrap_or_else(|| usage());
                if !(0.0..=1.0).contains(&share) {
                    usage();
                }
                options.moving = share;
            }
            "--verify-every" => options.verify_every = value().max(1),
            _ => usage(),
        }
    }

    let world = scene.world.clone();
    macro_rules! levels {
        ($tree:ident) => {
            if scene.name.starts_with("tall") {
                // Their square roots would not fit these worlds.
                usage()
            } else if scene.large {
                run(name, mode, &scene, &options, || $tree::<usize, LARGE_MAX_LEVEL>::new(world))
            } else {
                run(name, mode, &scene, &options, || $tree::<usize, MAX_LEVEL>::new(world))
            }
        };
    }
    match name {
        "HaloOverflowBoundGrid" => levels!(HaloOverflowBoundGrid),
        "SpanOverflowBoundGrid" => levels!(SpanOverflowBoundGrid),
        "HierarchicalLooseGrid" => levels!(HierarchicalLooseGrid),
        "UniformGrid" => run(name, mode, &scene, &options, || UniformGrid::new(world, GRID_CELL_SIZE)),
        "SpanGrid" => run(name, mode, &scene, &options, || SpanGrid::new(world, GRID_CELL_SIZE)),
        "HaloGrid" => run(name, mode, &scene, &options, || HaloGrid::new(world, GRID_CELL_SIZE)),
        "TwoLayerGrid" => run(name, mode, &scene, &options, || TwoLayerGrid::new(world, GRID_CELL_SIZE)),
        "SortAndSweep" => run(name, mode, &scene, &options, SortAndSweep::new),
        "SortAndSweep/scalar" => run(name, mode, &scene, &options, || ScalarSweep(SortAndSweep::new())),
        _ if name.starts_with("SpatialJoinSweep/") => {
            let (order, active) = match &name["SpatialJoinSweep/".len()..] {
                "sorted-btree" => (EventOrder::Sorted, ActiveSet::BTree),
                "sorted-vec" => (EventOrder::Sorted, ActiveSet::SortedVec),
                "kept-btree" => (EventOrder::Kept, ActiveSet::BTree),
                "kept-vec" => (EventOrder::Kept, ActiveSet::SortedVec),
                variant => match variant.strip_prefix("kept-cells").and_then(|h| h.parse::<f64>().ok()) {
                    Some(height) if height > 0.0 => (EventOrder::Kept, ActiveSet::Cells(height)),
                    _ => usage(),
                },
            };
            run(name, mode, &scene, &options, || SpatialJoinSweep::new(order, active))
        }
        #[cfg(feature = "parry")]
        _ if name.starts_with("ParryBvh/") => {
            let (maintenance, margin) = parry_config(&name["ParryBvh/".len()..]).unwrap_or_else(|| usage());
            run(name, mode, &scene, &options, || ParryBvh::new(maintenance, margin))
        }
        "Null" => run(name, mode, &scene, &options, || Null),
        #[cfg(feature = "packed")]
        _ if name.starts_with("PackedRTree/") => {
            let variant = &name["PackedRTree/".len()..];
            let (variant, node_size) = match variant.rsplit_once("-n") {
                Some((variant, size)) => match size.parse() {
                    Ok(size) if (2..=u16::MAX as usize).contains(&size) => (variant, size),
                    _ => usage(),
                },
                None => (variant, 16),
            };
            let (variant, defer) = match variant.rsplit_once("-defer") {
                Some((variant, percent)) => match percent.parse::<f64>() {
                    Ok(percent) if percent >= 0.0 && percent.is_finite() => (variant, percent / 100.0),
                    _ => usage(),
                },
                None => (variant, 0.0),
            };
            let (variant, coherent) = match variant.strip_suffix("-coherent") {
                Some(variant) => (variant, true),
                None => (variant, false),
            };
            let packing = match variant {
                "static-hilbert" => Packing::StaticHilbert,
                "static-hilbert-join" => Packing::StaticHilbertJoin,
                "geo-hilbert" => Packing::GeoHilbert,
                "geo-str" => Packing::GeoStr,
                "geo-hilbert-walk" => Packing::GeoHilbertWalk,
                _ => usage(),
            };
            run(name, mode, &scene, &options, || {
                let tree = PackedRTree::with_node_size(packing, node_size).deferred(defer);
                if coherent { tree.coherent() } else { tree }
            })
        }
        #[cfg(feature = "parry")]
        _ if name.starts_with("ParryRapier/") => {
            let skin = match &name["ParryRapier/".len()..] {
                "adaptive" => Skin::Adaptive,
                variant => match variant.strip_prefix('m').and_then(|m| m.parse::<f64>().ok()) {
                    // Change detection needs a skin.
                    Some(skin) if skin > 0.0 => Skin::Fixed(skin),
                    _ => usage(),
                },
            };
            run(name, mode, &scene, &options, || ParryRapier::new(skin))
        }
        #[cfg(feature = "parry")]
        _ if name.starts_with("ParryRaw/") => {
            let config = parry_config(&name["ParryRaw/".len()..]).unwrap_or_else(|| usage());
            run(name, mode, &scene, &options, || ParryRaw::new(config))
        }
        _ => usage(),
    }
}
