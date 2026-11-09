//! The candidate phase of the sweep of Freiburg's spatialjoin (Brosi, Bast
//! et al., SIGSPATIAL 2025, §3.1), ported from its disk-based event file to
//! memory: every item's left and right x are events sorted by x, entering
//! events before leaving ones at equal x. A sweep over the events keeps the
//! items whose x range holds the sweep line in an active set indexed by their
//! y intervals; an item that leaves is erased from it, then paired with every
//! active item whose y interval overlaps its own. Each pair is found once, at
//! the first of its two leaving events. As spatialjoin's event records carry
//! the y interval and its active set the payload, the events here carry the
//! item's bounds and the active set its x range, so that a sweep reads the
//! events in order and nothing else but the items of the pairs it reports.
//!
//! The y index follows `util::geo::IntervalIdx` of ad-freiburg/util: the
//! intervals go by span into buckets of spans below 10, 100, ..., 10^8 and an
//! overflow bucket for the rest, each sorted by left end. A search of a
//! bucket starts at the query's lower end minus the bucket's span limit, the
//! overflow bucket's being the largest span inserted into it so far, and
//! reads on while the left ends do not pass the query's upper end.
//!
//! Like spatialjoin, the candidates include touching boxes; they are tested
//! again as [`Rectangle::intersects`] does. Items with a NaN corner meet
//! nothing and stay out of the sweep. The search is a linear scan, to fill
//! the interface; this is not a spatial index.

use crate::handle::{ItemHandle, Slots};
use crate::rect::Rectangle;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::sync::Mutex;

/// When the events get sorted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventOrder {
    /// Afresh for every pair query, as spatialjoin sorts its event file once
    /// per join.
    Sorted,
    /// Kept between frames: [`refresh_order`](SpatialJoinSweep::refresh_order)
    /// restores it by insertion sort, which is cheap while items move little,
    /// as `SortAndSweep` keeps its order.
    Kept,
}

/// The y index of the active set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveSet {
    /// `IntervalIdx` with ordered sets as buckets, as spatialjoin's
    /// `std::set`.
    BTree,
    /// `IntervalIdx` with vectors kept sorted as buckets, inserted into and
    /// erased from by binary search.
    SortedVec,
    /// Not in spatialjoin: a control for what pruning in y gains with cheap
    /// updates. Cells of about the given height by lower end, unsorted, so
    /// that insertion and erasure take constant time; a search reads the
    /// cells from its lower end minus the largest span inserted so far up to
    /// its upper end.
    Cells(f64),
}

/// Shifts per event that the insertion sort of a refresh makes at most before
/// it falls back to a full sort.
const SHIFT_BUDGET: usize = 8;

/// Buckets below the overflow: spans below 10, 100, ..., 10^8, as
/// `IntervalIdx` has for 32-bit coordinates.
const BUCKETS: usize = 8;

/// An item's entering or leaving event, with the item's bounds.
#[derive(Clone, Copy)]
struct Event {
    /// `min_x` of an entering event, `max_x` of a leaving one.
    x: f64,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    slot: u32,
    leaving: bool,
}

impl Event {
    /// By x, then entering before leaving, then slot.
    fn cmp(&self, other: &Self) -> Ordering {
        self.x
            .total_cmp(&other.x)
            .then(self.leaving.cmp(&other.leaving))
            .then(self.slot.cmp(&other.slot))
    }

    fn set_bounds(&mut self, bounds: &Rectangle) {
        self.x = if self.leaving {
            bounds.max_x
        } else {
            bounds.min_x
        };
        self.min_x = bounds.min_x;
        self.min_y = bounds.min_y;
        self.max_x = bounds.max_x;
        self.max_y = bounds.max_y;
    }

    fn has_nan(&self) -> bool {
        self.min_x.is_nan() | self.min_y.is_nan() | self.max_x.is_nan() | self.max_y.is_nan()
    }

    fn bounds(&self) -> Rectangle {
        Rectangle::new(self.min_x, self.min_y, self.max_x, self.max_y)
    }
}

/// A y interval of the active set, with the item's x range for the exact
/// test, ordered by left end, then right end, then slot, as `IntervalIdx`
/// orders its sets.
#[derive(Clone, Copy, Debug)]
struct Interval {
    l: f64,
    r: f64,
    min_x: f64,
    max_x: f64,
    slot: u32,
}

impl PartialEq for Interval {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Interval {}

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Interval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.l
            .total_cmp(&other.l)
            .then(self.r.total_cmp(&other.r))
            .then(self.slot.cmp(&other.slot))
    }
}

/// One bucket of the y index.
trait Bucket: Default {
    fn insert(&mut self, interval: Interval);
    fn erase(&mut self, interval: &Interval);
    fn is_empty(&self) -> bool;
    /// Call `found` with every interval whose left end lies in `from..=hi`
    /// and whose right end is at least `lo`.
    fn scan(&self, from: f64, lo: f64, hi: f64, found: &mut impl FnMut(&Interval));
}

impl Bucket for BTreeSet<Interval> {
    fn insert(&mut self, interval: Interval) {
        BTreeSet::insert(self, interval);
    }
    fn erase(&mut self, interval: &Interval) {
        self.remove(interval);
    }
    fn is_empty(&self) -> bool {
        BTreeSet::is_empty(self)
    }
    fn scan(&self, from: f64, lo: f64, hi: f64, found: &mut impl FnMut(&Interval)) {
        let start = Interval {
            l: from,
            r: f64::NEG_INFINITY,
            min_x: 0.0,
            max_x: 0.0,
            slot: 0,
        };
        for interval in self.range(start..) {
            if interval.l > hi {
                break;
            }
            if interval.r >= lo {
                found(interval);
            }
        }
    }
}

#[derive(Default)]
struct SortedVec(Vec<Interval>);

impl Bucket for SortedVec {
    fn insert(&mut self, interval: Interval) {
        let at = self.0.partition_point(|other| *other < interval);
        self.0.insert(at, interval);
    }
    fn erase(&mut self, interval: &Interval) {
        if let Ok(at) = self.0.binary_search(interval) {
            self.0.remove(at);
        }
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn scan(&self, from: f64, lo: f64, hi: f64, found: &mut impl FnMut(&Interval)) {
        let start = self.0.partition_point(|interval| interval.l < from);
        for interval in &self.0[start..] {
            if interval.l > hi {
                break;
            }
            if interval.r >= lo {
                found(interval);
            }
        }
    }
}

/// The active set of a sweep, indexed by y.
trait ActiveIndex {
    fn insert(&mut self, interval: Interval);
    fn erase(&mut self, interval: &Interval);
    /// Every interval overlapping `lo..=hi`, edges included, and possibly
    /// others.
    fn overlapping(&self, lo: f64, hi: f64, found: impl FnMut(&Interval));
}

/// `IntervalIdx`: the buckets by span, and the largest span of the overflow.
struct IntervalIdx<B> {
    buckets: [B; BUCKETS + 1],
    max_span: f64,
}

impl<B: Bucket> ActiveIndex for IntervalIdx<B> {
    fn insert(&mut self, interval: Interval) {
        IntervalIdx::insert(self, interval);
    }
    fn erase(&mut self, interval: &Interval) {
        IntervalIdx::erase(self, interval);
    }
    fn overlapping(&self, lo: f64, hi: f64, found: impl FnMut(&Interval)) {
        IntervalIdx::overlapping(self, lo, hi, found);
    }
}

impl<B: Bucket> IntervalIdx<B> {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| B::default()),
            max_span: 0.0,
        }
    }

    fn bucket(span: f64) -> usize {
        let mut limit = 10.0;
        for bucket in 0..BUCKETS {
            if span < limit {
                return bucket;
            }
            limit *= 10.0;
        }
        BUCKETS
    }

    fn insert(&mut self, interval: Interval) {
        let span = interval.r - interval.l;
        let bucket = Self::bucket(span);
        self.buckets[bucket].insert(interval);
        if bucket == BUCKETS && span > self.max_span {
            self.max_span = span;
        }
    }

    fn erase(&mut self, interval: &Interval) {
        self.buckets[Self::bucket(interval.r - interval.l)].erase(interval);
    }

    /// Every interval overlapping `lo..=hi`, edges included.
    fn overlapping(&self, lo: f64, hi: f64, mut found: impl FnMut(&Interval)) {
        let mut limit = 10.0;
        for bucket in &self.buckets[..BUCKETS] {
            if !bucket.is_empty() {
                bucket.scan(lo - limit, lo, hi, &mut found);
            }
            limit *= 10.0;
        }
        let overflow = &self.buckets[BUCKETS];
        if !overflow.is_empty() {
            overflow.scan(lo - self.max_span, lo, hi, &mut found);
        }
    }
}

/// The cells of [`ActiveSet::Cells`] over the lower ends of one sweep's
/// items, with an overflow list for lower ends that are not finite.
struct CellIndex {
    origin: f64,
    inverse_height: f64,
    cells: Vec<Vec<Interval>>,
    overflow: Vec<Interval>,
    max_span: f64,
}

impl CellIndex {
    /// Cells of `height` over the finite lower ends of `events`, fewer and
    /// taller if that would make more than four cells per item.
    fn new(events: &[Event], height: f64) -> Self {
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for event in events {
            if event.min_y.is_finite() {
                low = low.min(event.min_y);
                high = high.max(event.min_y);
            }
        }
        let range = if low <= high { high - low } else { 0.0 };
        let count = ((range / height) as usize + 1).clamp(1, 64.max(2 * events.len()));
        Self {
            origin: if low <= high { low } else { 0.0 },
            inverse_height: count as f64 / range.max(f64::MIN_POSITIVE),
            cells: vec![Vec::new(); count],
            overflow: Vec::new(),
            max_span: 0.0,
        }
    }

    /// The cell of a lower end, clamped to the cells; `None` if not finite.
    fn cell(&self, y: f64) -> Option<usize> {
        if !y.is_finite() {
            return None;
        }
        let cell = ((y - self.origin) * self.inverse_height).max(0.0) as usize;
        Some(cell.min(self.cells.len() - 1))
    }
}

impl ActiveIndex for CellIndex {
    fn insert(&mut self, interval: Interval) {
        let span = interval.r - interval.l;
        if span > self.max_span {
            self.max_span = span;
        }
        match self.cell(interval.l) {
            Some(cell) => self.cells[cell].push(interval),
            None => self.overflow.push(interval),
        }
    }
    fn erase(&mut self, interval: &Interval) {
        let list = match self.cell(interval.l) {
            Some(cell) => &mut self.cells[cell],
            None => &mut self.overflow,
        };
        if let Some(at) = list.iter().position(|other| other.slot == interval.slot) {
            list.swap_remove(at);
        }
    }
    fn overlapping(&self, lo: f64, hi: f64, mut found: impl FnMut(&Interval)) {
        // An infinite largest span reaches every cell.
        let first = self.cell(lo - self.max_span).unwrap_or(0);
        let last = self.cell(hi).unwrap_or(self.cells.len() - 1);
        for cell in &self.cells[first..=last.max(first)] {
            for interval in cell {
                if interval.l <= hi && interval.r >= lo {
                    found(interval);
                }
            }
        }
        for interval in &self.overflow {
            if interval.l <= hi && interval.r >= lo {
                found(interval);
            }
        }
    }
}

/// The buffers of a pair query that sorts the events afresh: the sort keys,
/// and the events in their order.
#[derive(Default)]
struct Scratch {
    keys: Vec<(u64, u32, u32)>,
    events: Vec<Event>,
}

/// `x` as an integer in the order of `f64::total_cmp`.
fn order_key(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    }
}

pub struct SpatialJoinSweep<T: Copy> {
    order: EventOrder,
    active: ActiveSet,
    /// Every slot's item; a removed item's slot keeps it until reused.
    items: Vec<T>,
    slots: Slots<()>,
    /// Positions in `events` of each slot's entering and leaving event.
    positions: Vec<[u32; 2]>,
    /// Two events per live item, with its current bounds; with
    /// [`EventOrder::Kept`], sorted while `sorted`.
    events: Vec<Event>,
    sorted: bool,
    /// Kept between pair queries so that sorting afresh allocates nothing;
    /// a lock rather than a cell keeps the structure `Sync`.
    scratch: Mutex<Scratch>,
}

impl<T: Copy> SpatialJoinSweep<T> {
    pub fn new(order: EventOrder, active: ActiveSet) -> Self {
        Self {
            order,
            active,
            items: Vec::new(),
            slots: Slots::new(),
            positions: Vec::new(),
            events: Vec::new(),
            sorted: true,
            scratch: Mutex::default(),
        }
    }

    /// Insert an item and return the handle that identifies it for
    /// [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let position = self.events.len() as u32;
        let handle = self.slots.allocate(());
        let slot = handle.index();
        assert!(slot < 1 << 31, "Too many slots for the sort keys.");
        if slot == self.items.len() {
            self.items.push(item);
            self.positions.push([position, position + 1]);
        } else {
            self.items[slot] = item;
            self.positions[slot] = [position, position + 1];
        }
        for leaving in [false, true] {
            let mut event = Event {
                x: 0.0,
                min_x: 0.0,
                min_y: 0.0,
                max_x: 0.0,
                max_y: 0.0,
                slot: slot as u32,
                leaving,
            };
            event.set_bounds(&bounds);
            self.events.push(event);
        }
        self.sorted = false;
        handle
    }

    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        self.slots.get(handle);
        let slot = handle.index() as u32;
        self.slots.free(handle);
        self.events.retain(|event| event.slot != slot);
        // Every later event moved; the order among the rest is unchanged.
        for position in 0..self.events.len() {
            self.set_position(position);
        }
    }

    /// Write the bounds into both events. Panics if the handle has been
    /// removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.slots.get(handle);
        let [entering, leaving] = self.positions[handle.index()];
        self.events[entering as usize].set_bounds(&bounds);
        self.events[leaving as usize].set_bounds(&bounds);
        self.sorted = false;
    }

    /// With [`EventOrder::Kept`], sort the events again by insertion; nothing
    /// otherwise.
    pub fn refresh_order(&mut self) {
        if self.order == EventOrder::Kept && !self.sorted {
            if !self.insertion_sort() {
                self.events.sort_unstable_by(Event::cmp);
                for position in 0..self.events.len() {
                    self.set_position(position);
                }
            }
            self.sorted = true;
        }
    }

    fn set_position(&mut self, position: usize) {
        let event = &self.events[position];
        self.positions[event.slot as usize][event.leaving as usize] = position as u32;
    }

    /// Sort by insertion, giving up once the shifts exceed [`SHIFT_BUDGET`]
    /// per event, and return whether it finished.
    fn insertion_sort(&mut self) -> bool {
        let mut budget = SHIFT_BUDGET * self.events.len();
        for i in 1..self.events.len() {
            let event = self.events[i];
            if self.events[i - 1].cmp(&event).is_le() {
                continue;
            }
            let mut j = i;
            while j > 0 && self.events[j - 1].cmp(&event).is_gt() {
                self.events[j] = self.events[j - 1];
                self.set_position(j);
                j -= 1;
            }
            self.events[j] = event;
            self.set_position(j);
            let shifts = i - j;
            if shifts > budget {
                return false;
            }
            budget -= shifts;
        }
        true
    }

    /// Call `callback` once for every pair of distinct stored items whose
    /// bounds intersect, with the two items in either order. Sweeps the kept
    /// events if they are up to date, else a sorted copy: the events' keys
    /// (x, then entering before leaving, then slot) are sorted as integers,
    /// and the events gathered in their order.
    pub fn for_each_pair(&self, mut callback: impl FnMut(T, T)) {
        if self.order == EventOrder::Kept && self.sorted {
            self.sweep_events(&self.events, &mut callback);
            return;
        }
        let mut scratch = self
            .scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Scratch { keys, events } = &mut *scratch;
        keys.clear();
        keys.extend(self.events.iter().zip(0..).map(|(event, position)| {
            (
                order_key(event.x),
                (event.leaving as u32) << 31 | event.slot,
                position,
            )
        }));
        keys.sort_unstable();
        events.clear();
        events.extend(
            keys.iter()
                .map(|&(_, _, position)| self.events[position as usize]),
        );
        self.sweep_events(events, &mut callback);
    }

    fn sweep_events(&self, events: &[Event], callback: &mut impl FnMut(T, T)) {
        match self.active {
            ActiveSet::BTree => {
                self.sweep(events, IntervalIdx::<BTreeSet<Interval>>::new(), callback)
            }
            ActiveSet::SortedVec => self.sweep(events, IntervalIdx::<SortedVec>::new(), callback),
            ActiveSet::Cells(height) => {
                self.sweep(events, CellIndex::new(events, height), callback)
            }
        }
    }

    fn sweep(
        &self,
        events: &[Event],
        mut active: impl ActiveIndex,
        callback: &mut impl FnMut(T, T),
    ) {
        for event in events {
            if event.has_nan() {
                continue;
            }
            let interval = Interval {
                l: event.min_y,
                r: event.max_y,
                min_x: event.min_x,
                max_x: event.max_x,
                slot: event.slot,
            };
            if !event.leaving {
                active.insert(interval);
                continue;
            }
            active.erase(&interval);
            active.overlapping(event.min_y, event.max_y, |other| {
                if other.min_x < event.max_x
                    && event.min_x < other.max_x
                    && other.l < event.max_y
                    && event.min_y < other.r
                {
                    callback(
                        self.items[event.slot as usize],
                        self.items[other.slot as usize],
                    );
                }
            });
        }
    }

    /// Call `callback` once for every item whose bounds intersect `bounds`,
    /// testing every item.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        for event in self.events.iter().filter(|event| !event.leaving) {
            if event.bounds().intersects(bounds) {
                callback(self.items[event.slot as usize]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VARIANTS: [(EventOrder, ActiveSet); 6] = [
        (EventOrder::Sorted, ActiveSet::BTree),
        (EventOrder::Sorted, ActiveSet::SortedVec),
        (EventOrder::Kept, ActiveSet::BTree),
        (EventOrder::Kept, ActiveSet::SortedVec),
        (EventOrder::Sorted, ActiveSet::Cells(0.7)),
        (EventOrder::Kept, ActiveSet::Cells(3.0)),
    ];

    fn pairs(sweep: &SpatialJoinSweep<usize>) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        sweep.for_each_pair(|a, b| pairs.push((a.min(b), a.max(b))));
        pairs.sort_unstable();
        pairs
    }

    fn brute_force(items: &[Rectangle]) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for (a, a_bounds) in items.iter().enumerate() {
            for (b, b_bounds) in items.iter().enumerate().skip(a + 1) {
                if a_bounds.intersects(b_bounds) {
                    pairs.push((a, b));
                }
            }
        }
        pairs
    }

    /// Touching boxes, equal edges, zero-sized boxes, NaN of either sign, spans in several
    /// buckets and past the last limit, an infinite span, and negative coordinates.
    #[test]
    fn edge_cases_match_brute_force() {
        let items = [
            Rectangle::new(0.0, 0.0, 1.0, 1.0),
            Rectangle::new(1.0, 0.0, 2.0, 1.0),
            Rectangle::new(0.5, 1.0, 1.5, 2.0),
            Rectangle::new(0.5, 0.5, 0.5, 0.5),
            Rectangle::new(0.0, 0.0, 1.0, 1.0),
            Rectangle::new(0.2, f64::NAN, 0.8, 0.9),
            Rectangle::new(-5.0, -500.0, 0.1, 0.1),
            Rectangle::new(-1e9, -1e9, 1e9, 1e9),
            Rectangle::new(0.9, -2e8, 3.0, 0.05),
            Rectangle::new(0.5, 0.0, 0.5, 2.0),
            // A negative NaN sorts before every number, where it would break the
            // binary searches of a sorted bucket.
            Rectangle::new(0.3, -f64::NAN, 0.7, 0.8),
            Rectangle::new(-f64::NAN, 0.1, 0.9, 0.95),
            // A lower end of minus infinity, outside every cell.
            Rectangle::new(-10.0, f64::NEG_INFINITY, 10.0, 0.3),
        ];
        for (order, active) in VARIANTS {
            let mut sweep = SpatialJoinSweep::new(order, active);
            for (id, bounds) in items.iter().enumerate() {
                sweep.insert(bounds.clone(), id);
            }
            assert_eq!(
                pairs(&sweep),
                brute_force(&items),
                "{order:?} {active:?}, stale"
            );
            sweep.refresh_order();
            assert_eq!(
                pairs(&sweep),
                brute_force(&items),
                "{order:?} {active:?}, refreshed"
            );
        }
    }

    /// A slot freed and reused must not keep the old item's events, and the
    /// events of the others must stay reachable from their handles.
    #[test]
    fn removed_items_leave_the_sweep() {
        for (order, active) in VARIANTS {
            let mut sweep = SpatialJoinSweep::new(order, active);
            let handles: Vec<_> = (0..6)
                .map(|id| sweep.insert(Rectangle::new(id as f64, 0.0, id as f64 + 1.5, 1.0), id))
                .collect();
            sweep.refresh_order();
            sweep.remove(handles[2]);
            assert_eq!(
                pairs(&sweep),
                [(0, 1), (3, 4), (4, 5)],
                "{order:?} {active:?}"
            );
            let reused = sweep.insert(Rectangle::new(100.0, 0.0, 101.0, 1.0), 9);
            assert_eq!(reused.index(), handles[2].index());
            sweep.refresh_order();
            assert_eq!(
                pairs(&sweep),
                [(0, 1), (3, 4), (4, 5)],
                "{order:?} {active:?}"
            );
            sweep.update(Rectangle::new(1.2, 0.5, 1.4, 0.6), reused);
            sweep.update(Rectangle::new(50.0, 0.0, 51.0, 1.0), handles[5]);
            assert_eq!(
                pairs(&sweep),
                [(0, 1), (0, 9), (1, 9), (3, 4)],
                "{order:?} {active:?}"
            );
            sweep.refresh_order();
            assert_eq!(
                pairs(&sweep),
                [(0, 1), (0, 9), (1, 9), (3, 4)],
                "{order:?} {active:?}"
            );
        }
    }

    /// The kept order survives moves of more than the shift budget, and the
    /// handles still reach their events after the full sort.
    #[test]
    fn kept_order_recovers_from_large_moves() {
        for active in [ActiveSet::SortedVec, ActiveSet::Cells(2.0)] {
            let mut sweep = SpatialJoinSweep::new(EventOrder::Kept, active);
            let mut items: Vec<Rectangle> = (0..200)
                .map(|id| {
                    Rectangle::new(
                        id as f64,
                        (id % 7) as f64,
                        id as f64 + 3.0,
                        (id % 7) as f64 + 2.0,
                    )
                })
                .collect();
            let handles: Vec<_> = items
                .iter()
                .enumerate()
                .map(|(id, b)| sweep.insert(b.clone(), id))
                .collect();
            sweep.refresh_order();
            for round in 0..2 {
                for (id, bounds) in items.iter_mut().enumerate() {
                    let x = if round == 0 {
                        199.0 - id as f64
                    } else {
                        id as f64 * 0.5
                    };
                    *bounds = Rectangle::new(x, bounds.min_y, x + 3.0, bounds.max_y);
                    sweep.update(bounds.clone(), handles[id]);
                }
                sweep.refresh_order();
                assert_eq!(
                    pairs(&sweep),
                    brute_force(&items),
                    "{active:?}, round {round}"
                );
            }
        }
    }
}
