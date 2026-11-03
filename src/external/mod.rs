//! Structures of other research, with the interface of this crate's own so
//! that the benchmarks can compare them on the same items, queries and pair
//! rules: ports of published methods, [`TwoLayerGrid`], [`MultiGrid`] and
//! [`SpatialJoinSweep`], and
//! adapters of external libraries, `ParryBvh` with the `parry` feature and
//! `PackedRTree` with the `packed` feature. An
//! adapter keeps the library's own maintenance and query paths and adds only
//! what the interface needs: handles, exact filtering with
//! [`Rectangle::intersects`], and exact results between refreshes.
//!
//! [`Rectangle::intersects`]: crate::rect::Rectangle::intersects

pub mod multi_grid;
#[cfg(feature = "packed")]
pub mod packed_rtree;
#[cfg(feature = "parry")]
pub mod parry_bvh;
pub mod spatial_join_sweep;
pub mod two_layer_grid;

pub use multi_grid::*;
#[cfg(feature = "packed")]
pub use packed_rtree::*;
#[cfg(feature = "parry")]
pub use parry_bvh::*;
pub use spatial_join_sweep::*;
pub use two_layer_grid::*;
