//! BirdSorter core.
//!
//! Everything that is not GUI glue lives here: RAW IO, device/runtime selection,
//! the analysis pipeline (detection, sharpness features, scoring, species ID),
//! the SQLite data layer and the learning loop.
//!
//! This crate has **no Tauri / GUI dependencies** and is fully headless and
//! unit-testable. The `birdsorter-desktop` crate is a thin shell over this, and
//! `birdsorter-cli` drives it from the command line / CI.
//!
//! Porting note: the Python prototype (RAW-Sorter) is the reference spec. Every
//! metric and rule ported here is covered by golden/ordering tests so behaviour
//! cannot silently drift.

pub mod analysis;
pub mod data;
pub mod error;
pub mod io;
pub mod learning;
pub mod pipeline;
pub mod runtime;

pub use error::{CoreError, Result};
