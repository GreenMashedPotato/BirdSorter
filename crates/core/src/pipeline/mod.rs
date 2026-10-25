//! The ingest pipeline: decode -> detect subjects -> features -> verdict -> store.
//!
//! Dependencies are injected through the [`ImageDecoder`], [`SubjectDetector`]
//! and [`VerdictScorer`] traits so the whole pipeline — including cancel,
//! batching and per-file failure isolation — can be unit-tested with fakes,
//! with no RAW files, models or GUI involved.
//!
//! The loop is deliberately deterministic (single-threaded, batch-at-a-time).
//! Parallel decode is an optimisation to add later behind a bench, not a
//! correctness requirement.

pub mod ingest;
pub mod stages;

pub use ingest::{
    analyze_file_with, ingest_folder, rescore_folder, rules_scorer, scorer_for, CancelToken,
    DecodedImage, ImageDecoder, IngestContext, IngestProgress, RulesScorer, StageTimes,
    SubjectDetector, VerdictScorer, WarmRenders,
};
