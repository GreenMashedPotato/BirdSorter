//! BirdSorter desktop app.
//!
//! A deliberately thin Tauri v2 shell: every piece of real logic lives in
//! `birdsorter-core`, which is headless and unit-tested. This crate only wires
//! commands, events and the app state together, split by concern:
//!
//! - `library`  — folders, scanning, queries, trash, ratings
//! - `analysis` — the ingest pipeline, review detail, bursts, box editing, train
//! - `app`      — settings, species list, About info, cache maintenance
//! - `export`   — copy/move keepers out of the library
//! - `annotations`, `preview`, `models` — species/labels/undo, rendering, models
//! - `startup`  — model seeding, panic hook, Tauri builder
//!
//! Responsiveness rules:
//! - Anything that touches disk or CPU (scan, query, thumbnail) runs on a
//!   blocking worker via `spawn_blocking`, never on the IPC/main thread.
//! - Analysis decode parallelism is capped so a big shoot cannot saturate the
//!   machine.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analysis;
mod annotations;
mod app;
mod config;
mod export;
mod library;
mod models;
mod preview;
mod startup;
mod state;
mod util;
mod xmp_sync;

fn main() {
    startup::run();
}
