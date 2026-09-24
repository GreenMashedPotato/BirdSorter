//! Burst detection and best-of-burst ranking.
//!
//! Ported from the prototype's `analysis/burst.py`. A burst is a run of frames
//! whose capture times are no more than `gap_seconds` apart. This is what lets
//! a flight sequence collapse to its sharpest frame.

use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::Result;

/// Default maximum gap between frames of the same burst.
pub const DEFAULT_GAP_SECONDS: i64 = 2;

/// The minimum a frame needs to contribute to burst analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub file_id: i64,
    /// `None` when EXIF carried no usable timestamp.
    pub captured_at: Option<OffsetDateTime>,
    /// Verdict score; higher is sharper.
    pub score: f32,
}

/// A detected burst and its winner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Burst {
    /// File ids in capture order.
    pub frames: Vec<i64>,
    /// The sharpest frame in the burst.
    pub best: i64,
    pub size: usize,
}

/// Group frames into bursts. Frames without a timestamp become singletons.
///
/// Input order does not matter; frames with timestamps are sorted by capture
/// time first.
pub fn group_bursts(frames: &[Frame], gap_seconds: i64) -> Vec<Vec<i64>> {
    let mut timed: Vec<&Frame> = frames.iter().filter(|f| f.captured_at.is_some()).collect();
    timed.sort_by_key(|f| f.captured_at);

    let mut bursts: Vec<Vec<i64>> = Vec::new();
    let mut current: Vec<i64> = Vec::new();
    let mut prev: Option<OffsetDateTime> = None;

    for f in timed {
        let t = f.captured_at.expect("filtered to Some");
        let same_burst = match prev {
            Some(p) => (t - p).whole_seconds() <= gap_seconds,
            None => false,
        };
        if !same_burst && !current.is_empty() {
            bursts.push(std::mem::take(&mut current));
        }
        current.push(f.file_id);
        prev = Some(t);
    }
    if !current.is_empty() {
        bursts.push(current);
    }

    // Timestamp-less frames cannot be grouped; each stands alone.
    for f in frames.iter().filter(|f| f.captured_at.is_none()) {
        bursts.push(vec![f.file_id]);
    }
    bursts
}

/// The sharpest frame in a set; ties break to the smallest file id so the
/// result is deterministic.
pub fn best_in_burst(frames: &[Frame]) -> Option<i64> {
    frames
        .iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.file_id.cmp(&a.file_id))
        })
        .map(|f| f.file_id)
}

/// Group frames and rank each burst, ready for the UI.
pub fn find_bursts(frames: &[Frame], gap_seconds: i64) -> Vec<Burst> {
    group_bursts(frames, gap_seconds)
        .into_iter()
        .map(|ids| {
            let members: Vec<Frame> = ids
                .iter()
                .filter_map(|id| frames.iter().find(|f| f.file_id == *id).copied())
                .collect();
            let best = best_in_burst(&members).unwrap_or(ids[0]);
            Burst {
                size: ids.len(),
                frames: ids,
                best,
            }
        })
        .collect()
}

/// Group the files of a folder into bursts and rank each.
///
/// `scores` supplies each file's sharpness (higher = better); files without a
/// capture time become singletons. Returns bursts newest-last in capture order.
pub fn folder_bursts(conn: &Connection, folder_id: i64, gap_seconds: i64) -> Result<Vec<Burst>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.captured_at,
                COALESCE(a.sharpness_score, 0.0) AS score
           FROM files f
           LEFT JOIN analysis a ON a.file_id = f.id
          WHERE f.folder_id = ?1 AND f.trashed_at IS NULL",
    )?;
    let rows = stmt.query_map([folder_id], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, Option<String>>(1)?,
            r.get::<_, f32>(2)?,
        ))
    })?;

    let mut frames: Vec<Frame> = Vec::new();
    for row in rows {
        let (file_id, captured, score) = row?;
        frames.push(Frame {
            file_id,
            captured_at: captured
                .as_deref()
                .and_then(crate::io::exif::parse_capture_time),
            score,
        });
    }
    Ok(find_bursts(&frames, gap_seconds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::datetime;

    fn frame(id: i64, secs: i64, score: f32) -> Frame {
        Frame {
            file_id: id,
            captured_at: Some(datetime!(2026-01-01 10:00:00 UTC) + time::Duration::seconds(secs)),
            score,
        }
    }

    #[test]
    fn groups_consecutive_frames_and_splits_on_gaps() {
        let frames = vec![
            frame(1, 0, 0.1),
            frame(2, 1, 0.2),
            frame(3, 3, 0.3),  // within 2s of frame 2
            frame(4, 30, 0.4), // big gap
            frame(5, 31, 0.5),
        ];
        let bursts = group_bursts(&frames, 2);
        assert_eq!(bursts, vec![vec![1, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn sorts_unordered_input_by_capture_time() {
        let frames = vec![frame(3, 2, 0.3), frame(1, 0, 0.1), frame(2, 1, 0.2)];
        assert_eq!(group_bursts(&frames, 2), vec![vec![1, 2, 3]]);
    }

    #[test]
    fn timestamp_less_frames_are_singletons() {
        let frames = vec![
            Frame {
                file_id: 9,
                captured_at: None,
                score: 0.9,
            },
            frame(1, 0, 0.1),
            frame(2, 1, 0.2),
        ];
        let bursts = group_bursts(&frames, 2);
        assert_eq!(bursts, vec![vec![1, 2], vec![9]]);
    }

    #[test]
    fn best_in_burst_picks_highest_score_and_breaks_ties_by_id() {
        let frames = vec![frame(5, 0, 0.4), frame(2, 1, 0.9), frame(8, 2, 0.4)];
        assert_eq!(best_in_burst(&frames), Some(2));

        let tie = vec![frame(7, 0, 0.5), frame(3, 1, 0.5)];
        assert_eq!(best_in_burst(&tie), Some(3));
        assert_eq!(best_in_burst(&[]), None);
    }

    #[test]
    fn find_bursts_ranks_each_group() {
        let frames = vec![
            frame(1, 0, 0.2),
            frame(2, 1, 0.8),
            frame(3, 40, 0.3),
            frame(4, 41, 0.4),
        ];
        let bursts = find_bursts(&frames, 2);
        assert_eq!(bursts.len(), 2);
        assert_eq!(bursts[0].best, 2);
        assert_eq!(bursts[0].size, 2);
        assert_eq!(bursts[1].best, 4);
    }
}
