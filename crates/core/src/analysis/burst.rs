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
    /// True when the user picked this frame as the burst winner by hand.
    pub manual_winner: bool,
}

/// A detected burst and its winner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Burst {
    /// File ids in capture order.
    pub frames: Vec<i64>,
    /// The winning frame: the user's choice when there is one, else the sharpest.
    pub best: i64,
    pub size: usize,
    /// True when `best` is a manual choice rather than the highest score.
    pub winner_override: bool,
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

/// The best frame in a set: the user's manual winner when there is one,
/// otherwise the sharpest. Ties break to the smallest file id so the result is
/// deterministic.
pub fn best_in_burst(frames: &[Frame]) -> Option<i64> {
    if let Some(pick) = frames
        .iter()
        .filter(|f| f.manual_winner)
        .min_by_key(|f| f.file_id)
    {
        return Some(pick.file_id);
    }
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
            let winner_override = members.iter().any(|f| f.manual_winner);
            Burst {
                size: ids.len(),
                frames: ids,
                best,
                winner_override,
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
                COALESCE(a.sharpness_score, 0.0) AS score,
                f.burst_winner
           FROM files f
           LEFT JOIN analysis a ON a.file_id = f.id
          WHERE f.folder_id = ?1 AND f.trashed_at IS NULL",
    )?;
    let rows = stmt.query_map([folder_id], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, Option<String>>(1)?,
            r.get::<_, f32>(2)?,
            r.get::<_, i64>(3)? != 0,
        ))
    })?;

    let mut frames: Vec<Frame> = Vec::new();
    for row in rows {
        let (file_id, captured, score, manual_winner) = row?;
        frames.push(Frame {
            file_id,
            captured_at: captured
                .as_deref()
                .and_then(crate::io::exif::parse_capture_time),
            score,
            manual_winner,
        });
    }
    Ok(find_bursts(&frames, gap_seconds))
}

/// Mark `file_id` as its burst's winner, or clear the manual choice.
///
/// Exactly one frame per burst keeps the flag: marking a new winner clears the
/// previous one. Returns the folder's refreshed bursts so the caller can update
/// the grid without a second query.
pub fn set_burst_winner(conn: &Connection, file_id: i64, winner: bool) -> Result<Vec<Burst>> {
    let folder_id: i64 = conn.query_row(
        "SELECT folder_id FROM files WHERE id = ?1",
        [file_id],
        |r| r.get(0),
    )?;
    let bursts = folder_bursts(conn, folder_id, DEFAULT_GAP_SECONDS)?;
    let members: Vec<i64> = bursts
        .iter()
        .find(|b| b.frames.contains(&file_id))
        .map(|b| b.frames.clone())
        .unwrap_or_else(|| vec![file_id]);

    crate::data::db::with_tx(conn, || {
        let mut clear = conn.prepare("UPDATE files SET burst_winner = 0 WHERE id = ?1")?;
        for id in &members {
            clear.execute([id])?;
        }
        if winner {
            conn.execute("UPDATE files SET burst_winner = 1 WHERE id = ?1", [file_id])?;
        }
        Ok(())
    })?;
    folder_bursts(conn, folder_id, DEFAULT_GAP_SECONDS)
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
            manual_winner: false,
        }
    }

    fn chosen(id: i64, secs: i64, score: f32) -> Frame {
        Frame {
            manual_winner: true,
            ..frame(id, secs, score)
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
                manual_winner: false,
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
    fn manual_winner_overrides_the_sharpest_frame() {
        let frames = vec![frame(5, 0, 0.2), chosen(2, 1, 0.9), frame(8, 2, 0.4)];
        assert_eq!(best_in_burst(&frames), Some(2));
        let bursts = find_bursts(&frames, 2);
        assert_eq!(bursts[0].best, 2);
        assert!(bursts[0].winner_override);
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
