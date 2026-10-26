//! Burst detection and best-of-burst ranking.
//!
//! Ported from the prototype's `analysis/burst.py`. A burst is a run of frames
//! whose capture times are no more than `gap_seconds` apart. This is what lets
//! a flight sequence collapse to its sharpest frame.

use ndarray::Array3;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::Result;

/// Default maximum gap between frames of the same burst.
pub const DEFAULT_GAP_SECONDS: i64 = 2;

/// Miss-focus reason stored when a frame is clearly softer than its burst's best.
pub const MISS_FOCUS_SOFT_BURST: &str = "soft_vs_burst";

/// A "sharp" frame must reach this fraction of its burst's best score; below it,
/// it is downgraded to "uncertain". Measured on a real burst: the keeper scored
/// 0.68 and the soft frame 0.50 (0.73×), which the per-image metrics still rated
/// "sharp".
pub const SOFT_BURST_RATIO: f32 = 0.75;

/// Minimum Pearson correlation between adjacent frames' luminance signatures
/// to keep them in the same burst. Frames without a signature (old analyses)
/// fall back to time-only grouping.
pub const DEFAULT_SIMILARITY: f32 = 0.75;

/// A bracket needs at least this many frames and this much exposure span.
pub const BRACKET_MIN_FRAMES: usize = 3;
pub const BRACKET_SPAN_EV: f32 = 1.0;

/// 64 signed cells + one `f32` mean luma.
const SIGNATURE_BYTES: usize = 68;

/// An 8×8 box-averaged luminance signature: mean-subtracted and L2-normalised
/// for similarity, plus the frame's overall mean luma for exposure brackets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameSignature {
    pub cells: [f32; 64],
    /// Mean sRGB luma in `0..=1`.
    pub mean_luma: f32,
}

impl FrameSignature {
    /// Encode as 64 quantised cells plus a little-endian `f32` mean.
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(SIGNATURE_BYTES);
        for value in self.cells {
            bytes.push((value * 127.0).round().clamp(-127.0, 127.0) as i8 as u8);
        }
        bytes.extend_from_slice(&self.mean_luma.to_le_bytes());
        bytes
    }

    /// Decode [`Self::encode`]d bytes; `None` for anything else.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != SIGNATURE_BYTES {
            return None;
        }
        let mut cells = [0.0f32; 64];
        for (slot, &byte) in cells.iter_mut().zip(bytes) {
            *slot = f32::from(byte as i8) / 127.0;
        }
        let mean_luma = f32::from_le_bytes(bytes.get(64..68)?.try_into().ok()?);
        Some(Self { cells, mean_luma })
    }
}

/// Pearson correlation of two signatures in `-1..=1`. A flat/empty signature
/// correlates with nothing.
pub fn similarity(a: &FrameSignature, b: &FrameSignature) -> f32 {
    a.cells
        .iter()
        .zip(&b.cells)
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

/// The brightest-to-darkest exposure ratio in stops, from the frames' mean
/// luma. `None` when no frame has a signature.
pub fn exposure_span_ev(signatures: &[FrameSignature]) -> Option<f32> {
    let mut min = f32::INFINITY;
    let mut max = 0.0f32;
    for signature in signatures {
        let linear = srgb_to_linear(signature.mean_luma.clamp(0.0, 1.0)).max(1e-4);
        min = min.min(linear);
        max = max.max(linear);
    }
    if !min.is_finite() || max <= 0.0 {
        return None;
    }
    Some((max / min).log2())
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Compute an 8×8 box-averaged luminance signature. `None` for an empty image.
pub fn signature(rgb: &Array3<u8>) -> Option<FrameSignature> {
    let (height, width, channels) = rgb.dim();
    if height == 0 || width == 0 || channels == 0 {
        return None;
    }
    let mut sums = [0.0f64; 64];
    let mut counts = [0u64; 64];
    let mut total = 0.0f64;
    for y in 0..height {
        let gy = y * 8 / height;
        for x in 0..width {
            let gx = x * 8 / width;
            let index = gy.min(7) * 8 + gx.min(7);
            let luma = 0.2126 * f64::from(rgb[[y, x, 0]])
                + 0.7152 * f64::from(rgb[[y, x, 1]])
                + 0.0722 * f64::from(rgb[[y, x, 2]]);
            sums[index] += luma;
            counts[index] += 1;
            total += luma;
        }
    }
    let pixels = height as u64 * width as u64;
    if pixels == 0 {
        return None;
    }
    let mut cells: [f32; 64] = std::array::from_fn(|i| {
        if counts[i] == 0 {
            0.0
        } else {
            (sums[i] / counts[i] as f64) as f32
        }
    });
    // Remove the cell-mean so brightness is not mistaken for a different scene.
    let cell_mean = cells.iter().sum::<f32>() / 64.0;
    for cell in &mut cells {
        *cell -= cell_mean;
    }
    let norm = cells.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for cell in &mut cells {
            *cell /= norm;
        }
    } else {
        cells = [0.0; 64];
    }
    Some(FrameSignature {
        cells,
        mean_luma: (total / pixels as f64) as f32 / 255.0,
    })
}

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
    /// Visual signature for similarity grouping; `None` for old analyses.
    pub signature: Option<FrameSignature>,
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
    /// True when the frames look like an exposure bracket.
    pub bracket: bool,
    /// Brightest-to-darkest exposure span in stops, when signatures exist.
    pub exposure_span_ev: Option<f32>,
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
    let mut prev_signature: Option<&FrameSignature> = None;

    for f in timed {
        let t = f.captured_at.expect("filtered to Some");
        let same_burst = match prev {
            Some(p) => {
                let close_in_time = (t - p).whole_seconds() <= gap_seconds;
                let alike = match (prev_signature, f.signature.as_ref()) {
                    (Some(a), Some(b)) => similarity(a, b) >= DEFAULT_SIMILARITY,
                    // Old analyses have no signature: keep time-only grouping.
                    _ => true,
                };
                close_in_time && alike
            }
            None => false,
        };
        if !same_burst && !current.is_empty() {
            bursts.push(std::mem::take(&mut current));
        }
        current.push(f.file_id);
        prev = Some(t);
        prev_signature = f.signature.as_ref();
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
            let signatures: Vec<FrameSignature> =
                members.iter().filter_map(|f| f.signature).collect();
            let exposure_span_ev = exposure_span_ev(&signatures);
            let bracket = members.len() >= BRACKET_MIN_FRAMES
                && exposure_span_ev.is_some_and(|span| span >= BRACKET_SPAN_EV);
            Burst {
                size: ids.len(),
                frames: ids,
                best,
                winner_override,
                bracket,
                exposure_span_ev,
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
                f.burst_winner, a.signature
           FROM files f
           LEFT JOIN analysis a ON a.file_id = f.id
              AND (a.analyzed_content_hash IS NULL
                   OR a.analyzed_content_hash = f.content_hash)
          WHERE f.folder_id = ?1 AND f.trashed_at IS NULL",
    )?;
    let rows = stmt.query_map([folder_id], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, Option<String>>(1)?,
            r.get::<_, f32>(2)?,
            r.get::<_, i64>(3)? != 0,
            r.get::<_, Option<Vec<u8>>>(4)?,
        ))
    })?;

    let mut frames: Vec<Frame> = Vec::new();
    for row in rows {
        let (file_id, captured, score, manual_winner, signature) = row?;
        frames.push(Frame {
            file_id,
            captured_at: captured
                .as_deref()
                .and_then(crate::io::exif::parse_capture_time),
            score,
            manual_winner,
            signature: signature.as_deref().and_then(FrameSignature::decode),
        });
    }
    Ok(find_bursts(&frames, gap_seconds))
}

/// Downgrade frames that are "sharp" but clearly softer than their burst's best.
///
/// Per-image metrics cannot see subject movement that is mild in absolute terms:
/// a frame can clear the sharp threshold while being visibly softer than the
/// keeper next to it (0.50 vs 0.68 in the same burst). Within a burst the sharpest
/// frame is the reference, so anything below `ratio` of it is marked `uncertain`
/// with the [`MISS_FOCUS_SOFT_BURST`] reason. Already-uncertain/blurry frames and
/// the burst winner are left alone. Returns the number of newly downgraded rows.
pub fn mark_soft_members(conn: &Connection, folder_id: i64, ratio: f32) -> Result<usize> {
    crate::data::db::with_tx(conn, || {
        // Undo the previous pass first: the winner or a score may have changed,
        // and a frame that is no longer soft must not stay 'uncertain' until a
        // full rescore. Only rows this pass marked are touched.
        conn.execute(
            "UPDATE analysis SET predicted_label = 'sharp', miss_focus_reason = NULL
              WHERE miss_focus_reason = ?1
                AND file_id IN (SELECT id FROM files WHERE folder_id = ?2)
                AND (analyzed_content_hash IS NULL OR analyzed_content_hash =
                     (SELECT content_hash FROM files WHERE id = analysis.file_id))",
            rusqlite::params![MISS_FOCUS_SOFT_BURST, folder_id],
        )?;

        let bursts = folder_bursts(conn, folder_id, DEFAULT_GAP_SECONDS)?;
        let mut updated = 0usize;
        for burst in bursts.iter().filter(|b| b.size > 1) {
            let best: Option<f32> = conn
                .query_row(
                    "SELECT a.sharpness_score
                       FROM analysis a JOIN files f ON f.id = a.file_id
                      WHERE a.file_id = ?1
                        AND (a.analyzed_content_hash IS NULL
                             OR a.analyzed_content_hash = f.content_hash)",
                    [burst.best],
                    |r| r.get(0),
                )
                .optional()?;
            let Some(best) = best.filter(|&s| s > 0.0) else {
                continue;
            };
            for &id in burst.frames.iter().filter(|&&id| id != burst.best) {
                let row: Option<(f32, String)> = conn
                    .query_row(
                        "SELECT a.sharpness_score, a.predicted_label
                           FROM analysis a JOIN files f ON f.id = a.file_id
                          WHERE a.file_id = ?1
                            AND (a.analyzed_content_hash IS NULL
                                 OR a.analyzed_content_hash = f.content_hash)",
                        [id],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()?;
                let Some((score, label)) = row else {
                    continue;
                };
                if label == "sharp" && score < ratio * best {
                    conn.execute(
                        "UPDATE analysis SET predicted_label = 'uncertain', miss_focus_reason = ?2 \
                          WHERE file_id = ?1
                            AND (analyzed_content_hash IS NULL OR analyzed_content_hash =
                                 (SELECT content_hash FROM files WHERE id = ?1))",
                        rusqlite::params![id, MISS_FOCUS_SOFT_BURST],
                    )?;
                    updated += 1;
                }
            }
        }
        Ok(updated)
    })
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
            signature: None,
        }
    }

    fn unit_signature(index: usize, mean_luma: f32) -> FrameSignature {
        let mut cells = [0.0f32; 64];
        cells[index] = 1.0;
        FrameSignature { cells, mean_luma }
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
                signature: None,
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
    fn stored_winner_persists_marks_and_clears() {
        use crate::data::db::{connect_in_memory, init_schema};
        use crate::data::library::{add_folder, list_files, scan_folder};

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.cr3"), b"1").unwrap();
        std::fs::write(dir.path().join("b.cr3"), b"2").unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let ids: Vec<i64> = list_files(&conn, fid, false, false)
            .unwrap()
            .into_iter()
            .map(|lf| lf.file.id)
            .collect();
        let a = ids[0];

        // Marking writes through and is reported as an override.
        let bursts = set_burst_winner(&conn, a, true).unwrap();
        let g = bursts.iter().find(|b| b.frames.contains(&a)).unwrap();
        assert_eq!(g.best, a);
        assert!(g.winner_override);

        // It survives a fresh query (persisted in `files.burst_winner`).
        let reread = folder_bursts(&conn, fid, DEFAULT_GAP_SECONDS).unwrap();
        assert!(
            reread
                .iter()
                .find(|b| b.frames.contains(&a))
                .unwrap()
                .winner_override
        );

        // Clearing removes it.
        let cleared = set_burst_winner(&conn, a, false).unwrap();
        assert!(
            !cleared
                .iter()
                .find(|b| b.frames.contains(&a))
                .unwrap()
                .winner_override
        );
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

    #[test]
    fn signature_is_mean_subtracted_normalised_and_brightness_aware() {
        let flat = Array3::from_elem((16, 16, 3), 200u8);
        let sig = signature(&flat).unwrap();
        assert!(
            sig.cells.iter().all(|v| v.abs() < 1e-6),
            "a flat frame has no structure"
        );
        assert!((sig.mean_luma - 200.0 / 255.0).abs() < 0.01);

        let gradient = Array3::from_shape_fn((16, 16, 3), |(_, x, _)| (x * 16) as u8);
        let sig = signature(&gradient).unwrap();
        assert!(sig.cells.iter().sum::<f32>().abs() < 1e-4);
        let norm = sig.cells.iter().map(|v| v * v).sum::<f32>();
        assert!((norm - 1.0).abs() < 1e-3);
    }

    #[test]
    fn signatures_group_by_look_not_only_time() {
        let mut a = frame(1, 0, 0.1);
        a.signature = Some(unit_signature(0, 0.5));
        let mut b = frame(2, 1, 0.2);
        b.signature = Some(unit_signature(0, 0.5));
        let mut c = frame(3, 2, 0.3);
        c.signature = Some(unit_signature(1, 0.5));
        assert_eq!(group_bursts(&[a, b, c], 2), vec![vec![1, 2], vec![3]]);
    }

    #[test]
    fn missing_signatures_keep_time_only_grouping() {
        let frames = vec![frame(1, 0, 0.1), frame(2, 1, 0.2)];
        assert_eq!(group_bursts(&frames, 2), vec![vec![1, 2]]);
    }

    #[test]
    fn exposure_span_marks_a_bracket() {
        let mut frames = vec![frame(1, 0, 0.1), frame(2, 1, 0.2), frame(3, 2, 0.3)];
        for (frame, mean) in frames.iter_mut().zip([0.1, 0.25, 0.5]) {
            frame.signature = Some(unit_signature(0, mean));
        }
        let bursts = find_bursts(&frames, 2);
        assert!(bursts[0].bracket);
        assert!(bursts[0].exposure_span_ev.unwrap() >= BRACKET_SPAN_EV);

        let flat = find_bursts(&[frame(4, 0, 0.1), frame(5, 1, 0.2), frame(6, 2, 0.3)], 2);
        assert!(!flat[0].bracket);
        assert_eq!(flat[0].exposure_span_ev, None);
    }

    #[test]
    fn soft_burst_members_are_downgraded() {
        use crate::data::db::{connect_in_memory, init_schema};
        use crate::data::library::{add_folder, scan_folder};

        let dir = tempfile::tempdir().unwrap();
        for name in ["a.cr3", "b.cr3", "c.cr3"] {
            std::fs::write(dir.path().join(name), name.as_bytes()).unwrap();
        }
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        conn.execute_batch(
            "UPDATE files SET captured_at = '2026-01-01T10:00:00'
              WHERE filename IN ('a.cr3', 'b.cr3');
             UPDATE files SET captured_at = '2026-01-01T10:00:01' WHERE filename = 'c.cr3';
             INSERT INTO analysis (file_id, features_json, sharpness_score, predicted_label,
                                   analyzed_at, model_version)
                 SELECT id, '{}', 0.68, 'sharp', 'now', 'rules-v1'
                   FROM files WHERE filename = 'a.cr3';
             INSERT INTO analysis (file_id, features_json, sharpness_score, predicted_label,
                                   analyzed_at, model_version)
                 SELECT id, '{}', 0.50, 'sharp', 'now', 'rules-v1'
                   FROM files WHERE filename = 'b.cr3';
             INSERT INTO analysis (file_id, features_json, sharpness_score, predicted_label,
                                   analyzed_at, model_version)
                 SELECT id, '{}', 0.30, 'blurry', 'now', 'rules-v1'
                   FROM files WHERE filename = 'c.cr3';",
        )
        .unwrap();

        assert_eq!(mark_soft_members(&conn, fid, SOFT_BURST_RATIO).unwrap(), 1);

        let (label, reason): (String, Option<String>) = conn
            .query_row(
                "SELECT a.predicted_label, a.miss_focus_reason
                   FROM analysis a JOIN files f ON f.id = a.file_id
                  WHERE f.filename = 'b.cr3'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(label, "uncertain");
        assert_eq!(reason.as_deref(), Some(MISS_FOCUS_SOFT_BURST));

        // The keeper stays sharp; the already-blurry frame is untouched.
        for (name, expected) in [("a.cr3", "sharp"), ("c.cr3", "blurry")] {
            let label: String = conn
                .query_row(
                    "SELECT a.predicted_label FROM analysis a JOIN files f ON f.id = a.file_id
                      WHERE f.filename = ?1",
                    [name],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(label, expected, "{name}");
        }

        // Making the downgraded frame the sharpest restores it on the next pass
        // instead of leaving it 'uncertain' until a full rescore.
        conn.execute(
            "UPDATE analysis SET sharpness_score = 0.70
              WHERE file_id = (SELECT id FROM files WHERE filename = 'b.cr3')",
            [],
        )
        .unwrap();
        assert_eq!(mark_soft_members(&conn, fid, SOFT_BURST_RATIO).unwrap(), 0);
        let (label, reason): (String, Option<String>) = conn
            .query_row(
                "SELECT a.predicted_label, a.miss_focus_reason
                   FROM analysis a JOIN files f ON f.id = a.file_id
                  WHERE f.filename = 'b.cr3'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(label, "sharp");
        assert_eq!(reason, None);
    }

    #[test]
    fn replaced_content_is_not_used_for_burst_scores() {
        use crate::data::db::{connect_in_memory, init_schema};
        use crate::data::library::{add_folder, scan_folder};

        let dir = tempfile::tempdir().unwrap();
        for name in ["a.cr3", "b.cr3"] {
            std::fs::write(dir.path().join(name), name.as_bytes()).unwrap();
        }
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        conn.execute_batch(
            "UPDATE files SET captured_at = '2026-01-01T10:00:00'
                WHERE filename IN ('a.cr3', 'b.cr3');
             INSERT INTO analysis (file_id, features_json, sharpness_score, predicted_label,
                                   analyzed_at, model_version, analyzed_content_hash)
                 SELECT id, '{}', 0.68, 'sharp', 'now', 'rules-v1', content_hash
                   FROM files WHERE filename = 'a.cr3';
             INSERT INTO analysis (file_id, features_json, sharpness_score, predicted_label,
                                   analyzed_at, model_version, analyzed_content_hash)
                 SELECT id, '{}', 0.50, 'sharp', 'now', 'rules-v1', content_hash
                   FROM files WHERE filename = 'b.cr3';
             UPDATE files SET content_hash = 'replacement-hash' WHERE filename = 'b.cr3';",
        )
        .unwrap();

        assert_eq!(mark_soft_members(&conn, fid, SOFT_BURST_RATIO).unwrap(), 0);
        let label: String = conn
            .query_row(
                "SELECT a.predicted_label FROM analysis a JOIN files f ON f.id = a.file_id
                  WHERE f.filename = 'b.cr3'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            label, "sharp",
            "stale burst scores must not rewrite verdicts"
        );
    }
}
