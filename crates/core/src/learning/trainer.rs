//! Active-learning trainer.
//!
//! A small multinomial logistic regression over the 12 sharpness features,
//! trained in-process on the user's labels. Deliberately dependency-free (pure
//! Rust) so the app stays light: the trained model is just weights + bias
//! serialized to JSON.
//!
//! Feature standardization and k-fold cross-validation are included, so the
//! reported accuracy is honest.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::analysis::scorer::features_to_vector;
use crate::analysis::sharpness::SharpnessFeatures;
use crate::data::{labels, models, now_iso};
use crate::{CoreError, Result};

/// Minimum labels per class before training is allowed.
pub const MIN_PER_CLASS: usize = 5;
pub const N_FOLDS: usize = 5;
const ITERS: usize = 400;
const LR: f32 = 0.5;
const L2: f32 = 1e-3;
const N_FEATURES: usize = 12;

/// A trained linear classifier (softmax over standardized features).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearModel {
    pub version: String,
    pub classes: Vec<String>,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// `classes x features` weights.
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl LinearModel {
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
        }
        std::fs::write(path, serde_json::to_string(self)?).map_err(|e| CoreError::io(path, e))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| CoreError::io(path, e))?;
        Ok(serde_json::from_str(&text)?)
    }

    fn probabilities(&self, x: &[f32; N_FEATURES]) -> Vec<f32> {
        let mut z = vec![0f32; self.weights.len()];
        for (k, row) in self.weights.iter().enumerate() {
            let mut s = self.bias.get(k).copied().unwrap_or(0.0);
            for j in 0..N_FEATURES {
                let xs = (x[j] - self.mean[j]) / self.std[j].max(1e-6);
                s += row[j] * xs;
            }
            z[k] = s;
        }
        softmax(&z)
    }

    /// `(class_index, probabilities)`.
    pub fn predict(&self, x: &[f32; N_FEATURES]) -> (usize, Vec<f32>) {
        let p = self.probabilities(x);
        let idx = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (idx, p)
    }

    pub fn class_index(&self, name: &str) -> Option<usize> {
        self.classes.iter().position(|c| c == name)
    }
}

fn softmax(z: &[f32]) -> Vec<f32> {
    if z.is_empty() {
        return Vec::new();
    }
    let max = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|v| v / sum).collect()
    } else {
        vec![1.0 / z.len() as f32; z.len()]
    }
}

fn standardize_stats(xs: &[[f32; N_FEATURES]]) -> (Vec<f32>, Vec<f32>) {
    let n = xs.len().max(1) as f32;
    let mut mean = vec![0f32; N_FEATURES];
    for x in xs {
        for j in 0..N_FEATURES {
            mean[j] += x[j];
        }
    }
    for m in mean.iter_mut() {
        *m /= n;
    }
    let mut var = [0f32; N_FEATURES];
    for x in xs {
        for j in 0..N_FEATURES {
            let d = x[j] - mean[j];
            var[j] += d * d;
        }
    }
    let std = var.iter().map(|v| (v / n).sqrt().max(1e-6)).collect();
    (mean, std)
}

fn fit(
    xs: &[[f32; N_FEATURES]],
    y: &[usize],
    n_classes: usize,
    mean: &[f32],
    std: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let n = xs.len().max(1) as f32;
    let mut w = vec![vec![0f32; N_FEATURES]; n_classes];
    let mut b = vec![0f32; n_classes];

    let zx: Vec<Vec<f32>> = xs
        .iter()
        .map(|x| {
            (0..N_FEATURES)
                .map(|j| (x[j] - mean[j]) / std[j].max(1e-6))
                .collect()
        })
        .collect();

    for _ in 0..ITERS {
        let mut gw = vec![vec![0f32; N_FEATURES]; n_classes];
        let mut gb = vec![0f32; n_classes];
        for i in 0..xs.len() {
            let mut z = vec![0f32; n_classes];
            for k in 0..n_classes {
                let mut s = b[k];
                for j in 0..N_FEATURES {
                    s += w[k][j] * zx[i][j];
                }
                z[k] = s;
            }
            let p = softmax(&z);
            for k in 0..n_classes {
                let err = p[k] - if y[i] == k { 1.0 } else { 0.0 };
                for j in 0..N_FEATURES {
                    gw[k][j] += err * zx[i][j];
                }
                gb[k] += err;
            }
        }
        for k in 0..n_classes {
            for j in 0..N_FEATURES {
                w[k][j] -= LR * (gw[k][j] / n + L2 * w[k][j]);
            }
            b[k] -= LR * (gb[k] / n);
        }
    }
    (w, b)
}

fn cross_val_accuracy(
    xs: &[[f32; N_FEATURES]],
    y: &[usize],
    n_classes: usize,
    folds: usize,
) -> f32 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let folds = folds.clamp(2, n);
    let mut correct = 0usize;
    let mut total = 0usize;

    for f in 0..folds {
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();
        for i in 0..n {
            if i % folds == f {
                test_x.push(xs[i]);
                test_y.push(y[i]);
            } else {
                train_x.push(xs[i]);
                train_y.push(y[i]);
            }
        }
        if train_x.is_empty() || test_x.is_empty() {
            continue;
        }
        if train_y.iter().collect::<BTreeSet<_>>().len() < 2 {
            continue;
        }
        let (mean, std) = standardize_stats(&train_x);
        let (w, b) = fit(&train_x, &train_y, n_classes, &mean, &std);
        let model = LinearModel {
            version: "cv".into(),
            classes: Vec::new(),
            mean,
            std,
            weights: w,
            bias: b,
        };
        for i in 0..test_x.len() {
            let (idx, _) = model.predict(&test_x[i]);
            if idx == test_y[i] {
                correct += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    }
}

/// Whether there are enough labels to train, with a human-readable reason.
pub fn can_train(conn: &Connection, min_per_class: usize) -> Result<(bool, String)> {
    let counts = labels::label_counts(conn)?;
    let ready = counts.values().filter(|c| **c >= min_per_class).count();
    if ready >= 2 {
        Ok((
            true,
            format!("{ready} label classes have at least {min_per_class} examples"),
        ))
    } else {
        Ok((
            false,
            format!("need at least 2 classes with {min_per_class}+ labels (currently {ready})"),
        ))
    }
}

/// Result of a training run.
#[derive(Debug, Clone)]
pub struct TrainResult {
    pub version: String,
    pub n_samples: usize,
    pub cv_accuracy: f32,
    pub classes: Vec<String>,
    pub path: PathBuf,
}

/// Train a model from the user's labels and record it.
pub fn train(conn: &Connection, models_dir: &Path) -> Result<TrainResult> {
    let labeled = labels::all_labeled_features(conn)?;
    if labeled.len() < 4 {
        return Err(CoreError::invalid_input(format!(
            "not enough labelled files ({}); label more first",
            labeled.len()
        )));
    }

    let mut xs: Vec<[f32; N_FEATURES]> = Vec::with_capacity(labeled.len());
    let mut ys: Vec<String> = Vec::with_capacity(labeled.len());
    for (json, label) in &labeled {
        let features = SharpnessFeatures::from_json(json)?;
        xs.push(features_to_vector(&features));
        ys.push(label.as_str().to_string());
    }

    let classes: Vec<String> = ys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if classes.len() < 2 {
        return Err(CoreError::invalid_input(
            "need at least 2 different label classes",
        ));
    }
    let y_idx: Vec<usize> = ys
        .iter()
        .map(|s| classes.iter().position(|c| c == s).unwrap())
        .collect();
    let n_classes = classes.len();

    let cv_accuracy = cross_val_accuracy(&xs, &y_idx, n_classes, N_FOLDS);
    let (mean, std) = standardize_stats(&xs);
    let (weights, bias) = fit(&xs, &y_idx, n_classes, &mean, &std);

    let version = format!("logreg-{}", now_iso().replace([':', '.', '-'], ""));
    let model = LinearModel {
        version: version.clone(),
        classes: classes.clone(),
        mean,
        std,
        weights,
        bias,
    };
    std::fs::create_dir_all(models_dir).map_err(|e| CoreError::io(models_dir, e))?;
    let path = models_dir.join(format!("{version}.json"));
    model.save(&path)?;

    models::insert_model(
        conn,
        &version,
        xs.len() as i64,
        Some(cv_accuracy),
        &path.to_string_lossy(),
    )?;

    Ok(TrainResult {
        version,
        n_samples: xs.len(),
        cv_accuracy,
        classes,
        path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::scorer::score_rules;
    use crate::data::analysis_store::{upsert_analysis, AnalysisRecord};
    use crate::data::db::{connect_in_memory, init_schema};
    use crate::data::labels::{add_label, LabelSource, UserLabel};
    use crate::data::library::{add_folder, list_files, scan_folder};
    use crate::data::settings::ScoreSettings;

    fn features(sharp: bool) -> SharpnessFeatures {
        if sharp {
            SharpnessFeatures {
                has_subject: true,
                tenengrad_subject: 0.9,
                laplacian_subject: 0.9,
                fft_high_freq_subject: 0.9,
                brenner_subject: 0.9,
                subject_edge_distance: 1.0,
                center_edge_ratio: 1.0,
                ..Default::default()
            }
        } else {
            SharpnessFeatures {
                has_subject: true,
                tenengrad_subject: 0.05,
                laplacian_subject: 0.05,
                fft_high_freq_subject: 0.05,
                brenner_subject: 0.05,
                subject_edge_distance: 1.0,
                center_edge_ratio: 1.0,
                ..Default::default()
            }
        }
    }

    /// Insert `n` analyzed + labelled files per class.
    fn seed(conn: &Connection, dir: &Path, sharp: usize, blurry: usize) {
        let folder = add_folder(conn, dir).unwrap();
        for i in 0..(sharp + blurry) {
            let name = format!("img{i}.cr3");
            std::fs::write(dir.join(&name), name.as_bytes()).unwrap();
        }
        scan_folder(conn, folder).unwrap();
        let files = list_files(conn, folder, false, false).unwrap();
        for (i, lf) in files.iter().enumerate() {
            let is_sharp = i < sharp;
            let f = features(is_sharp);
            upsert_analysis(
                conn,
                &AnalysisRecord {
                    file_id: lf.file.id,
                    score: score_rules(&f, &ScoreSettings::default()),
                    features: f,
                    ..Default::default()
                },
            )
            .unwrap();
            add_label(
                conn,
                lf.file.id,
                if is_sharp {
                    UserLabel::Sharp
                } else {
                    UserLabel::Blurry
                },
                LabelSource::Manual,
            )
            .unwrap();
        }
    }

    #[test]
    fn can_train_gates_on_class_counts() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        seed(&conn, dir.path(), 3, 3); // 3 per class < 5
        let (ok, why) = can_train(&conn, MIN_PER_CLASS).unwrap();
        assert!(!ok, "{why}");
    }

    #[test]
    fn trains_a_separable_problem_with_high_accuracy() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        seed(&conn, dir.path(), 12, 12);

        let (ok, _) = can_train(&conn, MIN_PER_CLASS).unwrap();
        assert!(ok);

        let models_dir = dir.path().join("models");
        let result = train(&conn, &models_dir).unwrap();
        assert_eq!(result.n_samples, 24);
        assert!(result.classes.contains(&"sharp".to_string()));
        assert!(
            result.cv_accuracy > 0.8,
            "cv accuracy too low: {}",
            result.cv_accuracy
        );
        assert!(result.path.is_file());

        // Recorded as the latest model.
        let latest = models::latest_model(&conn).unwrap().unwrap();
        assert_eq!(latest.version, result.version);

        // The saved model predicts the classes it was trained on.
        let model = LinearModel::load(&result.path).unwrap();
        let (sharp_idx, _) = model.predict(&features_to_vector(&features(true)));
        assert_eq!(model.classes[sharp_idx], "sharp");
        let (blurry_idx, _) = model.predict(&features_to_vector(&features(false)));
        assert_eq!(model.classes[blurry_idx], "blurry");
    }

    #[test]
    fn train_refuses_without_enough_labels() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let err = train(&conn, &dir.path().join("m")).unwrap_err();
        assert!(err.to_string().contains("not enough labelled"));
    }
}
