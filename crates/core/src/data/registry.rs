//! Model registry: the downloadable ONNX artifacts behind the T1/T2 tiers.
//!
//! The installer ships no models. Each entry here describes one artifact that
//! can be fetched on demand, verified by SHA-256 and removed again. Downloads
//! go through the [`Fetcher`] trait so the logic is testable without a network.
//!
//! Where the artifacts are actually hosted is still an open decision (GitHub
//! Releases vs HuggingFace); the built-in specs use empty URLs until that is
//! settled, and [`download`] reports a clear error if asked to fetch one.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{CoreError, Result};

/// Feature tier a model unlocks (see `plan.md` §3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Works with no download at all.
    T0,
    /// Adds subject detection.
    T1,
    /// Adds species identification.
    T2,
}

/// One downloadable model artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub key: String,
    pub display_name: String,
    pub description: String,
    /// File name inside the model directory.
    pub filename: String,
    pub url: String,
    /// Lower-case hex SHA-256 of the file. Empty means "not pinned yet".
    pub sha256: String,
    pub size_bytes: u64,
    pub tier: Tier,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    Missing,
    Ready,
    /// Present but its hash does not match the spec.
    Corrupt,
}

/// Built-in model catalogue. URLs/hashes are placeholders until hosting is
/// decided, so these report [`ModelStatus::Missing`] and cannot be downloaded.
pub fn builtin_models() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            key: "bird_detector".into(),
            display_name: "Bird detector (YOLO11n)".into(),
            description: "Single-class bird detector, int8 quantized ONNX (~6 MB).".into(),
            filename: "bird_det_v1.onnx".into(),
            url: String::new(),
            sha256: String::new(),
            size_bytes: 6_000_000,
            tier: Tier::T1,
            tags: vec!["detector".into(), "birds".into()],
        },
        ModelSpec {
            key: "bird_species".into(),
            display_name: "Bird species (iNat21 Aves)".into(),
            description: "Distilled bird-only classifier, int8 ONNX (~25 MB).".into(),
            filename: "bird_species_v1.onnx".into(),
            url: String::new(),
            sha256: String::new(),
            size_bytes: 25_000_000,
            tier: Tier::T2,
            tags: vec!["species".into(), "birds".into(), "inat21".into()],
        },
    ]
}

pub fn spec_by_key(key: &str) -> Option<ModelSpec> {
    builtin_models().into_iter().find(|m| m.key == key)
}

/// Hex SHA-256 of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| CoreError::io(path, e))?;
    Ok(sha256_hex(&bytes))
}

pub fn model_path(dir: &Path, spec: &ModelSpec) -> PathBuf {
    dir.join(&spec.filename)
}

/// Inspect a model directory and report the artifact's status.
pub fn status(dir: &Path, spec: &ModelSpec) -> ModelStatus {
    let path = model_path(dir, spec);
    if !path.is_file() {
        return ModelStatus::Missing;
    }
    if spec.sha256.is_empty() {
        // Nothing to verify against.
        return ModelStatus::Ready;
    }
    match sha256_file(&path) {
        Ok(h) if h.eq_ignore_ascii_case(&spec.sha256) => ModelStatus::Ready,
        _ => ModelStatus::Corrupt,
    }
}

/// Anything that can fetch a URL's bytes.
pub trait Fetcher: Send + Sync {
    fn fetch(&self, url: &str) -> Result<Vec<u8>>;
}

/// Blocking HTTP fetcher.
#[derive(Debug, Default, Clone, Copy)]
pub struct HttpFetcher;

impl Fetcher for HttpFetcher {
    fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        let mut response = ureq::get(url)
            .call()
            .map_err(|e| CoreError::Other(format!("download failed for {url}: {e}")))?;
        response
            .body_mut()
            .read_to_vec()
            .map_err(|e| CoreError::Other(format!("download failed for {url}: {e}")))
    }
}

/// Download, verify and atomically install a model. Returns the final path.
pub fn download(dir: &Path, spec: &ModelSpec, fetcher: &dyn Fetcher) -> Result<PathBuf> {
    if spec.url.is_empty() {
        return Err(CoreError::Other(format!(
            "model '{}' has no download URL yet (hosting not configured)",
            spec.key
        )));
    }
    let bytes = fetcher.fetch(&spec.url)?;
    if !spec.sha256.is_empty() {
        let actual = sha256_hex(&bytes);
        if !actual.eq_ignore_ascii_case(&spec.sha256) {
            return Err(CoreError::Other(format!(
                "checksum mismatch for '{}': expected {}, got {}",
                spec.key, spec.sha256, actual
            )));
        }
    }
    std::fs::create_dir_all(dir).map_err(|e| CoreError::io(dir, e))?;
    let final_path = model_path(dir, spec);
    let tmp = final_path.with_extension("onnx.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| CoreError::io(&tmp, e))?;
    std::fs::rename(&tmp, &final_path).map_err(|e| CoreError::io(&final_path, e))?;
    Ok(final_path)
}

/// Delete an installed model. Returns whether a file was removed.
pub fn remove(dir: &Path, spec: &ModelSpec) -> Result<bool> {
    let path = model_path(dir, spec);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(CoreError::io(path, e)),
    }
}

/// Default directory for downloaded models: `<user data>/BirdSorter/models`.
pub fn default_model_dir() -> Result<PathBuf> {
    let dirs = directories::ProjectDirs::from("", "", "BirdSorter")
        .ok_or_else(|| CoreError::Other("no platform user-data directory".into()))?;
    Ok(dirs.data_dir().join("models"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A fetcher that serves canned bytes and records the URLs requested.
    struct FakeFetcher {
        responses: HashMap<String, Vec<u8>>,
    }

    impl Fetcher for FakeFetcher {
        fn fetch(&self, url: &str) -> Result<Vec<u8>> {
            self.responses
                .get(url)
                .cloned()
                .ok_or_else(|| CoreError::Other(format!("no fake response for {url}")))
        }
    }

    fn spec_with(url: &str, payload: &[u8]) -> ModelSpec {
        ModelSpec {
            key: "test_model".into(),
            display_name: "Test".into(),
            description: "test".into(),
            filename: "test.onnx".into(),
            url: url.into(),
            sha256: sha256_hex(payload),
            size_bytes: payload.len() as u64,
            tier: Tier::T1,
            tags: vec![],
        }
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn status_reports_missing_ready_and_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"model-bytes";
        let spec = spec_with("https://example/x.onnx", payload);

        assert_eq!(status(dir.path(), &spec), ModelStatus::Missing);

        std::fs::write(model_path(dir.path(), &spec), payload).unwrap();
        assert_eq!(status(dir.path(), &spec), ModelStatus::Ready);

        std::fs::write(model_path(dir.path(), &spec), b"tampered").unwrap();
        assert_eq!(status(dir.path(), &spec), ModelStatus::Corrupt);
    }

    #[test]
    fn download_installs_and_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"onnx-weights";
        let spec = spec_with("https://example/m.onnx", payload);
        let fetcher = FakeFetcher {
            responses: HashMap::from([(spec.url.clone(), payload.to_vec())]),
        };

        let path = download(dir.path(), &spec, &fetcher).unwrap();
        assert!(path.exists());
        assert_eq!(status(dir.path(), &spec), ModelStatus::Ready);
        assert_eq!(std::fs::read(&path).unwrap(), payload);

        // No temp files left behind.
        let count = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn download_rejects_a_checksum_mismatch_and_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut spec = spec_with("https://example/m.onnx", b"good");
        spec.sha256 = sha256_hex(b"something else");
        let fetcher = FakeFetcher {
            responses: HashMap::from([(spec.url.clone(), b"good".to_vec())]),
        };

        let err = download(dir.path(), &spec, &fetcher).unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"));
        assert_eq!(status(dir.path(), &spec), ModelStatus::Missing);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn download_without_a_url_is_a_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        let spec = spec_with("", b"x");
        let err = download(
            dir.path(),
            &spec,
            &FakeFetcher {
                responses: HashMap::new(),
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("no download URL"));
    }

    #[test]
    fn remove_deletes_the_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"bytes";
        let spec = spec_with("https://example/m.onnx", payload);
        std::fs::write(model_path(dir.path(), &spec), payload).unwrap();

        assert!(remove(dir.path(), &spec).unwrap());
        assert_eq!(status(dir.path(), &spec), ModelStatus::Missing);
        assert!(!remove(dir.path(), &spec).unwrap()); // already gone
    }

    #[test]
    fn builtin_models_are_unique_and_tiered() {
        let models = builtin_models();
        assert!(models.len() >= 2);
        let mut keys: Vec<&str> = models.iter().map(|m| m.key.as_str()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), models.len(), "duplicate model key");
        assert!(models.iter().any(|m| m.tier == Tier::T1));
        assert!(models.iter().any(|m| m.tier == Tier::T2));
        assert!(spec_by_key("bird_detector").is_some());
        assert!(spec_by_key("nope").is_none());
    }
}
