//! Model registry: the downloadable ONNX artifacts behind the T1/T2 tiers.
//!
//! The installer ships no models. Each entry describes one artifact that can be
//! fetched from a URL **or imported from a local file**, verified by SHA-256 and
//! removed again. A multi-file model (e.g. species = onnx + labels + meta) is
//! shipped as a single `.zip` bundle so download and import are one action.
//!
//! Download goes through the [`Fetcher`] trait so the logic is testable without
//! a network.
//!
//! Hosting is still an open decision (GitHub Releases vs HuggingFace); the
//! built-in specs use empty URLs until that is settled, and [`download`] reports
//! a clear error if asked to fetch one. **Local import always works.**

use std::io::Cursor;
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
    /// Downloaded artifact file name (a `.zip` when `unpack`, else the model file).
    pub artifact: String,
    /// Whether the artifact is a zip to extract into the model directory.
    pub unpack: bool,
    /// Files expected in the model directory once installed.
    pub files: Vec<String>,
    pub url: String,
    /// Lower-case hex SHA-256 of the artifact. Empty means "not pinned yet".
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
/// decided, so these report [`ModelStatus::Missing`] and cannot be downloaded —
/// but they can always be **imported from a local file**.
pub fn builtin_models() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            key: "bird_detector".into(),
            display_name: "Bird detector (YOLO11n)".into(),
            description: "Single-class bird detector, ONNX (~11 MB). Bundled with the app, so \
                          no download is needed; install from file to override."
                .into(),
            artifact: "bird_det_v1.onnx".into(),
            unpack: false,
            files: vec!["bird_det_v1.onnx".into()],
            url: String::new(),
            sha256: String::new(),
            size_bytes: 10_700_000,
            tier: Tier::T1,
            tags: vec!["detector".into(), "birds".into()],
        },
        ModelSpec {
            key: "bird_species".into(),
            display_name: "Bird species (ViT, 526 classes, int8)".into(),
            description: "ViT-B/16 fine-tuned on 526 bird species, int8 (~85 MB). \
                          Shipped as a zip: model + labels + preprocessing meta."
                .into(),
            artifact: "bird_species.zip".into(),
            unpack: true,
            files: vec![
                "bird_species.onnx".into(),
                "bird_species.labels.json".into(),
                "bird_species.meta.json".into(),
            ],
            url: String::new(),
            sha256: String::new(),
            size_bytes: 85_000_000,
            tier: Tier::T2,
            tags: vec![
                "species".into(),
                "birds".into(),
                "vit".into(),
                "int8".into(),
            ],
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
    dir.join(&spec.artifact)
}

/// Paths of the files this model installs.
pub fn installed_paths(dir: &Path, spec: &ModelSpec) -> Vec<PathBuf> {
    spec.files.iter().map(|f| dir.join(f)).collect()
}

/// Inspect the model directory and report the artifact's status.
pub fn status(dir: &Path, spec: &ModelSpec) -> ModelStatus {
    if spec.unpack {
        if spec.files.iter().all(|f| dir.join(f).is_file()) {
            ModelStatus::Ready
        } else {
            ModelStatus::Missing
        }
    } else {
        let path = model_path(dir, spec);
        if !path.is_file() {
            return ModelStatus::Missing;
        }
        if spec.sha256.is_empty() {
            return ModelStatus::Ready;
        }
        match sha256_file(&path) {
            Ok(h) if h.eq_ignore_ascii_case(&spec.sha256) => ModelStatus::Ready,
            _ => ModelStatus::Corrupt,
        }
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

fn verify(spec: &ModelSpec, bytes: &[u8]) -> Result<()> {
    if spec.sha256.is_empty() {
        return Ok(());
    }
    let actual = sha256_hex(bytes);
    if actual.eq_ignore_ascii_case(&spec.sha256) {
        Ok(())
    } else {
        Err(CoreError::Other(format!(
            "checksum mismatch for '{}': expected {}, got {}",
            spec.key, spec.sha256, actual
        )))
    }
}

/// Install already-fetched artifact bytes: verify, then unzip or write.
pub fn install_bytes(dir: &Path, spec: &ModelSpec, bytes: &[u8]) -> Result<Vec<PathBuf>> {
    verify(spec, bytes)?;
    std::fs::create_dir_all(dir).map_err(|e| CoreError::io(dir, e))?;

    if spec.unpack {
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| CoreError::Other(format!("bad zip for '{}': {e}", spec.key)))?;
        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| CoreError::Other(format!("bad zip entry: {e}")))?;
            // Flatten paths: only the file name is used, so a zip cannot escape.
            let Some(name) = Path::new(file.name()).file_name() else {
                continue;
            };
            if file.is_dir() {
                continue;
            }
            let dest = dir.join(name);
            let mut out = std::fs::File::create(&dest).map_err(|e| CoreError::io(&dest, e))?;
            std::io::copy(&mut file, &mut out).map_err(|e| CoreError::io(&dest, e))?;
        }
    } else {
        let final_path = model_path(dir, spec);
        let tmp = final_path.with_extension("part");
        std::fs::write(&tmp, bytes).map_err(|e| CoreError::io(&tmp, e))?;
        std::fs::rename(&tmp, &final_path).map_err(|e| CoreError::io(&final_path, e))?;
    }
    Ok(installed_paths(dir, spec))
}

/// Download and install a model from its configured URL.
pub fn download(dir: &Path, spec: &ModelSpec, fetcher: &dyn Fetcher) -> Result<Vec<PathBuf>> {
    if spec.url.is_empty() {
        return Err(CoreError::Other(format!(
            "model '{}' has no download URL yet — use \"Install from file…\" instead",
            spec.key
        )));
    }
    let bytes = fetcher.fetch(&spec.url)?;
    install_bytes(dir, spec, &bytes)
}

/// Install a model from a local file (a `.zip` bundle or a single `.onnx`).
pub fn import_file(dir: &Path, spec: &ModelSpec, source: &Path) -> Result<Vec<PathBuf>> {
    let bytes = std::fs::read(source).map_err(|e| CoreError::io(source, e))?;
    install_bytes(dir, spec, &bytes)
}

/// Delete every file this model installed. Returns how many were removed.
pub fn remove(dir: &Path, spec: &ModelSpec) -> Result<usize> {
    let mut removed = 0;
    // The artifact itself may still be present for single-file models.
    let mut candidates: Vec<PathBuf> = spec.files.iter().map(|f| dir.join(f)).collect();
    candidates.push(model_path(dir, spec));
    for path in candidates {
        match std::fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(CoreError::io(path, e)),
        }
    }
    Ok(removed)
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

    fn single_file_spec(url: &str, payload: &[u8]) -> ModelSpec {
        ModelSpec {
            key: "test_single".into(),
            display_name: "Test".into(),
            description: "test".into(),
            artifact: "test.onnx".into(),
            unpack: false,
            files: vec!["test.onnx".into()],
            url: url.into(),
            sha256: sha256_hex(payload),
            size_bytes: payload.len() as u64,
            tier: Tier::T1,
            tags: vec![],
        }
    }

    fn zip_bundle(spec_name: &str, entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut writer = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for (name, data) in entries {
                writer.start_file(*name, opts).unwrap();
                use std::io::Write;
                writer.write_all(data).unwrap();
            }
            writer.finish().unwrap();
        }
        let _ = spec_name;
        buf
    }

    fn bundle_spec(url: &str, payload: &[u8]) -> ModelSpec {
        ModelSpec {
            key: "test_bundle".into(),
            display_name: "Test bundle".into(),
            description: "test".into(),
            artifact: "bundle.zip".into(),
            unpack: true,
            files: vec!["a.onnx".into(), "a.labels.json".into()],
            url: url.into(),
            sha256: sha256_hex(payload),
            size_bytes: payload.len() as u64,
            tier: Tier::T2,
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
    fn single_file_download_installs_and_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"onnx-weights";
        let spec = single_file_spec("https://example/m.onnx", payload);
        let fetcher = FakeFetcher {
            responses: HashMap::from([(spec.url.clone(), payload.to_vec())]),
        };

        assert_eq!(status(dir.path(), &spec), ModelStatus::Missing);
        let paths = download(dir.path(), &spec, &fetcher).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(status(dir.path(), &spec), ModelStatus::Ready);
        assert_eq!(std::fs::read(&paths[0]).unwrap(), payload);
    }

    #[test]
    fn checksum_mismatch_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut spec = single_file_spec("https://example/m.onnx", b"good");
        spec.sha256 = sha256_hex(b"something else");
        let err = install_bytes(dir.path(), &spec, b"good").unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"));
        assert_eq!(status(dir.path(), &spec), ModelStatus::Missing);
    }

    #[test]
    fn zip_bundle_installs_all_files() {
        let dir = tempfile::tempdir().unwrap();
        let zip = zip_bundle(
            "bundle",
            &[("a.onnx", b"model"), ("a.labels.json", b"[\"x\"]")],
        );
        let spec = bundle_spec("https://example/b.zip", &zip);

        assert_eq!(status(dir.path(), &spec), ModelStatus::Missing);
        let paths = install_bytes(dir.path(), &spec, &zip).unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(status(dir.path(), &spec), ModelStatus::Ready);
        assert_eq!(std::fs::read(dir.path().join("a.onnx")).unwrap(), b"model");
    }

    #[test]
    fn import_from_a_local_file_works_without_a_url() {
        let dir = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();
        let zip = zip_bundle("b", &[("a.onnx", b"m"), ("a.labels.json", b"[]")]);
        let zip_path = src.path().join("bundle.zip");
        std::fs::write(&zip_path, &zip).unwrap();

        let spec = bundle_spec("", &zip);
        assert!(spec.url.is_empty());
        import_file(dir.path(), &spec, &zip_path).unwrap();
        assert_eq!(status(dir.path(), &spec), ModelStatus::Ready);
    }

    #[test]
    fn download_without_a_url_points_at_local_import() {
        let dir = tempfile::tempdir().unwrap();
        let spec = single_file_spec("", b"x");
        let err = download(
            dir.path(),
            &spec,
            &FakeFetcher {
                responses: HashMap::new(),
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("Install from file"));
    }

    #[test]
    fn remove_deletes_installed_files() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"bytes";
        let spec = single_file_spec("https://example/m.onnx", payload);
        install_bytes(dir.path(), &spec, payload).unwrap();
        assert_eq!(status(dir.path(), &spec), ModelStatus::Ready);

        assert_eq!(remove(dir.path(), &spec).unwrap(), 1);
        assert_eq!(status(dir.path(), &spec), ModelStatus::Missing);
        assert_eq!(remove(dir.path(), &spec).unwrap(), 0); // already gone
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
