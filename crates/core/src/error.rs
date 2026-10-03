//! Crate-wide error type.

use std::path::PathBuf;

use thiserror::Error;

/// Errors produced by the core pipeline.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("raw decode failed for {path}: {message}")]
    RawDecode { path: PathBuf, message: String },

    /// A model file could not be loaded (missing, truncated, wrong opset).
    #[error("could not load model {path}: {message}")]
    ModelLoad { path: PathBuf, message: String },

    /// A model loaded but failed to run (a provider/runtime error, a shape
    /// mismatch, …). `model` is the model's file name or role.
    #[error("model '{model}' failed: {message}")]
    Inference { model: String, message: String },

    /// Invalid or unavailable configuration (a bad setting, no platform dirs,
    /// a database newer than this build).
    #[error("configuration error: {message}")]
    Config { message: String },

    /// Invalid user input or a malformed source that cannot be acted on.
    #[error("invalid input: {message}")]
    InvalidInput { message: String },

    /// A referenced record or file does not exist.
    #[error("not found: {message}")]
    NotFound { message: String },

    /// A model download failed (network, timeout, size limit).
    #[error("download failed for {url}: {message}")]
    Download { url: String, message: String },

    /// A model archive/bundle is malformed or violates its spec (bad zip, wrong
    /// or duplicate entries, oversized, checksum mismatch).
    #[error("model '{key}': {message}")]
    ModelArchive { key: String, message: String },

    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// A typed variant not yet warranted — a genuine catch-all, not a dumping
    /// ground. Prefer `ModelLoad`/`Inference`/`Config`/`RawDecode` where they fit.
    #[error("{0}")]
    Other(String),
}

impl CoreError {
    /// Build an [`CoreError::Io`] carrying the offending path.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        CoreError::Io {
            path: path.into(),
            source,
        }
    }

    /// Build a [`CoreError::RawDecode`] carrying the offending path.
    pub fn raw_decode(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        CoreError::RawDecode {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Build a [`CoreError::ModelLoad`].
    pub fn model_load(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        CoreError::ModelLoad {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Build a [`CoreError::Inference`].
    pub fn inference(model: impl Into<String>, message: impl Into<String>) -> Self {
        CoreError::Inference {
            model: model.into(),
            message: message.into(),
        }
    }

    /// Build a [`CoreError::Config`].
    pub fn config(message: impl Into<String>) -> Self {
        CoreError::Config {
            message: message.into(),
        }
    }

    /// Build a [`CoreError::InvalidInput`].
    pub fn invalid_input(message: impl Into<String>) -> Self {
        CoreError::InvalidInput {
            message: message.into(),
        }
    }

    /// Build a [`CoreError::NotFound`].
    pub fn not_found(message: impl Into<String>) -> Self {
        CoreError::NotFound {
            message: message.into(),
        }
    }

    /// Build a [`CoreError::Download`].
    pub fn download(url: impl Into<String>, message: impl Into<String>) -> Self {
        CoreError::Download {
            url: url.into(),
            message: message.into(),
        }
    }

    /// Build a [`CoreError::ModelArchive`].
    pub fn model_archive(key: impl Into<String>, message: impl Into<String>) -> Self {
        CoreError::ModelArchive {
            key: key.into(),
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_variants_carry_their_context() {
        let e = CoreError::model_load("bird_eye.onnx", "truncated");
        assert!(matches!(e, CoreError::ModelLoad { .. }));
        assert!(e.to_string().contains("bird_eye.onnx"));
        assert!(e.to_string().contains("truncated"));

        let e = CoreError::inference("yolo11n.onnx", "DirectML reshape failed");
        assert!(matches!(e, CoreError::Inference { .. }));
        assert!(e.to_string().contains("yolo11n.onnx"));
        assert!(e.to_string().contains("DirectML"));

        assert!(matches!(
            CoreError::config("no platform dir"),
            CoreError::Config { .. }
        ));

        let e = CoreError::download("https://example/m.onnx", "timeout");
        assert!(matches!(e, CoreError::Download { .. }));
        assert!(e.to_string().contains("https://example/m.onnx"));
        assert!(e.to_string().contains("timeout"));

        let e = CoreError::model_archive("bird_species", "duplicate file 'a.onnx'");
        assert!(matches!(e, CoreError::ModelArchive { .. }));
        assert!(e.to_string().contains("bird_species"));
        assert!(e.to_string().contains("duplicate file"));

        let e = CoreError::invalid_input("not a directory: D:/x");
        assert!(matches!(e, CoreError::InvalidInput { .. }));
        assert!(e.to_string().contains("not a directory"));

        let e = CoreError::not_found("unknown folder_id 7");
        assert!(matches!(e, CoreError::NotFound { .. }));
        assert!(e.to_string().contains("folder_id 7"));
    }
}
