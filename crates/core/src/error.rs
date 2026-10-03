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

    /// Invalid or unavailable configuration (a bad setting, no platform dirs).
    #[error("configuration error: {message}")]
    Config { message: String },

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
    }
}
