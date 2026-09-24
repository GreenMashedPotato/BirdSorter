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

    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

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
}

pub type Result<T> = std::result::Result<T, CoreError>;
