//! Move files to the OS recycle bin / trash.
//!
//! Deletion is deliberately "soft": the file goes to the OS trash (recoverable)
//! and the DB row keeps `trashed_at` set. Restoring from the trash and rescanning
//! re-links the file (see `library::reconcile_trashed`).

use std::path::Path;

use crate::{CoreError, Result};

/// Send a file to the OS recycle bin / trash.
pub fn move_to_trash(path: &Path) -> Result<()> {
    trash::delete(path)
        .map_err(|e| CoreError::io(path, std::io::Error::other(format!("trash failed: {e}"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_a_typed_error() {
        let err = move_to_trash(Path::new("does/not/exist.cr3"));
        assert!(matches!(err, Err(CoreError::Io { .. })));
    }
}
