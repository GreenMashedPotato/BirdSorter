//! Content identity hashing.
//!
//! Files are identified by an xxh3-64 digest of their bytes, so renames and
//! moves never lose analysis data. Streaming, so multi-hundred-MB RAWs do not
//! have to be held in memory.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::{CoreError, Result};

const CHUNK: usize = 1 << 20; // 1 MiB

/// Hex digest of the xxh3-64 hash of a byte slice.
pub fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:016x}", xxhash_rust::xxh3::xxh3_64(bytes))
}

/// Hex digest of the xxh3-64 hash of a file's contents.
pub fn hash_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| CoreError::io(path, e))?;
    let mut reader = BufReader::with_capacity(CHUNK, file);
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = reader.read(&mut buf).map_err(|e| CoreError::io(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:016x}", hasher.digest()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn hash_is_stable_and_content_sensitive() {
        let a = hash_bytes(b"hello world");
        assert_eq!(a, hash_bytes(b"hello world"));
        assert_ne!(a, hash_bytes(b"hello worlD"));
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn file_hash_matches_bytes_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.bin");
        let payload = vec![7u8; 3 * (1 << 20) + 123];
        std::fs::write(&path, &payload).unwrap();
        assert_eq!(hash_file(&path).unwrap(), hash_bytes(&payload));
    }

    #[test]
    fn file_hash_is_rename_proof() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.cr3");
        let b = dir.path().join("renamed.cr3");
        let mut f = File::create(&a).unwrap();
        f.write_all(b"raw bytes").unwrap();
        drop(f);
        std::fs::rename(&a, &b).unwrap();
        assert_eq!(hash_file(&b).unwrap(), hash_bytes(b"raw bytes"));
    }

    #[test]
    fn missing_file_is_a_typed_error() {
        let err = hash_file("does/not/exist.cr3").unwrap_err();
        assert!(matches!(err, CoreError::Io { .. }));
    }
}
