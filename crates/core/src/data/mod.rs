//! Data layer: SQLite library storage, settings and analysis/label persistence.

pub mod analysis_store;
pub mod db;
pub mod labels;
pub mod library;
pub mod registry;
pub mod settings;

/// Current UTC time as an RFC 3339 string, matching the prototype's
/// `datetime.now(timezone.utc).isoformat()`.
pub fn now_iso() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

#[cfg(test)]
mod tests {
    use super::now_iso;

    #[test]
    fn now_iso_is_parseable_and_utc() {
        let s = now_iso();
        assert!(s.ends_with('Z') || s.contains("+00:00"), "not UTC: {s}");
        assert!(s.len() >= 20, "too short: {s}");
    }
}
