//! XMP sidecar ratings.
//!
//! Star ratings are the one piece of metadata every photo tool agrees on:
//! Lightroom, Bridge, darktable and ON1 all read/write `xmp:Rating` (0-5) in a
//! `<name>.xmp` sidecar next to the RAW. Writing it there means a rating set in
//! BirdSorter is visible in those tools (and vice versa) without ever touching
//! the RAW itself.
//!
//! Existing sidecars are **merged, never replaced**: only the `xmp:Rating`
//! attribute is changed, everything else (creator, EXIF, develop settings) is
//! preserved verbatim. A sidecar we cannot parse safely is left alone.

use std::path::{Path, PathBuf};

use crate::{CoreError, Result};

/// The sidecar path a RAW's rating belongs in.
pub fn sidecar_path(raw: &Path) -> PathBuf {
    raw.with_extension("xmp")
}

/// A minimal XMP document carrying just the rating.
fn fresh_document(rating: u8) -> String {
    format!(
        "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"BirdSorter\">\n\
         \x20<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\
         \x20 <rdf:Description rdf:about=\"\"\n\
         \x20  xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\"\n\
         \x20  xmp:Rating=\"{rating}\"/>\n\
         \x20</rdf:RDF>\n\
         </x:xmpmeta>\n"
    )
}

/// Replace the value of an `xmp:Rating="…"` attribute, if present.
fn replace_attribute(text: &str, rating: u8) -> Option<String> {
    let key = "xmp:Rating=\"";
    let start = text.find(key)? + key.len();
    let end = text[start..].find('"')? + start;
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..start]);
    out.push_str(&rating.to_string());
    out.push_str(&text[end..]);
    Some(out)
}

/// Replace an `<xmp:Rating>…</xmp:Rating>` element, if present.
fn replace_element(text: &str, rating: u8) -> Option<String> {
    let open = "<xmp:Rating>";
    let start = text.find(open)? + open.len();
    let end = text[start..].find("</xmp:Rating>")? + start;
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..start]);
    out.push_str(&rating.to_string());
    out.push_str(&text[end..]);
    Some(out)
}

/// Insert the attribute into the first `<rdf:Description …>` start tag.
fn insert_attribute(text: &str, rating: u8) -> Option<String> {
    let tag = "<rdf:Description";
    let start = text.find(tag)?;
    let gt = text[start..].find('>')? + start;
    let mut out = String::with_capacity(text.len() + 24);
    out.push_str(&text[..gt]);
    out.push_str(&format!("\n    xmp:Rating=\"{rating}\""));
    out.push_str(&text[gt..]);
    Some(out)
}

/// Merge `rating` into an existing XMP document, or `None` when it cannot be
/// edited safely.
fn merge(text: &str, rating: u8) -> Option<String> {
    replace_attribute(text, rating)
        .or_else(|| replace_element(text, rating))
        .or_else(|| insert_attribute(text, rating))
}

/// Write `rating` (0-5) into the RAW's XMP sidecar, merging when one exists.
pub fn write_rating(raw: &Path, rating: u8) -> Result<()> {
    let rating = rating.min(5);
    let sidecar = sidecar_path(raw);

    if sidecar.is_file() {
        let text = std::fs::read_to_string(&sidecar).map_err(|e| CoreError::io(&sidecar, e))?;
        let Some(merged) = merge(&text, rating) else {
            return Err(CoreError::Other(format!(
                "{} has an XMP sidecar without an rdf:Description, refusing to overwrite it",
                raw.display()
            )));
        };
        if merged == text {
            return Ok(());
        }
        std::fs::write(&sidecar, merged).map_err(|e| CoreError::io(&sidecar, e))?;
    } else {
        if let Some(parent) = sidecar.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
        }
        std::fs::write(&sidecar, fresh_document(rating)).map_err(|e| CoreError::io(&sidecar, e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed-down copy of a real Lightroom/ON1 sidecar.
    const EXISTING: &str = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 7.0">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
   xmp:MetadataDate="2026-05-01T12:28:10+07:00"
   xmp:Rating="5"
   tiff:Make="Canon">
   <dc:creator>
    <rdf:Seq>
     <rdf:li>Michael G</rdf:li>
    </rdf:Seq>
   </dc:creator>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"#;

    #[test]
    fn replaces_an_existing_attribute_and_keeps_everything_else() {
        let merged = merge(EXISTING, 3).unwrap();
        assert!(merged.contains("xmp:Rating=\"3"));
        assert!(!merged.contains("xmp:Rating=\"5\""));
        // Untouched neighbours.
        assert!(merged.contains("xmp:MetadataDate=\"2026-05-01T12:28:10+07:00\""));
        assert!(merged.contains("<rdf:li>Michael G</rdf:li>"));
        assert!(merged.contains("tiff:Make=\"Canon\""));
    }

    #[test]
    fn inserts_when_there_is_no_rating_yet() {
        let no_rating = EXISTING.replace("   xmp:Rating=\"5\"\n", "");
        let merged = merge(&no_rating, 4).unwrap();
        assert!(merged.contains("xmp:Rating=\"4\""));
        assert!(merged.contains("<rdf:li>Michael G</rdf:li>"));
    }

    #[test]
    fn handles_the_element_form() {
        let text = "<rdf:Description><xmp:Rating>2</xmp:Rating></rdf:Description>";
        assert_eq!(
            merge(text, 5).unwrap(),
            "<rdf:Description><xmp:Rating>5</xmp:Rating></rdf:Description>"
        );
    }

    #[test]
    fn refuses_to_edit_an_unrecognisable_document() {
        assert!(merge("not really xmp", 3).is_none());
    }

    #[test]
    fn writes_a_fresh_sidecar_when_none_exists() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("_5260921.CR3");
        std::fs::write(&raw, b"raw").unwrap();
        write_rating(&raw, 5).unwrap();
        let sidecar = sidecar_path(&raw);
        assert_eq!(sidecar.file_name().unwrap(), "_5260921.xmp");
        let text = std::fs::read_to_string(&sidecar).unwrap();
        assert!(text.contains("xmp:Rating=\"5\""));
        // Re-rating merges rather than duplicating.
        write_rating(&raw, 2).unwrap();
        let text = std::fs::read_to_string(&sidecar).unwrap();
        assert!(text.contains("xmp:Rating=\"2\""));
        assert!(!text.contains("xmp:Rating=\"5\""));
    }
}
