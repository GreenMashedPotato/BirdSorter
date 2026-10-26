//! XMP sidecar ratings, labels and keywords.
//!
//! Star ratings are the one piece of metadata every photo tool agrees on:
//! Lightroom, Bridge, darktable and ON1 all read/write `xmp:Rating` (0-5) in a
//! `<name>.xmp` sidecar next to the RAW. Writing it there means a rating set in
//! BirdSorter is visible in those tools (and vice versa) without ever touching
//! the RAW itself.
//!
//! Existing sidecars are **merged, never replaced**: only the targeted
//! attribute/element is changed, everything else (creator, EXIF, develop
//! settings, other keywords) is preserved verbatim. A sidecar we cannot parse
//! safely is copied to `<name>.xmp.bak` and left untouched.
//!
//! Both common naming conventions are resolved: `shot.xmp` (stem) and
//! `shot.CR3.xmp` (full name). An existing sidecar is edited in place; a new
//! one is written next to the RAW using the stem form.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{CoreError, Result};

static WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const XMP_NS: &str = "http://ns.adobe.com/xap/1.0/";
const DC_NS: &str = "http://purl.org/dc/elements/1.1/";
/// Refuse to load an absurdly large sidecar into memory.
const MAX_SIDECAR_BYTES: u64 = 4 * 1024 * 1024;

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let seq = WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("xmp.tmp.{}.{}", std::process::id(), seq));
    std::fs::write(&tmp, bytes).map_err(|e| CoreError::io(&tmp, e))?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(CoreError::io(path, e));
    }
    Ok(())
}

/// The canonical (stem-style) sidecar path: `shot.CR3` → `shot.xmp`.
pub fn sidecar_path(raw: &Path) -> PathBuf {
    raw.with_extension("xmp")
}

/// The full-name variant: `shot.CR3` → `shot.CR3.xmp`.
fn full_sidecar_path(raw: &Path) -> PathBuf {
    let mut name = raw.as_os_str().to_os_string();
    name.push(".xmp");
    PathBuf::from(name)
}

/// True for the RAW types that use a sidecar file. JPEG/TIFF keep their
/// metadata in the image itself, and `shot.xmp` next to `shot.JPG` is the
/// sidecar of `shot.CR3` in Lightroom — writing it would edit the RAW's rating.
fn uses_sidecar(raw: &Path) -> bool {
    raw.extension()
        .and_then(|e| e.to_str())
        .map(|ext| {
            let lower = ext.to_ascii_lowercase();
            crate::data::library::RAW_EXTENSIONS.contains(&lower.as_str())
        })
        .unwrap_or(false)
}

/// An existing sidecar for `raw`, preferring the stem form when both exist.
pub fn existing_sidecar(raw: &Path) -> Option<PathBuf> {
    if !uses_sidecar(raw) {
        return None;
    }
    let stem = sidecar_path(raw);
    if stem.is_file() {
        return Some(stem);
    }
    let full = full_sidecar_path(raw);
    full.is_file().then_some(full)
}

/// Where to write: an existing sidecar, else the canonical stem path. `None`
/// when this file type has no sidecar convention (see [`uses_sidecar`]).
fn sidecar_target(raw: &Path) -> Option<PathBuf> {
    if !uses_sidecar(raw) {
        return None;
    }
    Some(existing_sidecar(raw).unwrap_or_else(|| sidecar_path(raw)))
}

fn read_sidecar(path: &Path) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    if metadata.len() > MAX_SIDECAR_BYTES {
        return None;
    }
    std::fs::read_to_string(path).ok()
}

/// Copy an unreadable sidecar aside before refusing to edit it.
///
/// One stable name: retrying the action (or rating a whole folder) must not
/// grow an unbounded pile of `.bak` files for the same unreadable sidecar.
fn backup_unreadable(sidecar: &Path) -> Option<PathBuf> {
    let name = sidecar.file_name()?.to_string_lossy();
    let backup = sidecar.with_file_name(format!("{name}.bak"));
    std::fs::copy(sidecar, &backup).ok()?;
    Some(backup)
}

fn unreadable_error(sidecar: &Path) -> CoreError {
    let suffix = backup_unreadable(sidecar)
        .map(|backup| format!(" (copied to {})", backup.display()))
        .unwrap_or_default();
    CoreError::invalid_input(format!(
        "{} has an XMP sidecar that cannot be edited safely, refusing to overwrite it{suffix}",
        sidecar.display()
    ))
}

fn escape_attribute(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn escape_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn unescape_xml(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn keywords_block(keywords: &[String]) -> String {
    let mut block = String::from("<dc:subject>\n    <rdf:Bag>");
    for keyword in keywords {
        block.push_str(&format!("\n     <rdf:li>{}</rdf:li>", escape_text(keyword)));
    }
    block.push_str("\n    </rdf:Bag>\n   </dc:subject>");
    block
}

/// A minimal XMP document carrying the requested attributes and keywords.
fn fresh_document(attributes: &[(&str, &str)], keywords: &[String]) -> String {
    let mut attrs = String::new();
    for (key, value) in attributes {
        attrs.push_str(&format!("\n    {key}=\"{}\"", escape_attribute(value)));
    }
    let subject = if keywords.is_empty() {
        String::new()
    } else {
        format!("\n   {}", keywords_block(keywords))
    };
    format!(
        "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"BirdSorter\">\n\
         \x20<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\
         \x20 <rdf:Description rdf:about=\"\"\n\
         \x20  xmlns:xmp=\"{XMP_NS}\"\n\
         \x20  xmlns:dc=\"{DC_NS}\"{attrs}>{subject}\n\
         \x20 </rdf:Description>\n\
         \x20</rdf:RDF>\n\
         </x:xmpmeta>\n"
    )
}

/// The first `<rdf:Description …>` tag: `(start, '>' index, self-closing)`.
fn description_tag(text: &str) -> Option<(usize, usize, bool)> {
    let start = text.find("<rdf:Description")?;
    let gt = text[start..].find('>')? + start;
    let mut at = gt;
    while at > start && text.as_bytes()[at - 1].is_ascii_whitespace() {
        at -= 1;
    }
    let self_closing = at > start && text.as_bytes()[at - 1] == b'/';
    Some((start, gt, self_closing))
}

/// Position just before `>` (or before the `/` of a self-closing tag).
fn insertion_point(text: &str, tag_start: usize, gt: usize, self_closing: bool) -> usize {
    if self_closing {
        let mut at = gt;
        while at > tag_start && text.as_bytes()[at - 1].is_ascii_whitespace() {
            at -= 1;
        }
        at - 1
    } else {
        gt
    }
}

fn ensure_namespace(text: &str, prefix: &str, uri: &str) -> Option<String> {
    if text.contains(&format!("xmlns:{prefix}=")) {
        return Some(text.to_string());
    }
    let (start, gt, self_closing) = description_tag(text)?;
    let at = insertion_point(text, start, gt, self_closing);
    let mut out = String::with_capacity(text.len() + uri.len() + prefix.len() + 16);
    out.push_str(&text[..at]);
    out.push_str(&format!(" xmlns:{prefix}=\"{uri}\""));
    out.push_str(&text[at..]);
    Some(out)
}

/// `(attribute start, value start, value end, quote)` for `key="value"`.
fn find_attribute(text: &str, key: &str) -> Option<(usize, usize, usize, char)> {
    let mut search = 0;
    while let Some(rel) = text[search..].find(key) {
        let key_start = search + rel;
        let after = key_start + key.len();
        let rest = &text[after..];
        let ws1 = rest.len() - rest.trim_start().len();
        let eq = after + ws1;
        if text[eq..].starts_with('=') {
            let after_eq = eq + 1;
            let ws2 = text[after_eq..].len() - text[after_eq..].trim_start().len();
            let value_start = after_eq + ws2;
            if let Some(&byte) = text.as_bytes().get(value_start) {
                let quote = byte as char;
                if quote == '"' || quote == '\'' {
                    if let Some(end_rel) = text[value_start + 1..].find(quote) {
                        let value_end = value_start + 1 + end_rel;
                        return Some((key_start, value_start + 1, value_end, quote));
                    }
                }
            }
        }
        search = after;
    }
    None
}

fn read_attribute(text: &str, key: &str) -> Option<String> {
    let (_, start, end, _) = find_attribute(text, key)?;
    Some(unescape_xml(&text[start..end]))
}

fn replace_attribute(text: &str, key: &str, value: &str) -> Option<String> {
    let (_, start, end, _) = find_attribute(text, key)?;
    let mut out = String::with_capacity(text.len() + 8);
    out.push_str(&text[..start]);
    out.push_str(&escape_attribute(value));
    out.push_str(&text[end..]);
    Some(out)
}

fn remove_attribute(text: &str, key: &str) -> Option<String> {
    let (start, _value_start, value_end, _) = find_attribute(text, key)?;
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..start]);
    out.push_str(&text[value_end + 1..]);
    Some(out)
}

fn insert_attribute(text: &str, key: &str, value: &str) -> Option<String> {
    let (start, gt, self_closing) = description_tag(text)?;
    let at = insertion_point(text, start, gt, self_closing);
    let mut out = String::with_capacity(text.len() + key.len() + value.len() + 8);
    out.push_str(&text[..at]);
    out.push_str(&format!(" {key}=\"{}\"", escape_attribute(value)));
    out.push_str(&text[at..]);
    Some(out)
}

fn find_element(text: &str, key: &str) -> Option<(usize, usize, usize)> {
    let open = format!("<{key}>");
    let close = format!("</{key}>");
    let start = text.find(&open)?;
    let content_start = start + open.len();
    let content_end = text[content_start..].find(&close)? + content_start;
    Some((start, content_start, content_end))
}

fn read_element(text: &str, key: &str) -> Option<String> {
    let (_, start, end) = find_element(text, key)?;
    Some(text[start..end].to_string())
}

fn replace_element(text: &str, key: &str, value: &str) -> Option<String> {
    let (_, start, end) = find_element(text, key)?;
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..start]);
    out.push_str(&escape_text(value));
    out.push_str(&text[end..]);
    Some(out)
}

fn remove_element(text: &str, key: &str) -> Option<String> {
    let open = format!("<{key}>");
    let close = format!("</{key}>");
    let start = text.find(&open)?;
    let end = text[start..].find(&close)? + start + close.len();
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..start]);
    out.push_str(&text[end..]);
    Some(out)
}

/// Insert a child block before `</rdf:Description>` (expanding a self-closing tag).
fn insert_child(text: &str, block: &str) -> Option<String> {
    let (start, gt, self_closing) = description_tag(text)?;
    if self_closing {
        let at = insertion_point(text, start, gt, self_closing);
        let mut out = String::with_capacity(text.len() + block.len() + 24);
        out.push_str(&text[..at]);
        out.push('>');
        out.push_str(block);
        out.push_str("</rdf:Description>");
        out.push_str(&text[gt + 1..]);
        Some(out)
    } else {
        let close = text.find("</rdf:Description>")?;
        let mut out = String::with_capacity(text.len() + block.len());
        out.push_str(&text[..close]);
        out.push_str(block);
        out.push_str(&text[close..]);
        Some(out)
    }
}

fn merge_attribute_value(
    text: &str,
    key: &str,
    value: &str,
    prefix: &str,
    namespace: &str,
) -> Option<String> {
    if let Some(out) = replace_attribute(text, key, value) {
        return Some(out);
    }
    if let Some(out) = replace_element(text, key, value) {
        return Some(out);
    }
    let with_namespace = ensure_namespace(text, prefix, namespace)?;
    insert_attribute(&with_namespace, key, value)
}

fn remove_value(text: &str, key: &str) -> Option<String> {
    if !text.contains("<rdf:Description") {
        return None;
    }
    let mut current = text.to_string();
    if let Some(next) = remove_attribute(&current, key) {
        current = next;
    }
    if let Some(next) = remove_element(&current, key) {
        current = next;
    }
    Some(current)
}

fn write_attribute(
    raw: &Path,
    key: &str,
    value: Option<&str>,
    prefix: &str,
    namespace: &str,
) -> Result<()> {
    let sidecar = match sidecar_target(raw) {
        Some(sidecar) => sidecar,
        None => return Ok(()),
    };
    if sidecar.is_file() {
        let text = read_sidecar(&sidecar).ok_or_else(|| unreadable_error(&sidecar))?;
        let updated = match value {
            Some(value) => merge_attribute_value(&text, key, value, prefix, namespace),
            None => remove_value(&text, key),
        }
        .ok_or_else(|| unreadable_error(&sidecar))?;
        if updated != text {
            atomic_write(&sidecar, updated.as_bytes())?;
        }
    } else if let Some(value) = value {
        if let Some(parent) = sidecar.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
        }
        atomic_write(&sidecar, fresh_document(&[(key, value)], &[]).as_bytes())?;
    }
    Ok(())
}

/// Write `rating` (0-5) into the RAW's XMP sidecar, merging when one exists.
pub fn write_rating(raw: &Path, rating: u8) -> Result<()> {
    let value = rating.min(5).to_string();
    write_attribute(raw, "xmp:Rating", Some(&value), "xmp", XMP_NS)
}

/// The `xmp:Rating` in the RAW's sidecar, clamped to 0-5.
pub fn read_rating(raw: &Path) -> Option<u8> {
    let sidecar = existing_sidecar(raw)?;
    let text = read_sidecar(&sidecar)?;
    let value =
        read_attribute(&text, "xmp:Rating").or_else(|| read_element(&text, "xmp:Rating"))?;
    Some(value.trim().parse::<u8>().ok()?.min(5))
}

/// Write (or clear) the `xmp:Label` value.
pub fn write_label(raw: &Path, label: Option<&str>) -> Result<()> {
    let label = label.map(str::trim).filter(|value| !value.is_empty());
    write_attribute(raw, "xmp:Label", label, "xmp", XMP_NS)
}

/// The `xmp:Label` in the RAW's sidecar.
pub fn read_label(raw: &Path) -> Option<String> {
    let sidecar = existing_sidecar(raw)?;
    let text = read_sidecar(&sidecar)?;
    let value = match read_attribute(&text, "xmp:Label") {
        Some(value) => value,
        None => unescape_xml(read_element(&text, "xmp:Label")?.trim()),
    };
    let value = value.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn parse_keywords(text: &str) -> Vec<String> {
    let Some(start) = text.find("<dc:subject") else {
        return Vec::new();
    };
    let Some(end) = text[start..].find("</dc:subject>") else {
        return Vec::new();
    };
    let block = &text[start..start + end];
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(open) = block[at..].find("<rdf:li>") {
        let content = at + open + "<rdf:li>".len();
        let Some(close) = block[content..].find("</rdf:li>") else {
            break;
        };
        let value = unescape_xml(block[content..content + close].trim());
        if !value.is_empty() {
            out.push(value);
        }
        at = content + close + "</rdf:li>".len();
    }
    out
}

/// The `dc:subject` keywords in the RAW's sidecar.
pub fn read_keywords(raw: &Path) -> Vec<String> {
    let Some(sidecar) = existing_sidecar(raw) else {
        return Vec::new();
    };
    let Some(text) = read_sidecar(&sidecar) else {
        return Vec::new();
    };
    parse_keywords(&text)
}

fn set_keywords(text: &str, keywords: &[String]) -> Option<String> {
    let has_block = text.contains("<dc:subject");
    if keywords.is_empty() {
        if !has_block {
            return Some(text.to_string());
        }
        let start = text.find("<dc:subject")?;
        let end = text[start..].find("</dc:subject>")? + start + "</dc:subject>".len();
        let mut out = String::with_capacity(text.len());
        out.push_str(&text[..start]);
        out.push_str(&text[end..]);
        return Some(out);
    }
    let block = keywords_block(keywords);
    if has_block {
        let start = text.find("<dc:subject")?;
        let end = text[start..].find("</dc:subject>")? + start + "</dc:subject>".len();
        let mut out = String::with_capacity(text.len() + block.len());
        out.push_str(&text[..start]);
        out.push_str(&block);
        out.push_str(&text[end..]);
        Some(out)
    } else {
        let with_namespace = ensure_namespace(text, "dc", DC_NS)?;
        insert_child(&with_namespace, &format!("\n   {block}"))
    }
}

/// Merge keywords: remove `remove`, add `add`, preserving every other keyword.
pub fn write_keywords(raw: &Path, remove: &[String], add: &[String]) -> Result<()> {
    let Some(sidecar) = sidecar_target(raw) else {
        return Ok(());
    };
    let has_sidecar = sidecar.is_file();
    let existing = if has_sidecar {
        Some(read_sidecar(&sidecar).ok_or_else(|| unreadable_error(&sidecar))?)
    } else {
        None
    };
    let mut keywords = existing.as_deref().map(parse_keywords).unwrap_or_default();
    let remove: Vec<String> = remove.iter().map(|value| value.to_lowercase()).collect();
    keywords.retain(|keyword| !remove.contains(&keyword.to_lowercase()));
    for keyword in add {
        let keyword = keyword.trim();
        if !keyword.is_empty()
            && !keywords
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(keyword))
        {
            keywords.push(keyword.to_string());
        }
    }
    if !has_sidecar {
        if keywords.is_empty() {
            return Ok(());
        }
        if let Some(parent) = sidecar.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
        }
        atomic_write(&sidecar, fresh_document(&[], &keywords).as_bytes())?;
        return Ok(());
    }
    let text = existing.unwrap_or_default();
    let updated = set_keywords(&text, &keywords).ok_or_else(|| unreadable_error(&sidecar))?;
    if updated != text {
        atomic_write(&sidecar, updated.as_bytes())?;
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

    fn merge(text: &str, rating: u8) -> Option<String> {
        merge_attribute_value(text, "xmp:Rating", &rating.to_string(), "xmp", XMP_NS)
    }

    #[test]
    fn replaces_single_quoted_and_spaced_attributes() {
        let single = r#"<rdf:Description xmp:Rating='2' tiff:Make="Canon"/>"#;
        let merged = merge(single, 5).unwrap();
        assert!(merged.contains("xmp:Rating='5'"), "{merged}");
        assert!(merged.contains("tiff:Make=\"Canon\""));

        let spaced = r#"<rdf:Description xmp:Rating = "1"/>"#;
        let merged = merge(spaced, 4).unwrap();
        assert!(merged.contains("xmp:Rating = \"4\""), "{merged}");
    }

    #[test]
    fn replaces_an_existing_attribute_and_keeps_everything_else() {
        let merged = merge(EXISTING, 3).unwrap();
        assert!(merged.contains("xmp:Rating=\"3"));
        assert!(!merged.contains("xmp:Rating=\"5\""));
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
    fn inserts_into_a_self_closing_description() {
        let text = r#"<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/"/>"#;
        let merged = merge(text, 4).unwrap();
        assert!(merged.contains("xmp:Rating=\"4\""), "{merged}");
        assert!(merged.ends_with("/>"), "{merged}");
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
        assert_eq!(read_rating(&raw), Some(5));
        // Re-rating merges rather than duplicating.
        write_rating(&raw, 2).unwrap();
        let text = std::fs::read_to_string(&sidecar).unwrap();
        assert!(text.contains("xmp:Rating=\"2\""));
        assert!(!text.contains("xmp:Rating=\"5\""));
    }

    #[test]
    fn reads_ratings_and_labels_from_attributes_and_elements() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("shot.CR3");
        std::fs::write(&raw, b"raw").unwrap();
        std::fs::write(sidecar_path(&raw), EXISTING).unwrap();
        assert_eq!(read_rating(&raw), Some(5));
        assert_eq!(read_label(&raw), None);

        write_label(&raw, Some("Red")).unwrap();
        assert_eq!(read_label(&raw).as_deref(), Some("Red"));
        assert_eq!(read_rating(&raw), Some(5));

        let element = "<rdf:Description><xmp:Rating>3</xmp:Rating><xmp:Label>Blue</xmp:Label></rdf:Description>";
        std::fs::write(sidecar_path(&raw), element).unwrap();
        assert_eq!(read_rating(&raw), Some(3));
        assert_eq!(read_label(&raw).as_deref(), Some("Blue"));
    }

    #[test]
    fn full_name_sidecars_are_found_and_edited_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("shot.CR3");
        std::fs::write(&raw, b"raw").unwrap();
        let full = full_sidecar_path(&raw);
        std::fs::write(&full, EXISTING).unwrap();
        assert_eq!(existing_sidecar(&raw).as_deref(), Some(full.as_path()));

        write_rating(&raw, 2).unwrap();
        assert!(full.is_file());
        assert!(
            !sidecar_path(&raw).exists(),
            "must not create a duplicate stem sidecar"
        );
        assert_eq!(read_rating(&raw), Some(2));
    }

    #[test]
    fn labels_and_keywords_merge_without_losing_unrelated_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("shot.CR3");
        std::fs::write(&raw, b"raw").unwrap();
        std::fs::write(sidecar_path(&raw), EXISTING).unwrap();

        write_label(&raw, Some("Green")).unwrap();
        write_keywords(&raw, &[], &["bird".to_string(), "egret".to_string()]).unwrap();
        let text = std::fs::read_to_string(sidecar_path(&raw)).unwrap();
        assert!(text.contains("xmp:Label=\"Green\""));
        assert!(text.contains("<rdf:li>bird</rdf:li>"));
        assert!(text.contains("<rdf:li>egret</rdf:li>"));
        // The rating, creator and make survive.
        assert_eq!(read_rating(&raw), Some(5));
        assert!(text.contains("<rdf:li>Michael G</rdf:li>"));
        assert!(text.contains("tiff:Make=\"Canon\""));
        assert_eq!(
            read_keywords(&raw),
            vec!["bird".to_string(), "egret".to_string()]
        );

        write_keywords(&raw, &["bird".to_string()], &["heron".to_string()]).unwrap();
        assert_eq!(
            read_keywords(&raw),
            vec!["egret".to_string(), "heron".to_string()]
        );
        assert!(std::fs::read_to_string(sidecar_path(&raw))
            .unwrap()
            .contains("<rdf:li>Michael G</rdf:li>"));
    }

    #[test]
    fn clearing_a_label_leaves_the_rest_intact() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("shot.CR3");
        std::fs::write(&raw, b"raw").unwrap();
        write_rating(&raw, 4).unwrap();
        write_label(&raw, Some("Yellow")).unwrap();
        write_label(&raw, None).unwrap();
        assert_eq!(read_label(&raw), None);
        assert_eq!(read_rating(&raw), Some(4));
    }

    #[test]
    fn a_malformed_sidecar_is_backed_up_and_left_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("shot.CR3");
        std::fs::write(&raw, b"raw").unwrap();
        let sidecar = sidecar_path(&raw);
        std::fs::write(&sidecar, "not really xmp").unwrap();

        let err = write_rating(&raw, 3).unwrap_err();
        assert!(err.to_string().contains("refusing to overwrite"), "{err}");
        // The user's file must survive verbatim.
        assert_eq!(std::fs::read_to_string(&sidecar).unwrap(), "not really xmp");
        // Repeating the action must not pile up backups.
        let _ = write_rating(&raw, 4).unwrap_err();
        let backups: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .filter(|name| name.ends_with(".bak"))
            .collect();
        assert_eq!(backups, vec!["shot.xmp.bak"], "{backups:?}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("shot.xmp.bak")).unwrap(),
            "not really xmp"
        );
    }

    #[test]
    fn the_rating_is_clamped_to_five() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("shot.CR3");
        std::fs::write(&raw, b"raw").unwrap();
        write_rating(&raw, 9).unwrap();
        let text = std::fs::read_to_string(sidecar_path(&raw)).unwrap();
        assert!(text.contains("xmp:Rating=\"5\""), "{text}");
    }

    #[test]
    fn a_jpeg_never_gets_the_raw_sidecar() {
        // `shot.xmp` sits next to `shot.CR3`; a rating set on `shot.JPG` must not
        // rewrite it (Lightroom would show the JPEG's rating on the RAW).
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("shot.CR3");
        let jpeg = dir.path().join("shot.JPG");
        std::fs::write(&raw, b"raw").unwrap();
        std::fs::write(&jpeg, b"jpeg").unwrap();
        write_rating(&raw, 5).unwrap();

        write_rating(&jpeg, 1).unwrap();
        assert_eq!(read_rating(&raw), Some(5), "the RAW rating is untouched");
        assert_eq!(read_rating(&jpeg), None, "no sidecar is read for a JPEG");
        let sidecars = std::fs::read_dir(dir.path())
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".xmp")
            })
            .count();
        assert_eq!(sidecars, 1);
    }

    #[test]
    fn rewriting_the_same_rating_is_a_no_op_and_leaves_no_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("shot.CR3");
        std::fs::write(&raw, b"raw").unwrap();
        write_rating(&raw, 4).unwrap();
        let sidecar = sidecar_path(&raw);
        let before = std::fs::read_to_string(&sidecar).unwrap();

        write_rating(&raw, 4).unwrap();
        assert_eq!(std::fs::read_to_string(&sidecar).unwrap(), before);
        let entries: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(entries.len(), 2, "{entries:?}");
    }
}
