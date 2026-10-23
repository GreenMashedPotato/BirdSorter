//! Composite contact sheets: a grid of thumbnails with a filename/score
//! caption. Pure layout and drawing, so it is testable without a RAW or a UI.
//!
//! Captions use a built-in 5×7 uppercase font: no font dependency, no system
//! fonts, and identical output everywhere. The sheet is bounded by a total
//! pixel cap so a malformed request cannot exhaust memory.

use ndarray::Array3;

use crate::{CoreError, Result};

const MAX_PIXELS: u64 = 32_000_000;
const SHEET_BG: [u8; 3] = [22, 24, 28];
const CELL_BG: [u8; 3] = [32, 35, 41];
const CAPTION_BG: [u8; 3] = [28, 30, 35];
const CAPTION_FG: [u8; 3] = [225, 228, 236];

/// One thumbnail plus its caption and optional accent colour.
#[derive(Clone)]
pub struct ContactCell {
    pub rgb: Array3<u8>,
    pub caption: String,
    /// A short coloured bar in the caption strip (e.g. the verdict colour).
    pub accent: Option<[u8; 3]>,
}

/// Grid geometry. `cell_height` includes the caption strip.
#[derive(Debug, Clone, Copy)]
pub struct ContactOptions {
    pub cols: usize,
    pub cell_width: usize,
    pub cell_height: usize,
    pub margin: usize,
    pub caption_height: usize,
    pub font_scale: usize,
}

impl Default for ContactOptions {
    fn default() -> Self {
        Self {
            cols: 4,
            cell_width: 360,
            cell_height: 300,
            margin: 12,
            caption_height: 34,
            font_scale: 2,
        }
    }
}

/// Render a contact sheet. Cell order is row-major.
pub fn render(cells: &[ContactCell], options: &ContactOptions) -> Result<Array3<u8>> {
    if cells.is_empty() {
        return Err(CoreError::invalid_input(
            "a contact sheet needs at least one cell",
        ));
    }
    if options.cols == 0
        || options.cell_width < 16
        || options.cell_height <= options.caption_height + 4
    {
        return Err(CoreError::invalid_input(
            "invalid contact-sheet geometry (columns, cell width/height or caption height)",
        ));
    }
    let rows = cells.len().div_ceil(options.cols);
    let width = options
        .margin
        .checked_mul(options.cols + 1)
        .and_then(|value| value.checked_add(options.cell_width * options.cols))
        .ok_or_else(|| CoreError::invalid_input("contact sheet is too wide"))?;
    let height = options
        .margin
        .checked_mul(rows + 1)
        .and_then(|value| value.checked_add(options.cell_height * rows))
        .ok_or_else(|| CoreError::invalid_input("contact sheet is too tall"))?;
    if width == 0 || height == 0 || width as u64 * height as u64 > MAX_PIXELS {
        return Err(CoreError::invalid_input(format!(
            "contact sheet would be {width}x{height} px (over the {MAX_PIXELS}-pixel cap)"
        )));
    }

    let mut canvas = Array3::<u8>::zeros((height, width, 3));
    fill_rect(&mut canvas, 0, 0, width, height, SHEET_BG);
    let thumb_height = options.cell_height - options.caption_height;

    for (index, cell) in cells.iter().enumerate() {
        let col = index % options.cols;
        let row = index / options.cols;
        let x0 = options.margin + col * (options.cell_width + options.margin);
        let y0 = options.margin + row * (options.cell_height + options.margin);
        fill_rect(
            &mut canvas,
            x0,
            y0,
            options.cell_width,
            options.cell_height,
            CELL_BG,
        );

        let (img_h, img_w, _) = cell.rgb.dim();
        let (thumb_w, thumb_h) = fit_dimensions(img_w, img_h, options.cell_width, thumb_height);
        if thumb_w > 0 && thumb_h > 0 {
            let pixels: Vec<u8> = cell.rgb.iter().copied().collect();
            let Ok(w) = u32::try_from(img_w) else {
                continue;
            };
            let Ok(h) = u32::try_from(img_h) else {
                continue;
            };
            let Some(image) = image::RgbImage::from_raw(w, h, pixels) else {
                continue;
            };
            let Ok(tw) = u32::try_from(thumb_w) else {
                continue;
            };
            let Ok(th) = u32::try_from(thumb_h) else {
                continue;
            };
            let resized =
                image::imageops::resize(&image, tw, th, image::imageops::FilterType::Triangle);
            let ox = x0 + (options.cell_width - thumb_w) / 2;
            let oy = y0 + (thumb_height - thumb_h) / 2;
            for (x, y, pixel) in resized.enumerate_pixels() {
                set_pixel(&mut canvas, ox + x as usize, oy + y as usize, pixel.0);
            }
        }

        let caption_y = y0 + thumb_height;
        fill_rect(
            &mut canvas,
            x0,
            caption_y,
            options.cell_width,
            options.caption_height,
            CAPTION_BG,
        );
        if let Some(accent) = cell.accent {
            fill_rect(
                &mut canvas,
                x0,
                caption_y,
                4,
                options.caption_height,
                accent,
            );
        }
        let scale = options.font_scale.max(1);
        let text_height = 7 * scale;
        let text_y = caption_y + options.caption_height.saturating_sub(text_height) / 2;
        draw_text(
            &mut canvas,
            x0 + 10,
            text_y,
            &cell.caption.to_ascii_uppercase(),
            scale,
            CAPTION_FG,
            options.cell_width.saturating_sub(16),
        );
    }
    Ok(canvas)
}

/// The largest size that fits `width × height` inside `max_w × max_h` without
/// changing the aspect ratio.
fn fit_dimensions(width: usize, height: usize, max_w: usize, max_h: usize) -> (usize, usize) {
    if width == 0 || height == 0 || max_w == 0 || max_h == 0 {
        return (0, 0);
    }
    let scale = (max_w as f64 / width as f64).min(max_h as f64 / height as f64);
    let w = ((width as f64 * scale).round() as usize).max(1).min(max_w);
    let h = ((height as f64 * scale).round() as usize).max(1).min(max_h);
    (w, h)
}

fn set_pixel(canvas: &mut Array3<u8>, x: usize, y: usize, color: [u8; 3]) {
    let (height, width, _) = canvas.dim();
    if x >= width || y >= height {
        return;
    }
    for (channel, value) in color.iter().enumerate() {
        canvas[[y, x, channel]] = *value;
    }
}

fn fill_rect(
    canvas: &mut Array3<u8>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    color: [u8; 3],
) {
    let (canvas_h, canvas_w, _) = canvas.dim();
    for py in y..y.saturating_add(height).min(canvas_h) {
        for px in x..x.saturating_add(width).min(canvas_w) {
            for (channel, value) in color.iter().enumerate() {
                canvas[[py, px, channel]] = *value;
            }
        }
    }
}

fn draw_text(
    canvas: &mut Array3<u8>,
    x: usize,
    y: usize,
    text: &str,
    scale: usize,
    color: [u8; 3],
    max_width: usize,
) {
    let scale = scale.max(1);
    let advance = 6 * scale;
    let mut at = x;
    for character in text.chars() {
        if at.saturating_add(advance) > x.saturating_add(max_width) {
            break;
        }
        let glyph = glyph(character);
        for (row, pixels) in glyph.iter().enumerate() {
            for (col, pixel) in pixels.as_bytes().iter().enumerate() {
                if *pixel == b' ' {
                    continue;
                }
                for dy in 0..scale {
                    for dx in 0..scale {
                        set_pixel(canvas, at + col * scale + dx, y + row * scale + dy, color);
                    }
                }
            }
        }
        at += advance;
    }
}

/// A 5×7 uppercase font. Rows are top-to-bottom; unknown characters are blank.
fn glyph(character: char) -> [&'static str; 7] {
    match character {
        'A' => [
            " ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #",
        ],
        'B' => [
            "#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### ",
        ],
        'C' => [
            " ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### ",
        ],
        'D' => [
            "#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### ",
        ],
        'E' => [
            "#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####",
        ],
        'F' => [
            "#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    ",
        ],
        'G' => [
            " ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### ",
        ],
        'H' => [
            "#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #",
        ],
        'I' => [
            " ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### ",
        ],
        'J' => [
            "  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  ",
        ],
        'K' => [
            "#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #",
        ],
        'L' => [
            "#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####",
        ],
        'M' => [
            "#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #",
        ],
        'N' => [
            "#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #",
        ],
        'O' => [
            " ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### ",
        ],
        'P' => [
            "#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    ",
        ],
        'Q' => [
            " ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #",
        ],
        'R' => [
            "#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #",
        ],
        'S' => [
            " ####", "#    ", "#    ", " ### ", "    #", "    #", "#### ",
        ],
        'T' => [
            "#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ",
        ],
        'U' => [
            "#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### ",
        ],
        'V' => [
            "#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  ",
        ],
        'W' => [
            "#   #", "#   #", "#   #", "#   #", "# # #", "## ##", "#   #",
        ],
        'X' => [
            "#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #",
        ],
        'Y' => [
            "#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  ",
        ],
        'Z' => [
            "#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####",
        ],
        '0' => [
            " ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### ",
        ],
        '1' => [
            "  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### ",
        ],
        '2' => [
            " ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####",
        ],
        '3' => [
            "#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### ",
        ],
        '4' => [
            "   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # ",
        ],
        '5' => [
            "#####", "#    ", "#### ", "    #", "    #", "#   #", " ### ",
        ],
        '6' => [
            "  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### ",
        ],
        '7' => [
            "#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   ",
        ],
        '8' => [
            " ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### ",
        ],
        '9' => [
            " ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  ",
        ],
        '-' => [
            "     ", "     ", "     ", " ### ", "     ", "     ", "     ",
        ],
        '.' => [
            "     ", "     ", "     ", "     ", "     ", " ##  ", " ##  ",
        ],
        '_' => [
            "     ", "     ", "     ", "     ", "     ", "     ", "#####",
        ],
        ':' => [
            "     ", "  #  ", "  #  ", "     ", "  #  ", "  #  ", "     ",
        ],
        '/' => [
            "    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    ",
        ],
        '(' => [
            "   # ", "  #  ", " #   ", " #   ", " #   ", "  #  ", "   # ",
        ],
        ')' => [
            " #   ", "  #  ", "   # ", "   # ", "   # ", "  #  ", " #   ",
        ],
        '%' => [
            "##  #", "##  #", "   # ", "  #  ", " #   ", "#  ##", "#  ##",
        ],
        '#' => [
            " # # ", " # # ", "#####", " # # ", "#####", " # # ", " # # ",
        ],
        _ => ["     "; 7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_dimensions_preserves_aspect_ratio() {
        assert_eq!(fit_dimensions(100, 10, 20, 20), (20, 2));
        assert_eq!(fit_dimensions(10, 100, 20, 20), (2, 20));
        assert_eq!(fit_dimensions(0, 10, 20, 20), (0, 0));
    }

    #[test]
    fn renders_a_grid_with_captions() {
        let red = ContactCell {
            rgb: Array3::from_elem((6, 8, 3), 200u8),
            caption: "A1".into(),
            accent: Some([200, 80, 80]),
        };
        let mut green = red.clone();
        green.caption = "B2".into();
        green.rgb = Array3::from_elem((6, 8, 3), 10u8);
        let options = ContactOptions {
            cols: 2,
            cell_width: 40,
            cell_height: 30,
            margin: 4,
            caption_height: 10,
            font_scale: 1,
        };
        let sheet = render(&[red, green], &options).unwrap();
        // 2 columns: 4 + 2*40 + 2*4 = 92; 1 row: 4 + 30 + 4 = 38.
        assert_eq!(sheet.dim(), (38, 92, 3));
        assert_eq!(
            [sheet[[0, 0, 0]], sheet[[0, 0, 1]], sheet[[0, 0, 2]]],
            SHEET_BG
        );
        // The first thumbnail is centred in its 40x20 area.
        assert_eq!(sheet[[4, 11, 0]], 200);
        // "A" starts at x=14, y=25; its top row is ` ### ` (columns 1..3).
        // This also guards the font's row/column orientation.
        for x in 15..=17 {
            assert!(sheet[[25, x, 0]] > 150, "glyph pixel at x={x}");
        }
        assert_eq!(sheet[[25, 14, 0]], CAPTION_BG[0]);
        let caption_drawn = (24..34)
            .flat_map(|y| (4..44).map(move |x| (y, x)))
            .any(|(y, x)| sheet[[y, x, 0]] > 150);
        assert!(caption_drawn, "caption text was not drawn");
        // The accent bar is on the left of the caption.
        assert_eq!(
            [sheet[[24, 4, 0]], sheet[[24, 4, 1]], sheet[[24, 4, 2]]],
            [200, 80, 80]
        );
    }

    #[test]
    fn rejects_empty_or_unbounded_sheets() {
        let options = ContactOptions::default();
        assert!(render(&[], &options).is_err());
        let cell = ContactCell {
            rgb: Array3::from_elem((2, 2, 3), 0u8),
            caption: "x".into(),
            accent: None,
        };
        assert!(render(&[cell], &ContactOptions { cols: 0, ..options }).is_err());
        assert!(render(
            &[ContactCell {
                rgb: Array3::from_elem((2, 2, 3), 0u8),
                caption: "x".into(),
                accent: None,
            }],
            &ContactOptions {
                cols: 1000,
                cell_width: 1000,
                cell_height: 1000,
                ..options
            }
        )
        .is_err());
    }
}
