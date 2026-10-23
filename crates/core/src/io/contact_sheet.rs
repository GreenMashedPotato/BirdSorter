//! Composite contact sheets: a grid of thumbnails with a filename/score
//! caption. Pure layout and drawing, so it is testable without a RAW or a UI.
//!
//! Captions use a built-in 5×7 uppercase font: no font dependency, no system
//! fonts, and identical output everywhere. The sheet is bounded by a total
//! pixel cap so a malformed request cannot exhaust memory.

use ndarray::Array3;

use crate::{CoreError, Result};

const MAX_PIXELS: u64 = 32_000_000;
/// Print runs are rendered page by page; bound the whole batch as well.
const MAX_PRINT_PIXELS: u64 = 64_000_000;
const MAX_PRINT_PAGES: usize = 100;
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

/// A paper size, portrait, in millimetres.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageSize {
    A4,
    Letter,
}

impl PageSize {
    pub fn millimetres(self) -> (f64, f64) {
        match self {
            PageSize::A4 => (210.0, 297.0),
            PageSize::Letter => (215.9, 279.4),
        }
    }
}

/// A print layout: fixed page size and dpi, with a rows × columns grid per
/// page. Cells beyond one page continue on the next page.
#[derive(Debug, Clone, Copy)]
pub struct PrintOptions {
    pub page: PageSize,
    pub cols: usize,
    pub rows: usize,
    /// Render resolution; 150 dpi matches an ordinary office printer.
    pub dpi: u32,
    pub margin_mm: f64,
    pub caption_height: usize,
    pub font_scale: usize,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            page: PageSize::A4,
            cols: 4,
            rows: 5,
            dpi: 150,
            margin_mm: 10.0,
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

    for (index, cell) in cells.iter().enumerate() {
        let col = index % options.cols;
        let row = index / options.cols;
        let x0 = options.margin + col * (options.cell_width + options.margin);
        let y0 = options.margin + row * (options.cell_height + options.margin);
        draw_cell(
            &mut canvas,
            cell,
            CellLayout {
                x: x0,
                y: y0,
                width: options.cell_width,
                height: options.cell_height,
                caption_height: options.caption_height,
                font_scale: options.font_scale,
            },
        );
    }
    Ok(canvas)
}

/// Render a print run: one image per page, cells in row-major order.
///
/// The grid is centred on the paper; pages after the first continue the cell
/// list. The batch is bounded by `MAX_PRINT_PIXELS`.
pub fn render_pages(cells: &[ContactCell], options: &PrintOptions) -> Result<Vec<Array3<u8>>> {
    if cells.is_empty() {
        return Err(CoreError::invalid_input(
            "a contact sheet needs at least one cell",
        ));
    }
    if options.cols == 0 || options.rows == 0 {
        return Err(CoreError::invalid_input(
            "print layouts need at least one row and one column",
        ));
    }
    if !(72..=600).contains(&options.dpi) {
        return Err(CoreError::invalid_input(
            "print dpi must be between 72 and 600",
        ));
    }
    if !options.margin_mm.is_finite() || options.margin_mm < 0.0 {
        return Err(CoreError::invalid_input(
            "print margin must be non-negative",
        ));
    }

    let page = PagePixels::new(options)?;
    let per_page = options
        .cols
        .checked_mul(options.rows)
        .ok_or_else(|| CoreError::invalid_input("print grid is too large"))?;
    let pages = cells.len().div_ceil(per_page);
    if pages == 0 || pages > MAX_PRINT_PAGES {
        return Err(CoreError::invalid_input(format!(
            "{pages} pages is over the {MAX_PRINT_PAGES}-page cap"
        )));
    }
    let total = (page.width as u64)
        .saturating_mul(page.height as u64)
        .saturating_mul(pages as u64);
    if total > MAX_PRINT_PIXELS {
        return Err(CoreError::invalid_input(format!(
            "{pages} pages would be {total} pixels (over the {MAX_PRINT_PIXELS}-pixel cap)"
        )));
    }

    let mut out = Vec::with_capacity(pages);
    for page_index in 0..pages {
        let mut canvas = Array3::<u8>::zeros((page.height, page.width, 3));
        fill_rect(&mut canvas, 0, 0, page.width, page.height, SHEET_BG);
        let start = page_index * per_page;
        let end = (start + per_page).min(cells.len());
        for (slot, cell) in cells[start..end].iter().enumerate() {
            let col = slot % options.cols;
            let row = slot / options.cols;
            draw_cell(
                &mut canvas,
                cell,
                CellLayout {
                    x: page.left + col * page.cell_width,
                    y: page.top + row * page.cell_height,
                    width: page.cell_width,
                    height: page.cell_height,
                    caption_height: options.caption_height,
                    font_scale: options.font_scale,
                },
            );
        }
        out.push(canvas);
    }
    Ok(out)
}

/// Page and cell geometry at the requested dpi.
struct PagePixels {
    width: usize,
    height: usize,
    cell_width: usize,
    cell_height: usize,
    left: usize,
    top: usize,
}

impl PagePixels {
    fn new(options: &PrintOptions) -> Result<Self> {
        let (width_mm, height_mm) = options.page.millimetres();
        let to_px = |millimetres: f64| -> usize {
            (millimetres * f64::from(options.dpi) / 25.4)
                .round()
                .max(1.0) as usize
        };
        let width = to_px(width_mm);
        let height = to_px(height_mm);
        if width as u64 * height as u64 > MAX_PIXELS {
            return Err(CoreError::invalid_input("page is too large"));
        }
        let margin = to_px(options.margin_mm);
        let usable_w = width
            .checked_sub(margin.saturating_mul(2))
            .filter(|value| *value > 0)
            .ok_or_else(|| CoreError::invalid_input("print margin leaves no page area"))?;
        let usable_h = height
            .checked_sub(margin.saturating_mul(2))
            .filter(|value| *value > 0)
            .ok_or_else(|| CoreError::invalid_input("print margin leaves no page area"))?;
        // Thumbnails keep a 3:2 landscape box, so a 2-column page does not turn
        // into one huge empty cell; a dense grid is limited by the slot instead.
        const PRINT_ASPECT: f64 = 1.5;
        let slot_w = usable_w / options.cols;
        let slot_h = usable_h / options.rows;
        let mut cell_width = slot_w;
        let mut cell_height = ((slot_w as f64 / PRINT_ASPECT).round() as usize)
            .saturating_add(options.caption_height);
        if cell_height > slot_h {
            cell_height = slot_h;
            let available = cell_height.saturating_sub(options.caption_height);
            cell_width = ((available as f64 * PRINT_ASPECT).round() as usize).min(slot_w);
        }
        if cell_width < 16 || cell_height <= options.caption_height + 4 {
            return Err(CoreError::invalid_input(
                "the page is too small for this rows × columns grid",
            ));
        }
        Ok(Self {
            width,
            height,
            cell_width,
            cell_height,
            left: (width - cell_width * options.cols) / 2,
            top: (height - cell_height * options.rows) / 2,
        })
    }
}

/// Where one cell sits and how tall its caption strip is.
struct CellLayout {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    caption_height: usize,
    font_scale: usize,
}

/// Draw one cell (background, fitted thumbnail, caption strip) into `canvas`.
fn draw_cell(canvas: &mut Array3<u8>, cell: &ContactCell, layout: CellLayout) {
    let CellLayout {
        x: x0,
        y: y0,
        width: cell_width,
        height: cell_height,
        caption_height,
        font_scale,
    } = layout;
    fill_rect(canvas, x0, y0, cell_width, cell_height, CELL_BG);
    let thumb_height = cell_height.saturating_sub(caption_height);
    let (img_h, img_w, _) = cell.rgb.dim();
    let (thumb_w, thumb_h) = fit_dimensions(img_w, img_h, cell_width, thumb_height);
    if thumb_w > 0 && thumb_h > 0 {
        let pixels: Vec<u8> = cell.rgb.iter().copied().collect();
        if let (Ok(w), Ok(h), Ok(tw), Ok(th)) = (
            u32::try_from(img_w),
            u32::try_from(img_h),
            u32::try_from(thumb_w),
            u32::try_from(thumb_h),
        ) {
            if let Some(image) = image::RgbImage::from_raw(w, h, pixels) {
                let resized =
                    image::imageops::resize(&image, tw, th, image::imageops::FilterType::Triangle);
                let ox = x0 + (cell_width - thumb_w) / 2;
                let oy = y0 + (thumb_height - thumb_h) / 2;
                for (x, y, pixel) in resized.enumerate_pixels() {
                    set_pixel(canvas, ox + x as usize, oy + y as usize, pixel.0);
                }
            }
        }
    }

    let caption_y = y0 + thumb_height;
    fill_rect(
        canvas,
        x0,
        caption_y,
        cell_width,
        caption_height,
        CAPTION_BG,
    );
    if let Some(accent) = cell.accent {
        fill_rect(canvas, x0, caption_y, 4, caption_height, accent);
    }
    let scale = font_scale.max(1);
    let text_height = 7 * scale;
    let text_y = caption_y + caption_height.saturating_sub(text_height) / 2;
    draw_text(
        canvas,
        x0 + 10,
        text_y,
        &cell.caption.to_ascii_uppercase(),
        scale,
        CAPTION_FG,
        cell_width.saturating_sub(16),
    );
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

    fn print_cell() -> ContactCell {
        ContactCell {
            rgb: Array3::from_elem((6, 8, 3), 120u8),
            caption: "A1".into(),
            accent: Some([200, 80, 80]),
        }
    }

    #[test]
    fn print_layout_uses_the_paper_size_at_150_dpi() {
        let options = PrintOptions {
            page: PageSize::A4,
            cols: 2,
            rows: 2,
            ..PrintOptions::default()
        };
        let pages = render_pages(&[print_cell(), print_cell()], &options).unwrap();
        assert_eq!(pages.len(), 1);
        // A4 is 210 × 297 mm: 1240 × 1754 px at 150 dpi.
        assert_eq!(pages[0].dim(), (1754, 1240, 3));
        assert_eq!(
            [
                pages[0][[0, 0, 0]],
                pages[0][[0, 0, 1]],
                pages[0][[0, 0, 2]]
            ],
            SHEET_BG
        );

        let letter = PrintOptions {
            page: PageSize::Letter,
            cols: 2,
            rows: 2,
            ..PrintOptions::default()
        };
        let pages = render_pages(&[print_cell()], &letter).unwrap();
        // Letter is 215.9 × 279.4 mm: 1275 × 1650 px at 150 dpi.
        assert_eq!(pages[0].dim(), (1650, 1275, 3));
    }

    #[test]
    fn print_layout_continues_on_the_next_page() {
        let options = PrintOptions {
            page: PageSize::A4,
            cols: 2,
            rows: 2,
            ..PrintOptions::default()
        };
        let cells: Vec<_> = (0..5).map(|_| print_cell()).collect();
        let pages = render_pages(&cells, &options).unwrap();
        assert_eq!(pages.len(), 2, "5 cells at 4 per page = 2 pages");
        // The second page has one cell at the top-left; the next slot is bare.
        let page = &pages[1];
        let geometry = PagePixels::new(&options).unwrap();
        let second_x = geometry.left + geometry.cell_width;
        let second_y = geometry.top;
        assert_eq!(
            [page[[second_y, second_x, 0]], page[[second_y, second_x, 1]]],
            [SHEET_BG[0], SHEET_BG[1]]
        );
        // The first slot is the cell background, not the page background.
        assert_eq!(page[[geometry.top, geometry.left, 0]], CELL_BG[0]);
    }

    #[test]
    fn print_layout_validates_geometry_and_bounds() {
        let cell = print_cell();
        let one = std::slice::from_ref(&cell);
        let options = PrintOptions::default();
        assert!(render_pages(&[], &options).is_err());
        assert!(render_pages(one, &PrintOptions { rows: 0, ..options }).is_err());
        assert!(render_pages(one, &PrintOptions { dpi: 10, ..options }).is_err());
        assert!(render_pages(
            one,
            &PrintOptions {
                margin_mm: 500.0,
                ..options
            }
        )
        .is_err());
        // A grid too dense for a caption strip must be refused.
        assert!(render_pages(
            one,
            &PrintOptions {
                cols: 40,
                rows: 60,
                ..options
            }
        )
        .is_err());
    }
}
