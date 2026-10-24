//! Desktop-wide rendering and resource constants.

/// Long side used for analysis decodes.
pub(crate) const ANALYSIS_LONG_SIDE: u32 = 2400;
/// Long side of grid thumbnails.
pub(crate) const THUMB_LONG_SIDE: u32 = 512;
/// Long side of the "Developed" preview (full demosaic is 45 MP / ~60 MB JPEG).
pub(crate) const DEVELOP_MAX_LONG_SIDE: u32 = 6000;
/// Upper bound on parallel decode workers during analysis, so the desktop stays
/// responsive on high-core-count machines.
pub(crate) const MAX_DECODE_THREADS: usize = 8;
