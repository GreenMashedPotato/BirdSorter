//! Desktop-wide rendering and resource constants.

/// Long side used for analysis decodes.
pub(crate) const ANALYSIS_LONG_SIDE: u32 = 2400;
/// Long side of grid thumbnails.
pub(crate) const THUMB_LONG_SIDE: u32 = 512;
/// Long side of the "Developed" preview (full demosaic is 45 MP / ~60 MB JPEG).
pub(crate) const DEVELOP_MAX_LONG_SIDE: u32 = 6000;
/// Upper bound on parallel decode workers during analysis, so peak memory stays
/// bounded on high-core-count machines.
///
/// Measured on a 20-core i7-14700F (247 CR3, 7.15 GB): 4 and 8 workers sustain
/// the same throughput (2.23 vs 2.24 img/s, species off; 0.82 vs 0.80 with
/// species), while 8 workers peaks at 4.0 GB RSS instead of 2.5 GB. The analysis
/// stage is the bottleneck, not decoding, so the lower bound wins.
pub(crate) const MAX_DECODE_THREADS: usize = 4;
