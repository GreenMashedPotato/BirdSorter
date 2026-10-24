//! Tiled ONNX denoise: the inference half of [`crate::io::denoise`].

use std::path::Path;

use ndarray::Array3;
use ort::value::Tensor;

use crate::io::denoise::{denoise_tiled, TilePlan};
use crate::runtime::detector::DevicePreference;
use crate::runtime::session::{ort_err, OnnxSession};
use crate::{CoreError, Result};

/// File name of the denoise graph inside the model directory.
pub const DENOISE_ONNX: &str = "denoise_srgb.onnx";

/// A loaded denoise model.
pub struct OnnxDenoiser {
    session: OnnxSession,
    plan: TilePlan,
}

impl OnnxDenoiser {
    /// Load `denoise_srgb.onnx` from `dir`; `None` when it is not installed.
    pub fn load(dir: &Path, device: DevicePreference) -> Result<Option<Self>> {
        Self::load_with(dir, device, TilePlan::default())
    }

    /// [`OnnxDenoiser::load`] with an explicit tile plan.
    pub fn load_with(dir: &Path, device: DevicePreference, plan: TilePlan) -> Result<Option<Self>> {
        let path = dir.join(DENOISE_ONNX);
        if !path.is_file() {
            return Ok(None);
        }
        let session = OnnxSession::load(&path, device)?;
        Ok(Some(Self { session, plan }))
    }

    pub fn available(&self) -> bool {
        true
    }

    /// True while the session runs on a GPU execution provider.
    pub fn is_gpu(&self) -> bool {
        self.session.is_gpu()
    }

    /// Denoise one RGB image (full strength; the caller blends toward the
    /// original with [`crate::io::denoise::blend_strength`]).
    pub fn denoise(&self, rgb: &Array3<u8>) -> Result<Array3<u8>> {
        let plan = self.plan;
        denoise_tiled(rgb, &plan, |tile, w, h| {
            let tensor = Tensor::from_array(([1i64, 3, h as i64, w as i64], tile.to_vec()))
                .map_err(ort_err)?;
            self.session.run_tensor(&tensor, |dims, data| {
                if dims.len() != 4 || dims[1] != 3 {
                    return Err(CoreError::inference(
                        "denoise",
                        format!("unexpected output shape {dims:?}"),
                    ));
                }
                let (oh, ow) = (dims[2], dims[3]);
                // NAFNet rounds an unaligned tile down to its stride; a tile that
                // shrank cannot be blended back, so say so instead of skewing it.
                if oh != h || ow != w {
                    return Err(CoreError::inference(
                        "denoise",
                        format!("model returned {ow}x{oh} for a {w}x{h} tile"),
                    ));
                }
                Ok(data.to_vec())
            })
        })
    }
}
