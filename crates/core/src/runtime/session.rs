//! Shared ONNX Runtime session wrapper.
//!
//! Every model (detector, species, eye) runs a single `f32` input tensor and
//! shares one failure policy: if a GPU execution provider cannot actually
//! execute the model, rebuild the session on CPU **once** and retry; if CPU
//! fails, return the error. Centralising it here removes three copies of that
//! state machine and makes the transitions unit-testable without an ONNX model.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;

use crate::runtime::detector::{build_session, DevicePreference};
use crate::{CoreError, Result};

/// Map an ONNX Runtime error to a core error.
pub(crate) fn ort_err(e: impl std::fmt::Display) -> CoreError {
    CoreError::Other(format!("onnxruntime: {e}"))
}

/// Run `attempt` against `session`; on a failure **while a GPU provider is
/// active**, rebuild on CPU once and retry. `gpu` doubles as the one-shot guard:
/// it is cleared when the fallback is claimed, so a second failure (or a genuine
/// CPU failure) returns the typed error instead of rebuilding again.
fn run_with_cpu_fallback<S, R>(
    label: &str,
    gpu: &AtomicBool,
    session: &mut S,
    rebuild_cpu: impl FnOnce() -> Result<S>,
    mut attempt: impl FnMut(&mut S) -> Result<R>,
) -> Result<R> {
    match attempt(session) {
        Ok(out) => Ok(out),
        Err(e) => {
            if !gpu.swap(false, Ordering::Relaxed) {
                return Err(e);
            }
            tracing::warn!(model = label, error = %e, "provider failed; rebuilding the session on CPU");
            let cpu = rebuild_cpu()?;
            *session = cpu;
            attempt(session)
        }
    }
}

/// An ONNX session plus the shared provider-fallback policy.
pub(crate) struct OnnxSession {
    session: Mutex<Session>,
    model: PathBuf,
    /// Short name for diagnostics (the model file name).
    label: String,
    input_name: String,
    /// True while the session runs on a GPU execution provider.
    gpu: AtomicBool,
}

impl OnnxSession {
    /// Load `model`, preferring a GPU provider when the build has one enabled.
    pub(crate) fn load(model: &Path, device: DevicePreference) -> Result<Self> {
        let (session, ep) = build_session(model, device)?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        let label = model
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| model.display().to_string());
        Ok(Self {
            session: Mutex::new(session),
            model: model.to_path_buf(),
            label,
            input_name,
            gpu: AtomicBool::new(ep.is_some()),
        })
    }

    /// True while the session runs on a GPU execution provider.
    pub(crate) fn is_gpu(&self) -> bool {
        self.gpu.load(Ordering::Relaxed)
    }

    /// Run one `f32` input tensor and decode its first output with `f`.
    ///
    /// The session lock is held across the run so a provider fallback cannot
    /// race with another inference.
    pub(crate) fn run_tensor<R>(
        &self,
        tensor: &Tensor<f32>,
        mut f: impl FnMut(&[usize], &[f32]) -> Result<R>,
    ) -> Result<R> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| CoreError::Other("onnx session lock poisoned".into()))?;
        let model = self.model.clone();
        let input_name = self.input_name.clone();
        // The attempt returns `Ok(Err(..))` for a post-processing failure: only
        // a runtime/provider error (the outer `Err`) may rebuild on CPU. A bad
        // output must not permanently downgrade the session to CPU.
        let post = run_with_cpu_fallback(
            &self.label,
            &self.gpu,
            &mut *session,
            || build_session(&model, DevicePreference::Cpu).map(|(s, _)| s),
            |session| {
                let outputs = session
                    .run(ort::inputs![input_name.clone() => tensor])
                    .map_err(ort_err)?;
                let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
                let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
                Ok(f(&dims, data))
            },
        )
        .map_err(|e| CoreError::inference(&self.label, e.to_string()))?;
        post
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A synthetic session: fails its first `fail_times` runs, then succeeds.
    #[derive(Default)]
    struct FakeSession {
        fail_times: usize,
        attempts: usize,
    }

    impl FakeSession {
        fn failing() -> Self {
            Self {
                fail_times: usize::MAX,
                attempts: 0,
            }
        }
        fn working() -> Self {
            Self {
                fail_times: 0,
                attempts: 0,
            }
        }
        fn run(&mut self) -> Result<usize> {
            self.attempts += 1;
            if self.attempts <= self.fail_times {
                Err(CoreError::Other("provider boom".into()))
            } else {
                Ok(self.attempts)
            }
        }
    }

    #[test]
    fn gpu_failure_rebuilds_on_cpu_once_and_retries() {
        let gpu = AtomicBool::new(true);
        let mut session = FakeSession::failing();
        let rebuilds = Cell::new(0);

        let out = run_with_cpu_fallback(
            "model.onnx",
            &gpu,
            &mut session,
            || {
                rebuilds.set(rebuilds.get() + 1);
                Ok(FakeSession::working())
            },
            |s| s.run(),
        )
        .unwrap();

        assert_eq!(out, 1, "the CPU session succeeds on its first attempt");
        assert_eq!(rebuilds.get(), 1);
        assert!(!gpu.load(Ordering::Relaxed), "provider flag is cleared");
    }

    #[test]
    fn cpu_failure_is_a_typed_error_without_rebuilding() {
        let gpu = AtomicBool::new(false);
        let mut session = FakeSession::failing();
        let rebuilds = Cell::new(0);

        let err = run_with_cpu_fallback(
            "model.onnx",
            &gpu,
            &mut session,
            || {
                rebuilds.set(rebuilds.get() + 1);
                Ok(FakeSession::working())
            },
            |s| s.run(),
        )
        .unwrap_err();

        assert_eq!(
            rebuilds.get(),
            0,
            "a CPU failure must not trigger a rebuild"
        );
        assert!(err.to_string().contains("provider boom"));
    }

    #[test]
    fn only_one_cpu_rebuild_happens_per_instance() {
        let gpu = AtomicBool::new(true);
        let mut session = FakeSession::failing();
        let rebuilds = Cell::new(0);
        let rebuild_failing = || {
            rebuilds.set(rebuilds.get() + 1);
            Ok(FakeSession::failing())
        };

        // Both the GPU attempt and the CPU retry fail: one rebuild, then error.
        let first = run_with_cpu_fallback("model.onnx", &gpu, &mut session, rebuild_failing, |s| {
            s.run()
        });
        assert!(first.is_err());
        assert_eq!(rebuilds.get(), 1);
        assert!(!gpu.load(Ordering::Relaxed));

        // A later run is already on CPU: it must not rebuild again.
        let second = run_with_cpu_fallback(
            "model.onnx",
            &gpu,
            &mut session,
            || {
                rebuilds.set(rebuilds.get() + 1);
                Ok(FakeSession::working())
            },
            |s| s.run(),
        );
        assert!(second.is_err());
        assert_eq!(rebuilds.get(), 1, "the fallback is one-shot");
    }

    #[test]
    fn a_failed_rebuild_propagates() {
        let gpu = AtomicBool::new(true);
        let mut session = FakeSession::failing();

        let err = run_with_cpu_fallback(
            "model.onnx",
            &gpu,
            &mut session,
            || Err(CoreError::Other("could not build CPU session".into())),
            |s| s.run(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("could not build CPU session"));
    }

    #[test]
    fn a_post_processing_error_keeps_the_gpu_provider() {
        let gpu = AtomicBool::new(true);
        let mut session = FakeSession::working();
        let rebuilds = Cell::new(0);

        let out: Result<Result<usize>> = run_with_cpu_fallback(
            "model.onnx",
            &gpu,
            &mut session,
            || {
                rebuilds.set(rebuilds.get() + 1);
                Ok(FakeSession::working())
            },
            |s| {
                s.run()?;
                Ok(Err(CoreError::Other("bad output".into())))
            },
        );

        assert!(out.unwrap().is_err(), "the error is still returned");
        assert_eq!(rebuilds.get(), 0, "a bad output must not rebuild on CPU");
        assert!(gpu.load(Ordering::Relaxed), "the GPU provider is kept");
    }

    #[test]
    fn a_successful_run_keeps_the_gpu_provider() {
        let gpu = AtomicBool::new(true);
        let mut session = FakeSession::working();
        let rebuilds = Cell::new(0);

        let out = run_with_cpu_fallback(
            "model.onnx",
            &gpu,
            &mut session,
            || {
                rebuilds.set(rebuilds.get() + 1);
                Ok(FakeSession::working())
            },
            |s| s.run(),
        )
        .unwrap();

        assert_eq!(out, 1);
        assert_eq!(rebuilds.get(), 0);
        assert!(gpu.load(Ordering::Relaxed), "no failure, no fallback");
    }
}
