//! Model-manager commands: list, download, install from file, and remove the
//! optional ONNX models.
//!
//! These are self-contained (they only need the model registry and the shared
//! blocking bridge), so they live apart from the library/review commands.

use serde::Serialize;

use birdsorter_core::data::registry;

use crate::util::{blocking, err};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelStatusDto {
    key: String,
    display_name: String,
    description: String,
    tier: String,
    size_bytes: u64,
    status: String,
    files: Vec<String>,
    url: String,
    /// A planned model with no artefact yet (needs a training run).
    planned: bool,
}

#[tauri::command]
pub(crate) async fn models_status() -> Result<Vec<ModelStatusDto>, String> {
    blocking(|| {
        let dir = registry::default_model_dir().map_err(err)?;
        Ok(registry::builtin_models()
            .into_iter()
            .map(|s| {
                let status = registry::status(&dir, &s);
                ModelStatusDto {
                    key: s.key.clone(),
                    display_name: s.display_name.clone(),
                    description: s.description.clone(),
                    tier: format!("{:?}", s.tier).to_lowercase(),
                    size_bytes: s.size_bytes,
                    status: format!("{status:?}").to_lowercase(),
                    files: s.files.clone(),
                    url: s.url.clone(),
                    planned: s.tags.iter().any(|t| t == "planned"),
                }
            })
            .collect())
    })
    .await
}

#[tauri::command]
pub(crate) async fn download_model(key: String) -> Result<usize, String> {
    blocking(move || {
        let spec = registry::spec_by_key(&key).ok_or_else(|| format!("unknown model '{key}'"))?;
        let dir = registry::default_model_dir().map_err(err)?;
        let paths = registry::download(&dir, &spec, &registry::HttpFetcher).map_err(err)?;
        Ok(paths.len())
    })
    .await
}

#[tauri::command]
pub(crate) async fn import_model(key: String, path: String) -> Result<usize, String> {
    blocking(move || {
        let spec = registry::spec_by_key(&key).ok_or_else(|| format!("unknown model '{key}'"))?;
        let dir = registry::default_model_dir().map_err(err)?;
        let paths = registry::import_file(&dir, &spec, std::path::Path::new(&path)).map_err(err)?;
        Ok(paths.len())
    })
    .await
}

#[tauri::command]
pub(crate) async fn remove_model(key: String) -> Result<usize, String> {
    blocking(move || {
        let spec = registry::spec_by_key(&key).ok_or_else(|| format!("unknown model '{key}'"))?;
        let dir = registry::default_model_dir().map_err(err)?;
        registry::remove(&dir, &spec).map_err(err)
    })
    .await
}
