use crate::Analysis;
use anyhow::{Result, ensure};
use std::path::Path;
const MAGIC: &[u8; 8] = b"RSHNM001";
fn write(analysis: &Analysis, path: &Path, compact: bool) -> Result<()> {
    analysis.validate()?;
    let mut value = analysis.clone();
    if compact {
        value.mixture_stft.data.clear();
        value.harmonic_stft.data.clear();
    }
    let raw = bincode::serialize(&value)?;
    let compressed = zstd::encode_all(raw.as_slice(), 3)?;
    let mut file = MAGIC.to_vec();
    file.extend_from_slice(&compressed);
    std::fs::write(path, file)?;
    Ok(())
}
pub fn save(analysis: &Analysis, path: &Path) -> Result<()> {
    write(analysis, path, false)
}
pub fn save_compact(analysis: &Analysis, path: &Path) -> Result<()> {
    write(analysis, path, true)
}
pub fn load(path: &Path) -> Result<Analysis> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.starts_with(MAGIC),
        "invalid Rust HNM model signature/version"
    );
    let raw = zstd::decode_all(&bytes[MAGIC.len()..])?;
    let analysis: bincode::Result<Analysis> = bincode::deserialize(&raw);
    let value = analysis?;
    value.validate()?;
    Ok(value)
}
