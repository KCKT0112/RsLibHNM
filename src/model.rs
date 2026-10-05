use crate::{
    Analysis,
    types::{Features, Harmonics, Spectrum},
};
use anyhow::{Result, ensure};
use std::path::Path;

const MAGIC_FULL: &[u8; 8] = b"RSHNM001";
const MAGIC_COMPACT: &[u8; 8] = b"RSHNC001";
#[derive(serde::Serialize, serde::Deserialize)]
struct DiskHarmonics {
    channels: usize,
    frames: usize,
    capacity: usize,
    f0: Vec<f32>,
    confidence: Vec<f32>,
    frequency: Vec<f32>,
    slope: Vec<f32>,
    amplitude: Vec<f32>,
    phase: Vec<f32>,
    amplitude_gradient: Vec<f32>,
    count: Vec<u32>,
    window_length: Vec<u32>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct DiskFeatures {
    noise_psd: Vec<f32>,
    noise_psd_edges: Vec<f32>,
    noise_band_energy: Vec<f32>,
    noise_band_edges: Vec<f32>,
    noise_modulation: Vec<f32>,
    vector: Vec<f32>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct DiskModel {
    metadata: crate::Metadata,
    harmonics: DiskHarmonics,
    features: DiskFeatures,
}
fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}
fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}
fn disk(a: &Analysis) -> DiskModel {
    let h = &a.harmonics;
    let f = &a.features;
    DiskModel {
        metadata: a.metadata.clone(),
        harmonics: DiskHarmonics {
            channels: h.channels,
            frames: h.frames,
            capacity: h.capacity,
            f0: f32s(&h.f0),
            confidence: f32s(&h.confidence),
            frequency: f32s(&h.frequency),
            slope: f32s(&h.slope),
            amplitude: f32s(&h.amplitude),
            phase: f32s(&h.phase),
            amplitude_gradient: f32s(&h.amplitude_gradient),
            count: h.count.iter().map(|x| *x as u32).collect(),
            window_length: h.window_length.iter().map(|x| *x as u32).collect(),
        },
        features: DiskFeatures {
            noise_psd: f32s(&f.noise_psd),
            noise_psd_edges: f32s(&f.noise_psd_edges),
            noise_band_energy: f32s(&f.noise_band_energy),
            noise_band_edges: f32s(&f.noise_band_edges),
            noise_modulation: f32s(&f.noise_modulation),
            vector: f32s(&f.vector),
        },
    }
}
fn inflate(d: DiskModel) -> Analysis {
    let h = d.harmonics;
    let f = d.features;
    Analysis {
        metadata: d.metadata,
        mixture_stft: Spectrum {
            channels: h.channels,
            frames: h.frames,
            bins: 0,
            data: Vec::new(),
        },
        harmonic_stft: Spectrum {
            channels: h.channels,
            frames: h.frames,
            bins: 0,
            data: Vec::new(),
        },
        harmonics: Harmonics {
            channels: h.channels,
            frames: h.frames,
            capacity: h.capacity,
            f0: f64s(&h.f0),
            confidence: f64s(&h.confidence),
            frequency: f64s(&h.frequency),
            slope: f64s(&h.slope),
            amplitude: f64s(&h.amplitude),
            phase: f64s(&h.phase),
            amplitude_gradient: f64s(&h.amplitude_gradient),
            count: h.count.iter().map(|x| *x as usize).collect(),
            window_length: h.window_length.iter().map(|x| *x as usize).collect(),
        },
        features: Features {
            noise_envelope: vec![0.0; h.channels * h.frames],
            noise_psd: f64s(&f.noise_psd),
            noise_psd_edges: f64s(&f.noise_psd_edges),
            noise_band_energy: f64s(&f.noise_band_energy),
            noise_band_edges: f64s(&f.noise_band_edges),
            noise_modulation: f64s(&f.noise_modulation),
            spectral_cepstrum: Vec::new(),
            bap_cepstrum: Vec::new(),
            lip_response: Vec::new(),
            mel_log_spectrum: Vec::new(),
            bap: Vec::new(),
            rd: Vec::new(),
            rd_confidence: Vec::new(),
            vector: f64s(&f.vector),
        },
    }
}
pub fn save(analysis: &Analysis, path: &Path) -> Result<()> {
    analysis.validate()?;
    let raw = bincode::serialize(analysis)?;
    let compressed = zstd::encode_all(raw.as_slice(), 3)?;
    let mut file = MAGIC_FULL.to_vec();
    file.extend_from_slice(&compressed);
    std::fs::write(path, file)?;
    Ok(())
}
pub fn save_compact(analysis: &Analysis, path: &Path) -> Result<()> {
    analysis.validate()?;
    let raw = bincode::serialize(&disk(analysis))?;
    let compressed = zstd::encode_all(raw.as_slice(), 5)?;
    let mut file = MAGIC_COMPACT.to_vec();
    file.extend_from_slice(&compressed);
    std::fs::write(path, file)?;
    Ok(())
}
pub fn load(path: &Path) -> Result<Analysis> {
    let bytes = std::fs::read(path)?;
    let compact = bytes.starts_with(MAGIC_COMPACT);
    ensure!(
        compact || bytes.starts_with(MAGIC_FULL),
        "invalid Rust HNM model signature/version"
    );
    let raw = zstd::decode_all(&bytes[8..])?;
    let analysis = if compact {
        inflate(bincode::deserialize::<DiskModel>(&raw)?)
    } else {
        bincode::deserialize::<Analysis>(&raw)?
    };
    analysis.validate()?;
    Ok(analysis)
}
