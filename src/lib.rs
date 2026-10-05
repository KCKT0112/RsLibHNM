pub mod audio;
pub mod config;
pub mod dsp;
pub mod envelope;
pub mod features;
pub mod glottal;
pub mod harmonics;
pub mod model;
pub mod pitch;
pub mod synthesis;
pub mod types;
use anyhow::{Result, ensure};
use config::Config;
use types::{Audio, Features, Harmonics, Spectrum};

pub const ANALYSIS_VERSION: u32 = 1;
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Analysis {
    pub metadata: Metadata,
    pub mixture_stft: Spectrum,
    pub harmonic_stft: Spectrum,
    pub harmonics: Harmonics,
    pub features: Features,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Metadata {
    pub analysis_version: u32,
    pub sample_rate: u32,
    pub num_samples: usize,
    pub channels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub left_padding: usize,
    pub input_assumption: String,
    pub feature_dimension: usize,
    pub config: Config,
}
impl Analysis {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.metadata.analysis_version == ANALYSIS_VERSION,
            "unsupported analysis version"
        );
        self.metadata.config.validate(self.metadata.sample_rate)?;
        let channels = self.metadata.channels;
        let frames = self.harmonics.frames;
        let bins = self.metadata.n_fft / 2 + 1;
        let cap = self.metadata.config.maxnhar;
        ensure!(
            channels > 0 && self.metadata.num_samples > 0 && self.metadata.feature_dimension == 72,
            "invalid metadata"
        );
        let spectra_present =
            !self.mixture_stft.data.is_empty() || !self.harmonic_stft.data.is_empty();
        if spectra_present {
            ensure!(
                self.mixture_stft.channels == channels
                    && self.harmonic_stft.channels == channels
                    && self.mixture_stft.frames == frames
                    && self.harmonic_stft.frames == frames
                    && self.mixture_stft.bins == bins
                    && self.harmonic_stft.bins == bins,
                "STFT shape mismatch"
            );
            ensure!(
                self.mixture_stft.data.len() == channels * frames * bins
                    && self.harmonic_stft.data.len() == channels * frames * bins,
                "STFT data mismatch"
            );
        }
        ensure!(
            self.harmonics.channels == channels
                && self.harmonics.capacity == cap
                && self.harmonics.f0.len() == frames
                && self.harmonics.frequency.len() == frames * cap
                && self.harmonics.slope.len() == frames * cap
                && self.harmonics.amplitude.len() == channels * frames * cap
                && self.harmonics.phase.len() == channels * frames * cap
                && self.harmonics.amplitude_gradient.len() == channels * frames
                && self.harmonics.count.len() == frames
                && self.harmonics.window_length.len() == frames,
            "harmonic data shape mismatch"
        );
        ensure!(
            self.mixture_stft
                .data
                .iter()
                .chain(&self.harmonic_stft.data)
                .all(|z| z.re.is_finite() && z.im.is_finite())
                && self
                    .harmonics
                    .f0
                    .iter()
                    .chain(&self.harmonics.amplitude)
                    .chain(&self.features.vector)
                    .all(|x| x.is_finite()),
            "model contains non-finite data"
        );
        ensure!(
            self.features.vector.len() == channels * frames * 72,
            "feature shape mismatch"
        );
        Ok(())
    }
    pub fn noise_stft(&self) -> Spectrum {
        let mut out = self.mixture_stft.clone();
        for (x, h) in out.data.iter_mut().zip(&self.harmonic_stft.data) {
            *x -= h;
        }
        out
    }
    pub fn reconstruct(&self, component: &str) -> Result<Audio> {
        ensure!(
            !self.mixture_stft.data.is_empty() && !self.harmonic_stft.data.is_empty(),
            "compact model has no replay spectra"
        );
        let spectrum = if component == "noise" {
            return dsp::istft(
                &self.noise_stft(),
                self.metadata.n_fft,
                self.metadata.hop_length,
                self.metadata.num_samples,
                self.metadata.sample_rate,
            );
        } else {
            match component {
                "mixture" => &self.mixture_stft,
                "harmonic" => &self.harmonic_stft,
                _ => anyhow::bail!("component must be mixture, harmonic, or noise"),
            }
        };
        dsp::istft(
            spectrum,
            self.metadata.n_fft,
            self.metadata.hop_length,
            self.metadata.num_samples,
            self.metadata.sample_rate,
        )
    }
}
pub fn analyze(audio: &Audio, config: Config) -> Result<Analysis> {
    let hop = config.hop(audio.sample_rate);
    let mixture = dsp::stft(audio, config.n_fft, hop);
    let frames = mixture.frames;
    let (f0, confidence) = pitch::estimate(audio, &config, hop, frames);
    let (harmonics, harmonic_wave) = harmonics::fit(audio, &f0, &confidence, hop, &config)?;
    let harmonic_audio = Audio {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        data: harmonic_wave,
        encoding: "FLOAT64".into(),
    };
    let harmonic_stft = dsp::stft(&harmonic_audio, config.n_fft, hop);
    let residual = Audio {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        data: audio
            .data
            .iter()
            .zip(&harmonic_audio.data)
            .map(|(x, h)| x - h)
            .collect(),
        encoding: "FLOAT64".into(),
    };
    let features = features::extract(&harmonic_audio, &residual, &harmonics, &config, hop)?;
    let metadata = Metadata {
        analysis_version: ANALYSIS_VERSION,
        sample_rate: audio.sample_rate,
        num_samples: audio.samples(),
        channels: audio.channels,
        n_fft: config.n_fft,
        hop_length: hop,
        left_padding: config.n_fft / 2,
        input_assumption: "isolated_clean_vocal".into(),
        feature_dimension: 72,
        config,
    };
    let result = Analysis {
        metadata,
        mixture_stft: mixture,
        harmonic_stft,
        harmonics,
        features,
    };
    result.validate()?;
    Ok(result)
}
