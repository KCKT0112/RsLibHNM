use anyhow::{Result, ensure};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Interleaved sample-major audio: data[sample * channels + channel].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Audio {
    pub sample_rate: u32,
    pub channels: usize,
    pub data: Vec<f64>,
    pub encoding: String,
}
impl Audio {
    pub fn samples(&self) -> usize {
        self.data.len() / self.channels.max(1)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_rate >= 1000 && self.channels > 0,
            "invalid audio rate/channels"
        );
        ensure!(
            !self.data.is_empty() && self.data.len() % self.channels == 0,
            "invalid audio shape"
        );
        ensure!(
            self.data.iter().all(|x| x.is_finite()),
            "audio must be finite"
        );
        Ok(())
    }
}

/// Channel/frame/bin order, index = (c * frames + frame) * bins + bin.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Spectrum {
    pub channels: usize,
    pub frames: usize,
    pub bins: usize,
    pub data: Vec<Complex64>,
}

/// Frequency/slope: frame-major (frames*capacity). Amplitude/phase: c,f,k.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Harmonics {
    pub channels: usize,
    pub frames: usize,
    pub capacity: usize,
    pub f0: Vec<f64>,
    pub confidence: Vec<f64>,
    pub frequency: Vec<f64>,
    pub slope: Vec<f64>,
    pub amplitude: Vec<f64>,
    pub phase: Vec<f64>,
    pub amplitude_gradient: Vec<f64>, // c,f
    pub count: Vec<usize>,
    pub window_length: Vec<usize>,
}
impl Harmonics {
    pub fn new(channels: usize, frames: usize, capacity: usize) -> Self {
        Self {
            channels,
            frames,
            capacity,
            f0: vec![0.; frames],
            confidence: vec![0.; frames],
            frequency: vec![0.; frames * capacity],
            slope: vec![0.; frames * capacity],
            amplitude: vec![0.; channels * frames * capacity],
            phase: vec![0.; channels * frames * capacity],
            amplitude_gradient: vec![0.; channels * frames],
            count: vec![0; frames],
            window_length: vec![0; frames],
        }
    }
    pub fn index(&self, c: usize, f: usize, k: usize) -> usize {
        (c * self.frames + f) * self.capacity + k
    }
}

/// All feature matrices use contiguous c,frame,component ordering.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Features {
    pub noise_envelope: Vec<f64>, // c,f
    pub noise_psd: Vec<f64>,      // c,f,npsd
    pub noise_psd_edges: Vec<f64>,
    pub noise_band_energy: Vec<f64>, // c,f,4
    pub noise_band_edges: Vec<f64>,  // 5
    pub noise_modulation: Vec<f64>,  // c,f,4,1+2*maxnhar_e
    pub spectral_cepstrum: Vec<f64>, // c,f,order_spec+1
    pub bap_cepstrum: Vec<f64>,      // c,f,order_bap+1
    pub lip_response: Vec<f64>,      // npsd
    pub mel_log_spectrum: Vec<f64>,  // c,f,64
    pub bap: Vec<f64>,               // c,f,5
    pub rd: Vec<f64>,                // c,f
    pub rd_confidence: Vec<f64>,     // c,f
    pub vector: Vec<f64>,            // c,f,72
}
