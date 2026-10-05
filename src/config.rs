use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub n_fft: usize,
    pub hop_length: Option<usize>,
    pub vocal_f0_min: f64,
    pub vocal_f0_max: f64,
    pub maxnhar: usize,
    pub maxnhar_e: usize,
    pub npsd: usize,
    pub nchannel: usize,
    pub chanfreq: [f64; 3],
    pub order_spec: usize,
    pub order_bap: usize,
    pub rel_winsize: f64,
    pub lip_radius: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_length: None,
            vocal_f0_min: 70.,
            vocal_f0_max: 1200.,
            maxnhar: 120,
            maxnhar_e: 5,
            npsd: 256,
            nchannel: 4,
            chanfreq: [3000., 6000., 10000.],
            order_spec: 64,
            order_bap: 5,
            rel_winsize: 4.,
            lip_radius: 1.5,
        }
    }
}
impl Config {
    pub fn validate(&self, sample_rate: u32) -> Result<()> {
        ensure!(sample_rate >= 1000, "sample rate must be >=1000 Hz");
        ensure!(
            self.n_fft >= 128 && self.n_fft.is_power_of_two(),
            "n_fft must be a power of two >=128"
        );
        ensure!(
            (1..=512).contains(&self.maxnhar) && (1..=32).contains(&self.maxnhar_e),
            "invalid harmonic capacity"
        );
        ensure!(
            self.npsd >= 8 && self.nchannel == 4,
            "npsd >=8 and nchannel=4 required"
        );
        ensure!(
            self.order_spec == 64 && self.order_bap == 5,
            "72D contract requires order_spec=64 and order_bap=5"
        );
        ensure!(
            self.vocal_f0_min.is_finite()
                && self.vocal_f0_max.is_finite()
                && self.vocal_f0_min > 0.
                && self.vocal_f0_max > self.vocal_f0_min
                && self.vocal_f0_min < sample_rate as f64 / 2.,
            "invalid F0 range"
        );
        ensure!(
            self.chanfreq.iter().all(|v| v.is_finite() && *v > 0.)
                && self.chanfreq.windows(2).all(|w| w[0] < w[1]),
            "invalid noise band frequencies"
        );
        ensure!(
            (3. ..=12.).contains(&self.rel_winsize)
                && self.lip_radius.is_finite()
                && self.lip_radius > 0.,
            "invalid window or lip radius"
        );
        if let Some(h) = self.hop_length {
            ensure!(h > 0 && h <= self.n_fft / 2, "invalid hop length");
        }
        Ok(())
    }
    pub fn hop(&self, sample_rate: u32) -> usize {
        self.hop_length.unwrap_or_else(|| {
            ((sample_rate as f64 * 0.005).round_ties_even() as usize).clamp(1, self.n_fft / 2)
        })
    }
}
