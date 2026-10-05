use crate::{
    config::Config,
    dsp::Fft,
    envelope, harmonics,
    types::{Features, Harmonics},
};
use anyhow::{Result, ensure};
use num_complex::Complex64;
use rand::Rng;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};

fn render_noise(
    h: &Harmonics,
    features: &Features,
    config: &Config,
    sr: u32,
    samples: usize,
    hop: usize,
    f0_scale: f64,
    seed: u64,
) -> Result<Vec<f64>> {
    ensure!(
        f0_scale.is_finite() && f0_scale > 0.,
        "f0_scale must be finite and positive"
    );
    ensure!(
        features.noise_psd.len() == h.channels * h.frames * config.npsd,
        "noise PSD shape mismatch"
    );
    let nfft = config.n_fft;
    let bins = nfft / 2 + 1;
    let mut fft = Fft::new(nfft);
    let win = crate::dsp::hann(nfft);
    let mut out = vec![0.; samples * h.channels];
    let mut norm = vec![0.; samples];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let nyq = sr as f64 / 2.;
    let cells = &features.noise_psd_edges;
    let band_edges = &features.noise_band_edges;
    ensure!(
        cells.len() == config.npsd + 1 && band_edges.len() == 5,
        "noise edge shape mismatch"
    );
    let order =
        (features.noise_modulation.len() / (h.channels * h.frames * 4)).saturating_sub(1) / 2;
    let mut correction = vec![0.; h.frames];
    for j in 1..h.frames {
        correction[j] = correction[j - 1]
            + std::f64::consts::PI * (h.f0[j] + h.f0[j - 1]) * hop as f64 / sr as f64
                * (f0_scale - 1.);
    }
    for j in 0..h.frames {
        let center = j * hop;
        let half = nfft / 2;
        let a = center.saturating_sub(half);
        let b = (center + half + 1).min(samples);
        if b <= a {
            continue;
        }
        let mut raw = vec![0.; h.channels * nfft];
        for band in 0..4 {
            let low = band_edges[band];
            let high = band_edges[band + 1];
            if high <= low {
                continue;
            }
            for c in 0..h.channels {
                let target = features.noise_band_energy[(c * h.frames + j) * 4 + band];
                if !(target > 0. && target.is_finite()) {
                    continue;
                }
                let mut z = vec![Complex64::new(0., 0.); nfft];
                for k in 0..bins {
                    let fr = k as f64 * sr as f64 / nfft as f64;
                    if fr < low || fr >= high || (k == nfft / 2 && high < nyq) {
                        continue;
                    }
                    let pos = ((fr - cells[0]) / (cells[cells.len() - 1] - cells[0])
                        * (cells.len() - 1) as f64)
                        .clamp(0., (cells.len() - 2) as f64);
                    let cell = pos.floor() as usize;
                    let density =
                        features.noise_psd[(c * h.frames + j) * config.npsd + cell].max(0.);
                    if k == 0 || k == nfft / 2 {
                        z[k] = Complex64::new(
                            rng.sample::<f64, _>(rand_distr::StandardNormal)
                                * (density * sr as f64 * nfft as f64).sqrt(),
                            0.,
                        );
                    } else {
                        let sc = (density * sr as f64 * nfft as f64 / 4.).sqrt();
                        z[k] = Complex64::new(
                            rng.sample::<f64, _>(rand_distr::StandardNormal) * sc,
                            rng.sample::<f64, _>(rand_distr::StandardNormal) * sc,
                        );
                        z[nfft - k] = z[k].conj();
                    }
                }
                let mut frame = fft.inverse(&z);
                let coeff_base = ((c * h.frames + j) * 4 + band) * (1 + 2 * order);
                let amp = features
                    .noise_modulation
                    .get(coeff_base)
                    .copied()
                    .unwrap_or(0.);
                if amp > 1e-15 && h.f0[j] > 0. {
                    let phase0 = correction[j];
                    for (i, v) in frame.iter_mut().enumerate() {
                        let t = (i as f64 - half as f64) / sr as f64;
                        let phase = 2. * std::f64::consts::PI * h.f0[j] * f0_scale * t + phase0;
                        let mut m = 1.;
                        for k in 1..=order {
                            let co = features
                                .noise_modulation
                                .get(coeff_base + k)
                                .copied()
                                .unwrap_or(0.);
                            let si = features
                                .noise_modulation
                                .get(coeff_base + order + k)
                                .copied()
                                .unwrap_or(0.);
                            m += (co * (k as f64 * phase).cos() + si * (k as f64 * phase).sin())
                                / amp;
                        }
                        *v *= m.clamp(0., 4.);
                    }
                }
                let mut energy = 0.;
                for i in 0..nfft {
                    energy += frame[i] * frame[i] * win[i] * win[i];
                }
                energy /= win.iter().map(|x| x * x).sum::<f64>().max(1e-30);
                if energy > 1e-30 {
                    let gain = (target / energy).sqrt();
                    for i in 0..nfft {
                        raw[c * nfft + i] += frame[i] * gain;
                    }
                }
            }
        }
        for i in a..b {
            let q = i + half - center;
            if q < nfft {
                for c in 0..h.channels {
                    out[i * h.channels + c] += raw[c * nfft + q] * win[q];
                }
                norm[i] += win[q] * win[q];
            }
        }
    }
    for i in 0..samples {
        if norm[i] > 1e-20 {
            for c in 0..h.channels {
                out[i * h.channels + c] /= norm[i].sqrt();
            }
        } else {
            for c in 0..h.channels {
                out[i * h.channels + c] = 0.;
            }
        }
    }
    Ok(out)
}

/// Parameter-only HNM synthesis; no source waveform/STFT is consulted.
pub fn noise(
    h: &Harmonics,
    features: &Features,
    config: &Config,
    sr: u32,
    samples: usize,
    hop: usize,
    f0_scale: f64,
    seed: u64,
) -> Result<Vec<f64>> {
    render_noise(h, features, config, sr, samples, hop, f0_scale, seed)
}
pub fn synthesize(
    h: &Harmonics,
    features: &Features,
    config: &Config,
    sr: u32,
    samples: usize,
    hop: usize,
    f0_scale: f64,
    seed: u64,
    preserve_spectral_envelope: bool,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    ensure!(h.channels > 0 && h.frames > 0, "invalid harmonic model");
    let hh = if preserve_spectral_envelope {
        envelope::resample_harmonic_envelope(h, sr, f0_scale)?
    } else {
        h.clone()
    };
    let harmonic = harmonics::render(&hh, sr, samples, hop, f0_scale)?;
    let n = render_noise(h, features, config, sr, samples, hop, f0_scale, seed)?;
    let total = harmonic.iter().zip(&n).map(|(a, b)| a + b).collect();
    Ok((total, harmonic, n))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_scale() {
        let h = Harmonics::new(1, 1, 1);
        let f = Features::default();
        let c = Config::default();
        assert!(noise(&h, &f, &c, 16000, 10, 80, 0., 1).is_err());
    }
}
