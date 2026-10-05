use crate::types::Harmonics;
use anyhow::{Result, ensure};

/// Interpolate measured log harmonic magnitudes at shifted absolute frequencies.
pub fn resample_harmonic_envelope(
    h: &Harmonics,
    sample_rate: u32,
    f0_scale: f64,
) -> Result<Harmonics> {
    ensure!(
        f0_scale.is_finite() && f0_scale > 0.0,
        "f0_scale must be finite and positive"
    );
    let mut out = h.clone();
    if f0_scale == 1.0 {
        return Ok(out);
    }
    let nyquist = sample_rate as f64 * 0.5;
    for frame in 0..h.frames {
        if !h.f0[frame].is_finite() || h.f0[frame] <= 0.0 {
            continue;
        }
        let count = h.count[frame].min(h.capacity);
        let valid: Vec<usize> = (0..count)
            .filter(|&k| {
                let f = h.frequency[frame * h.capacity + k];
                f.is_finite() && f > 0.0 && f < nyquist
            })
            .collect();
        if valid.len() < 4 {
            continue;
        }
        for channel in 0..h.channels {
            let peak = valid
                .iter()
                .map(|&k| h.amplitude[h.index(channel, frame, k)].max(0.0))
                .fold(0.0_f64, f64::max);
            if peak <= 0.0 {
                continue;
            }
            // Ignore tiny numerical partials before deciding if a spectral envelope exists.
            let valid: Vec<usize> = valid
                .iter()
                .copied()
                .filter(|&k| h.amplitude[h.index(channel, frame, k)] >= peak * 1e-3)
                .collect();
            if valid.len() < 4 {
                continue;
            }
            let frequencies: Vec<f64> = valid
                .iter()
                .map(|&k| h.frequency[frame * h.capacity + k])
                .collect();
            if frequencies.windows(2).any(|p| p[1] <= p[0]) {
                continue;
            }
            let logs: Vec<f64> = valid
                .iter()
                .map(|&k| {
                    (h.amplitude[h.index(channel, frame, k)] / peak)
                        .max(1e-6)
                        .ln()
                })
                .collect();
            let spacing = frequencies.windows(2).map(|p| p[1] - p[0]).sum::<f64>()
                / (frequencies.len() - 1) as f64;
            let mut edited = vec![0.0; count];
            for k in 0..count {
                let target = h.frequency[frame * h.capacity + k] * f0_scale;
                if !(target > 0.0 && target < nyquist) {
                    continue;
                }
                let pos = frequencies.partition_point(|f| *f < target);
                let logmag = if pos == 0 {
                    logs[0]
                } else if pos == frequencies.len() {
                    if target > frequencies[pos - 1] + spacing {
                        continue;
                    }
                    logs[pos - 1]
                } else {
                    let lo = pos - 1;
                    let t = (target - frequencies[lo]) / (frequencies[pos] - frequencies[lo]);
                    logs[lo] * (1.0 - t) + logs[pos] * t
                };
                edited[k] = logmag.exp() * peak;
            }
            let reference = valid
                .iter()
                .map(|&k| h.amplitude[h.index(channel, frame, k)].powi(2))
                .sum::<f64>()
                .sqrt();
            let energy = edited.iter().map(|v| v * v).sum::<f64>().sqrt();
            let gain = if energy > 0.0 {
                (reference / energy).clamp(0.5, 2.0)
            } else {
                1.0
            };
            for k in 0..count {
                out.amplitude[h.index(channel, frame, k)] = edited[k] * gain;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scale_one_is_exact() {
        let h = Harmonics::new(1, 1, 2);
        let x = resample_harmonic_envelope(&h, 16000, 1.0).unwrap();
        assert_eq!(x.amplitude, h.amplitude);
    }
    #[test]
    fn sparse_tone_falls_back_without_exploding_residual_partials() {
        let mut h = Harmonics::new(1, 1, 12);
        h.f0[0] = 200.0;
        h.count[0] = 12;
        for k in 0..12 {
            h.frequency[k] = 200.0 * (k + 1) as f64;
            h.amplitude[k] = if k == 4 { 1.0 } else { 1e-12 };
        }
        let y = resample_harmonic_envelope(&h, 16000, 1.25).unwrap();
        assert_eq!(y.amplitude, h.amplitude);
    }
}
