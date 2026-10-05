use crate::config::Config;
use crate::types::Audio;
use rustfft::{FftPlanner, num_complex::Complex};

/// Estimate an initial F0 using normalized lag differences.  The selected
/// microphone is the one with the greatest total energy; microphones with
/// opposite polarity are deliberately not mixed.
pub fn estimate(audio: &Audio, config: &Config, hop: usize, frames: usize) -> (Vec<f64>, Vec<f64>) {
    let channels = audio.channels.max(1);
    let samples = audio.data.len() / channels;
    let sr = audio.sample_rate as f64;
    if channels == 0 || samples == 0 || frames == 0 || hop == 0 {
        return (vec![0.0; frames], vec![0.0; frames]);
    }
    let mut strongest = 0usize;
    let mut strongest_energy = 0.0;
    let mut max_abs: f64 = 0.0;
    for c in 0..channels {
        let mut e = 0.0;
        for n in 0..samples {
            let x = audio.data[n * channels + c];
            e += x * x;
            max_abs = max_abs.max(x.abs());
        }
        if e > strongest_energy {
            strongest_energy = e;
            strongest = c;
        }
    }
    let fmin = config.vocal_f0_min.max(1e-9);
    let fmax = config.vocal_f0_max.min(sr * 0.5 - 1e-9);
    if fmax <= fmin || strongest_energy <= 0.0 {
        return (vec![0.; frames], vec![0.; frames]);
    }
    let lo = (sr / fmax).floor().max(2.0) as usize;
    let hi = (sr / fmin).ceil() as usize;
    let mut length = (2 * hi + 3).max((0.045 * sr).round() as usize);
    if length % 2 == 0 {
        length += 1;
    }
    let half = length / 2;
    let fft_len = (2 * length - 1).next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(fft_len);
    let inverse = planner.plan_fft_inverse(fft_len);
    let mut buf = vec![Complex::new(0.0, 0.0); fft_len];
    // These workspaces have fixed dimensions for the whole analysis. Reusing
    // them avoids one allocation per frame without changing any arithmetic.
    let mut y = vec![0.0; length];
    let mut cumulative = vec![0.0; length + 1];
    let mut nd = vec![1.0; hi.max(1)];
    let mut minima = Vec::with_capacity(hi.saturating_sub(lo).max(1));
    let energy_floor = (max_abs * 1e-5).max(1e-12);
    let mut f0 = vec![0.0; frames];
    let mut confidence = vec![0.0; frames];
    for frame in 0..frames {
        let center = frame.saturating_mul(hop) as isize;
        let start = center - half as isize;
        y.fill(0.0);
        for i in 0..length {
            let n = start + i as isize;
            if n >= 0 && (n as usize) < samples {
                y[i] = audio.data[n as usize * channels + strongest];
            }
        }
        let mean = y.iter().sum::<f64>() / length as f64;
        for x in &mut y {
            *x -= mean;
        }
        let rms = (y.iter().map(|x| x * x).sum::<f64>() / length as f64).sqrt();
        if rms < energy_floor {
            continue;
        }
        buf.fill(Complex::new(0.0, 0.0));
        for (z, x) in buf.iter_mut().zip(y.iter()) {
            z.re = *x;
        }
        forward.process(&mut buf);
        for z in &mut buf {
            *z *= z.conj();
        }
        inverse.process(&mut buf);
        let scale = fft_len as f64;
        cumulative[0] = 0.0;
        for i in 0..length {
            cumulative[i + 1] = cumulative[i] + y[i] * y[i];
        }
        nd.fill(1.0);
        let max_lag = hi.min(length.saturating_sub(1));
        for lag in 1..=max_lag {
            let overlap_e = cumulative[length - lag] + cumulative[length] - cumulative[lag];
            let ac = buf[lag].re / scale;
            let diff = (overlap_e - 2.0 * ac).max(0.0);
            nd[lag - 1] = if overlap_e > 1e-25 {
                diff / overlap_e
            } else {
                1.0
            };
        }
        minima.clear();
        let end = max_lag.min(hi.saturating_sub(1));
        for lag in lo.max(2)..end {
            if lag >= 2 && nd[lag - 1] <= nd[lag - 2] && nd[lag - 1] < nd[lag] {
                minima.push(lag);
            }
        }
        if minima.is_empty() {
            continue;
        }
        let chosen = minima
            .iter()
            .copied()
            .find(|&lag| nd[lag - 1] < 0.18)
            .unwrap_or_else(|| {
                *minima
                    .iter()
                    .min_by(|a, b| nd[**a - 1].partial_cmp(&nd[**b - 1]).unwrap())
                    .unwrap()
            });
        let value = nd[chosen - 1];
        if value > 0.32 || chosen < 2 || chosen + 1 > nd.len() {
            continue;
        }
        let ym = nd[chosen - 2];
        let y0 = nd[chosen - 1];
        let yp = nd[chosen];
        let curvature = ym - 2.0 * y0 + yp;
        let offset = (0.5 * (ym - yp) / curvature.max(1e-15)).clamp(-0.5, 0.5);
        let freq = sr / (chosen as f64 + offset);
        if freq >= config.vocal_f0_min && freq <= config.vocal_f0_max {
            f0[frame] = freq;
            confidence[frame] = (1.0 - value).clamp(0.0, 1.0);
        }
    }
    (f0, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn estimates_sine_pitch_and_energy_channel() {
        let sr = 8000u32;
        let n = 8000usize;
        let mut data = vec![0.0; n * 2];
        for i in 0..n {
            data[2 * i + 1] = (2.0 * std::f64::consts::PI * 200.0 * i as f64 / sr as f64).sin();
        }
        let audio = Audio {
            sample_rate: sr,
            channels: 2,
            data,
            encoding: "FLOAT64".into(),
        };
        let mut cfg = Config::default();
        cfg.vocal_f0_min = 100.;
        cfg.vocal_f0_max = 400.;
        let (f, c) = estimate(&audio, &cfg, 40, 10);
        assert!(f.iter().any(|x| (*x - 200.).abs() < 5.));
        assert!(c.iter().any(|x| *x > 0.5));
    }
}
