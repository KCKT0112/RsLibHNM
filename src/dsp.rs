use anyhow::{Result, ensure};
use num_complex::Complex64;
use rustfft::{Fft as RustFft, FftPlanner};
use std::sync::Arc;

use crate::types::{Audio, Spectrum};

/// A reusable real-input FFT wrapper.
///
/// The plans and working buffers are retained between calls.  `forward` and
/// `inverse` require exactly `n` samples; this keeps accidental partial FFTs
/// from silently changing the transform's normalization.
pub struct Fft {
    n: usize,
    forward_plan: Arc<dyn RustFft<f64>>,
    inverse_plan: Arc<dyn RustFft<f64>>,
    buffer: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "FFT length must be positive");
        let mut planner = FftPlanner::<f64>::new();
        let forward_plan = planner.plan_fft_forward(n);
        let inverse_plan = planner.plan_fft_inverse(n);
        let scratch_len = forward_plan
            .get_inplace_scratch_len()
            .max(inverse_plan.get_inplace_scratch_len());
        Self {
            n,
            forward_plan,
            inverse_plan,
            buffer: vec![Complex64::new(0.0, 0.0); n],
            scratch: vec![Complex64::new(0.0, 0.0); scratch_len],
        }
    }

    fn check_input(&self, input: &[Complex64]) {
        assert_eq!(input.len(), self.n, "FFT input length does not match plan");
        assert!(
            input.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
            "FFT input must be finite"
        );
    }

    /// Forward transform of a real signal, writing all `n` complex bins into `output`.
    /// The output buffer is reused by callers that perform many transforms.
    pub fn forward_into(&mut self, input: &[f64], output: &mut [Complex64]) {
        assert_eq!(input.len(), self.n, "FFT input length does not match plan");
        assert_eq!(
            output.len(),
            self.n,
            "FFT output length does not match plan"
        );
        assert!(
            input.iter().all(|x| x.is_finite()),
            "FFT input must be finite"
        );
        for (dst, &x) in output.iter_mut().zip(input) {
            *dst = Complex64::new(x, 0.0);
        }
        self.forward_plan
            .process_with_scratch(output, &mut self.scratch);
    }

    /// Forward transform of a real signal, returning all `n` complex bins.
    pub fn forward(&mut self, input: &[f64]) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); self.n];
        self.forward_into(input, &mut output);
        output
    }

    fn inverse_into_complex(&mut self, input: &[Complex64], output: &mut [Complex64]) {
        self.check_input(input);
        assert_eq!(
            output.len(),
            self.n,
            "FFT output length does not match plan"
        );
        output.copy_from_slice(input);
        self.inverse_plan
            .process_with_scratch(output, &mut self.scratch);
        let scale = 1.0 / self.n as f64;
        for z in output.iter_mut() {
            *z *= scale;
        }
    }

    /// Inverse transform into a reusable real output buffer.
    pub fn inverse_into(&mut self, input: &[Complex64], output: &mut [f64]) {
        assert_eq!(
            output.len(),
            self.n,
            "FFT output length does not match plan"
        );
        self.check_input(input);
        self.buffer.copy_from_slice(input);
        self.inverse_plan
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        let scale = 1.0 / self.n as f64;
        for (dst, z) in output.iter_mut().zip(&self.buffer) {
            *dst = (z * scale).re;
        }
    }

    fn inverse_complex(&mut self, input: &[Complex64]) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); self.n];
        self.inverse_into_complex(input, &mut output);
        output
    }

    /// Analytic-signal magnitude into a reusable output buffer.
    pub fn analytic_envelope_into(&mut self, input: &[f64], output: &mut [f64]) {
        assert_eq!(
            input.len(),
            self.n,
            "analytic-envelope input length does not match plan"
        );
        assert_eq!(
            output.len(),
            self.n,
            "analytic-envelope output length does not match plan"
        );
        assert!(
            input.iter().all(|x| x.is_finite()),
            "analytic-envelope input must be finite"
        );
        for (dst, &x) in self.buffer.iter_mut().zip(input) {
            *dst = Complex64::new(x, 0.0);
        }
        self.forward_plan
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        let positive = self.n.div_ceil(2);
        for (k, z) in self.buffer.iter_mut().enumerate() {
            let multiplier = if k == 0 || (self.n % 2 == 0 && k == self.n / 2) {
                1.0
            } else if k < positive {
                2.0
            } else {
                0.0
            };
            *z *= multiplier;
        }
        self.inverse_plan
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        let scale = 1.0 / self.n as f64;
        for (dst, z) in output.iter_mut().zip(&self.buffer) {
            *dst = (*z * scale).norm();
        }
    }
    /// Analytic-signal magnitude from an already transformed real spectrum.
    ///
    /// This avoids the forward transform required by `analytic_envelope_into`
    /// when a caller already has the frequency-domain band-pass result. The
    /// supplied spectrum is interpreted as the full two-sided FFT of a real
    /// signal, matching the Hilbert multiplier used by that method.
    pub(crate) fn analytic_envelope_from_spectrum_into(
        &mut self,
        input: &[Complex64],
        output: &mut [f64],
    ) {
        self.check_input(input);
        assert_eq!(
            output.len(),
            self.n,
            "analytic-envelope output length does not match plan"
        );
        self.buffer.copy_from_slice(input);
        let positive = self.n.div_ceil(2);
        for (k, z) in self.buffer.iter_mut().enumerate() {
            let multiplier = if k == 0 || (self.n % 2 == 0 && k == self.n / 2) {
                1.0
            } else if k < positive {
                2.0
            } else {
                0.0
            };
            *z *= multiplier;
        }
        self.inverse_plan
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        let scale = 1.0 / self.n as f64;
        for (dst, z) in output.iter_mut().zip(&self.buffer) {
            *dst = (*z * scale).norm();
        }
    }

    /// Inverse transform of a full complex spectrum, with `1/n` normalization.
    /// The returned values are the real component of the inverse transform.
    pub fn inverse(&mut self, input: &[Complex64]) -> Vec<f64> {
        self.inverse_complex(input)
            .into_iter()
            .map(|z| z.re)
            .collect()
    }

    /// Magnitude of the analytic signal obtained using the full-spectrum
    /// Hilbert-transform multiplier.
    pub fn analytic_envelope(&mut self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.; self.n];
        self.analytic_envelope_into(input, &mut output);
        output
    }
}

/// Symmetric Hann window matching `np.hanning(n)` in the reference implementation.
pub fn hann(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denominator = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / denominator).cos())
        .collect()
}

fn validate_stft_args(audio: &Audio, n_fft: usize, hop: usize) {
    assert!(n_fft > 0, "n_fft must be positive");
    assert!(hop > 0, "hop must be positive");
    audio.validate().expect("invalid audio for STFT");
}

/// Centered real STFT.  Spectrum data is contiguous in channel/frame/bin
/// order and contains the nonnegative-frequency bins only.
pub fn stft(audio: &Audio, n_fft: usize, hop: usize) -> Spectrum {
    validate_stft_args(audio, n_fft, hop);
    let samples = audio.samples();
    let channels = audio.channels;
    let left = n_fft / 2;
    let frames = samples.div_ceil(hop) + 1;
    let bins = n_fft / 2 + 1;
    let total = (frames - 1)
        .checked_mul(hop)
        .and_then(|x| x.checked_add(n_fft))
        .expect("STFT dimensions overflow");
    let output_len = channels
        .checked_mul(frames)
        .and_then(|x| x.checked_mul(bins))
        .expect("STFT dimensions overflow");
    let mut fft = Fft::new(n_fft);
    let window = hann(n_fft);
    let mut frame = vec![0.0; n_fft];
    let mut out = Vec::with_capacity(output_len);
    let mut transformed = vec![Complex64::new(0.0, 0.0); n_fft];
    for channel in 0..channels {
        for frame_idx in 0..frames {
            let start = frame_idx * hop;
            let signed_start = start as isize - left as isize;
            for (i, value) in frame.iter_mut().enumerate() {
                let source = signed_start + i as isize;
                *value = if source >= 0 && (source as usize) < samples {
                    audio.data[source as usize * channels + channel] * window[i]
                } else {
                    0.0
                };
            }
            fft.forward_into(&frame, &mut transformed);
            out.extend_from_slice(&transformed[..bins]);
        }
    }
    // `total` is deliberately computed above so an overflowing frame grid is
    // rejected even when all individual indexing operations happen to fit.
    let _ = total;
    Spectrum {
        channels,
        frames,
        bins,
        data: out,
    }
}

/// Inverse centered STFT using Hann-square overlap-add normalization.
pub fn istft(
    spec: &Spectrum,
    n_fft: usize,
    hop: usize,
    samples: usize,
    sample_rate: u32,
) -> Result<Audio> {
    ensure!(n_fft > 0, "n_fft must be positive");
    ensure!(hop > 0, "hop must be positive");
    ensure!(samples > 0, "sample count must be positive");
    ensure!(sample_rate >= 1000, "sample rate must be >=1000 Hz");
    ensure!(
        spec.channels > 0 && spec.frames > 0,
        "spectrum must contain channels and frames"
    );
    let bins = n_fft / 2 + 1;
    ensure!(spec.bins == bins, "spectrum has incompatible bin count");
    let expected_len = spec
        .channels
        .checked_mul(spec.frames)
        .and_then(|x| x.checked_mul(bins))
        .ok_or_else(|| anyhow::anyhow!("spectrum dimensions overflow"))?;
    ensure!(
        spec.data.len() == expected_len,
        "spectrum data length does not match shape"
    );
    ensure!(
        spec.data
            .iter()
            .all(|z| z.re.is_finite() && z.im.is_finite()),
        "spectrum must be finite"
    );
    let left = n_fft / 2;
    let total = (spec.frames - 1)
        .checked_mul(hop)
        .and_then(|x| x.checked_add(n_fft))
        .ok_or_else(|| anyhow::anyhow!("ISTFT dimensions overflow"))?;
    let valid_end = left
        .checked_add(samples)
        .ok_or_else(|| anyhow::anyhow!("requested sample range overflows"))?;
    ensure!(
        valid_end <= total,
        "requested samples exceed the STFT coverage"
    );

    let window = hann(n_fft);
    let window_square: Vec<f64> = window.iter().map(|w| w * w).collect();
    let output_len = spec
        .channels
        .checked_mul(total)
        .ok_or_else(|| anyhow::anyhow!("ISTFT dimensions overflow"))?;
    let data_len = samples
        .checked_mul(spec.channels)
        .ok_or_else(|| anyhow::anyhow!("requested output dimensions overflow"))?;
    let mut output = vec![0.0; output_len];
    let mut normalization = vec![0.0; total];
    let mut fft = Fft::new(n_fft);
    let mut full = vec![Complex64::new(0.0, 0.0); n_fft];
    let mut frame_time = vec![0.0; n_fft];
    for channel in 0..spec.channels {
        for frame_idx in 0..spec.frames {
            let base = (channel * spec.frames + frame_idx) * bins;
            full.fill(Complex64::new(0.0, 0.0));
            full[..bins].copy_from_slice(&spec.data[base..base + bins]);
            // Fill the negative-frequency half by conjugate symmetry.  The
            // Nyquist bin (when present) is its own conjugate and is skipped.
            for k in 1..bins {
                let mirror = n_fft - k;
                if mirror != k {
                    full[mirror] = full[k].conj();
                }
            }
            fft.inverse_into(&full, &mut frame_time);
            let start = frame_idx * hop;
            for i in 0..n_fft {
                output[channel * total + start + i] += frame_time[i] * window[i];
            }
            if channel == 0 {
                for i in 0..n_fft {
                    normalization[start + i] += window_square[i];
                }
            }
        }
    }
    let tiny = f64::MIN_POSITIVE;
    let mut data = vec![0.0; data_len];
    for i in 0..samples {
        let norm = normalization[left + i];
        ensure!(
            norm > tiny && norm.is_finite(),
            "STFT overlap-add has uncovered samples"
        );
        for channel in 0..spec.channels {
            data[i * spec.channels + channel] = output[channel * total + left + i] / norm;
        }
    }
    Ok(Audio {
        sample_rate,
        channels: spec.channels,
        data,
        encoding: "FLOAT64".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio(channels: usize, samples: usize) -> Audio {
        let mut data = Vec::with_capacity(channels * samples);
        for i in 0..samples {
            for c in 0..channels {
                data.push((i as f64 * 0.13 + c as f64 * 0.7).sin());
            }
        }
        Audio {
            sample_rate: 16_000,
            channels,
            data,
            encoding: "FLOAT64".into(),
        }
    }

    #[test]
    fn stft_frame_shape_and_channel_layout() {
        let input = audio(2, 17);
        let spec = stft(&input, 8, 4);
        assert_eq!((spec.channels, spec.frames, spec.bins), (2, 6, 5));
        assert_eq!(spec.data.len(), 2 * 6 * 5);
        assert_ne!(spec.data[0], spec.data[6 * 5]);
    }

    #[test]
    fn multichannel_round_trip_preserves_interleaved_layout() {
        let input = audio(2, 32);
        let spec = stft(&input, 8, 2);
        let output = istft(&spec, 8, 2, 32, 16_000).expect("valid overlap-add");
        assert_eq!(output.channels, 2);
        assert_eq!(output.data.len(), input.data.len());
        for (actual, expected) in output.data.iter().zip(input.data.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn impulse_round_trip() {
        let mut input = audio(1, 32);
        input.data.fill(0.0);
        input.data[16] = 1.0;
        let spec = stft(&input, 8, 2);
        let output = istft(&spec, 8, 2, 32, 16_000).expect("valid overlap-add");
        for (actual, expected) in output.data.iter().zip(input.data.iter()) {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
    }

    #[test]
    fn analytic_envelope_is_finite() {
        let n = 32;
        let input: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin())
            .collect();
        let mut fft = Fft::new(n);
        let envelope = fft.analytic_envelope(&input);
        assert!(envelope.iter().all(|x| x.is_finite() && *x >= 0.0));
    }
}
