use crate::config::Config;
use crate::types::{Audio, Harmonics};
use anyhow::{Result, ensure};
use nalgebra::DMatrix;
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;

// Row-major scratch matrices keep the hot frame path out of nalgebra's dynamic
// allocation/multiplication machinery. `stride` allows the basis to shrink
// from the search shape to its final harmonic count without moving storage.
struct FlatMatrix {
    rows: usize,
    cols: usize,
    stride: usize,
    data: Vec<f64>,
}

impl FlatMatrix {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            stride: cols,
            data: vec![0.; rows * cols],
        }
    }
    fn with_capacity(rows: usize, capacity: usize) -> Self {
        Self {
            rows,
            cols: capacity,
            stride: capacity,
            data: vec![0.; rows * capacity],
        }
    }
    #[inline]
    fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.stride + col]
    }
    #[inline]
    fn set_cols(&mut self, cols: usize) {
        if cols > self.stride {
            self.stride = cols;
            self.data.resize(self.rows * cols, 0.);
        }
        self.cols = cols;
    }
}

fn fill_basis(a: &mut FlatMatrix, t: &[f64], f0: f64, slope: f64, count: usize) {
    a.set_cols(count * 2);
    for (i, &time) in t.iter().enumerate() {
        let carrier = 2.0 * PI * (f0 * time + 0.5 * slope * time * time);
        let row = i * a.stride;
        for k in 0..count {
            let ph = carrier * (k + 1) as f64;
            a.data[row + k] = ph.cos();
            a.data[row + count + k] = ph.sin();
        }
    }
}

struct SolveScratch {
    gram: Vec<f64>,
    rhs: Vec<f64>,
    factor: Vec<f64>,
    solution: Vec<f64>,
}

impl SolveScratch {
    fn new(cols: usize, outputs: usize) -> Self {
        Self {
            gram: vec![0.; cols * cols],
            rhs: vec![0.; cols * outputs],
            factor: vec![0.; cols * cols],
            solution: vec![0.; cols * outputs],
        }
    }
}

// Flat-buffer Cholesky for the positive-definite Gram matrix. Invalid pivots
// take the unchanged nalgebra SVD fallback in the caller.
fn cholesky_solve(
    matrix: &[f64],
    rhs: &[f64],
    n: usize,
    outputs: usize,
    factor: &mut [f64],
    out: &mut [f64],
) -> bool {
    factor.copy_from_slice(matrix);
    for i in 0..n {
        for j in 0..=i {
            let mut value = factor[i * n + j];
            for k in 0..j {
                value -= factor[i * n + k] * factor[j * n + k];
            }
            if i == j {
                if !value.is_finite() || value <= 0. {
                    return false;
                }
                factor[i * n + j] = value.sqrt();
            } else {
                factor[i * n + j] = value / factor[j * n + j];
            }
        }
    }
    for i in 0..n {
        for c in 0..outputs {
            let mut value = rhs[i * outputs + c];
            for k in 0..i {
                value -= factor[i * n + k] * out[k * outputs + c];
            }
            out[i * outputs + c] = value / factor[i * n + i];
        }
    }
    for i in (0..n).rev() {
        for c in 0..outputs {
            let mut value = out[i * outputs + c];
            for k in i + 1..n {
                value -= factor[k * n + i] * out[k * outputs + c];
            }
            out[i * outputs + c] = value / factor[i * n + i];
        }
    }
    true
}

fn solve_with_scratch(a: &FlatMatrix, y: &FlatMatrix, weight: &[f64], scratch: &mut SolveScratch) {
    let (rows, cols, outputs) = (a.rows, a.cols, y.cols);
    debug_assert_eq!(y.rows, rows);
    debug_assert_eq!(scratch.gram.len(), cols * cols);
    debug_assert_eq!(scratch.rhs.len(), cols * outputs);
    scratch.gram.fill(0.);
    scratch.rhs.fill(0.);
    for i in 0..rows {
        let w2 = weight[i] * weight[i];
        let ar = i * a.stride;
        let yr = i * y.stride;
        for p in 0..cols {
            let ap = a.data[ar + p];
            for q in 0..=p {
                scratch.gram[p * cols + q] += w2 * ap * a.data[ar + q];
            }
            for c in 0..outputs {
                scratch.rhs[p * outputs + c] += w2 * ap * y.data[yr + c];
            }
        }
    }
    for p in 0..cols {
        for q in 0..p {
            scratch.gram[q * cols + p] = scratch.gram[p * cols + q];
        }
    }
    if cholesky_solve(
        &scratch.gram,
        &scratch.rhs,
        cols,
        outputs,
        &mut scratch.factor,
        &mut scratch.solution,
    ) {
        return;
    }
    // Preserve SVD behavior for singular/indefinite Gram matrices; this path is
    // cold for ordinary frames and retains the original 1e-8 tolerance.
    let mut aw = DMatrix::zeros(rows, cols);
    let mut yw = DMatrix::zeros(rows, outputs);
    for i in 0..rows {
        let ar = i * a.stride;
        let yr = i * y.stride;
        for q in 0..cols {
            aw[(i, q)] = a.data[ar + q] * weight[i];
        }
        for c in 0..outputs {
            yw[(i, c)] = y.data[yr + c] * weight[i];
        }
    }
    if let Ok(solution) = aw.svd(true, true).solve(&yw, 1e-8) {
        for p in 0..cols {
            for c in 0..outputs {
                scratch.solution[p * outputs + c] = solution[(p, c)];
            }
        }
    } else {
        scratch.solution.fill(0.);
    }
}

fn fit_envelope(a: &FlatMatrix, y: &FlatMatrix, w: &[f64], u: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (channels, rows, cols) = (y.cols, a.rows, a.cols);
    let mut gradient = vec![0.; channels];
    let mut solve_scratch = SolveScratch::new(cols, channels);
    solve_with_scratch(a, y, w, &mut solve_scratch);
    let mut coeff = solve_scratch.solution.clone();
    let (mut g0, mut g1, mut g2) = (
        vec![0.; cols * cols],
        vec![0.; cols * cols],
        vec![0.; cols * cols],
    );
    let (mut r0, mut r1) = (vec![0.; cols * channels], vec![0.; cols * channels]);
    for i in 0..rows {
        let w2 = w[i] * w[i];
        let ar = i * a.stride;
        let yr = i * y.stride;
        for p in 0..cols {
            for q in 0..=p {
                let aa = w2 * a.data[ar + p] * a.data[ar + q];
                g0[p * cols + q] += aa;
                g1[p * cols + q] += aa * u[i];
                g2[p * cols + q] += aa * u[i] * u[i];
            }
            for c in 0..channels {
                let ay = w2 * a.data[ar + p] * y.data[yr + c];
                r0[p * channels + c] += ay;
                r1[p * channels + c] += ay * u[i];
            }
        }
    }
    for p in 0..cols {
        for q in 0..p {
            g0[q * cols + p] = g0[p * cols + q];
            g1[q * cols + p] = g1[p * cols + q];
            g2[q * cols + p] = g2[p * cols + q];
        }
    }
    let mut carrier = vec![0.; rows];
    let (mut normal, mut normal_factor) = (vec![0.; cols * cols], vec![0.; cols * cols]);
    let (mut normal_rhs, mut channel) = (vec![0.; cols], vec![0.; cols]);
    let mut fallback_a = FlatMatrix::new(rows, cols);
    let mut fallback_target = FlatMatrix::new(rows, 1);
    let mut fallback_scratch = SolveScratch::new(cols, 1);
    for c in 0..channels {
        for k in 0..cols {
            channel[k] = coeff[k * channels + c];
        }
        for _ in 0..5 {
            for i in 0..rows {
                let mut value = 0.;
                for k in 0..cols {
                    value += a.data[i * a.stride + k] * channel[k];
                }
                carrier[i] = value;
            }
            let g = bounded_minimize(
                |g| {
                    let mut err = 0.;
                    for i in 0..rows {
                        let d = w[i]
                            * (y.data[i * y.stride + c] - carrier[i] * (1. + g * u[i]).max(0.));
                        err += d * d;
                    }
                    err
                },
                -4.,
                4.,
            );
            let env_min = u.iter().map(|x| 1. + g * x).fold(f64::INFINITY, f64::min);
            let solved = if env_min > 0.05 {
                for p in 0..cols {
                    for q in 0..cols {
                        normal[p * cols + q] =
                            g0[p * cols + q] + 2. * g * g1[p * cols + q] + g * g * g2[p * cols + q];
                    }
                    normal_rhs[p] = r0[p * channels + c] + g * r1[p * channels + c];
                }
                cholesky_solve(
                    &normal,
                    &normal_rhs,
                    cols,
                    1,
                    &mut normal_factor,
                    &mut channel,
                )
            } else {
                false
            };
            if !solved {
                for i in 0..rows {
                    let e = (1. + g * u[i]).max(0.);
                    for k in 0..cols {
                        fallback_a.data[i * fallback_a.stride + k] = a.data[i * a.stride + k] * e;
                    }
                    fallback_target.data[i] = y.data[i * y.stride + c];
                }
                solve_with_scratch(&fallback_a, &fallback_target, w, &mut fallback_scratch);
                channel.copy_from_slice(&fallback_scratch.solution);
            }
            gradient[c] = g;
        }
        for k in 0..cols {
            coeff[k * channels + c] = channel[k];
        }
    }
    (coeff, gradient)
}

fn bounded_minimize(mut f: impl FnMut(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
    let ratio = 0.6180339887498949;
    let mut x1 = hi - ratio * (hi - lo);
    let mut x2 = lo + ratio * (hi - lo);
    let mut y1 = f(x1);
    let mut y2 = f(x2);
    for _ in 0..25 {
        if y1 > y2 {
            lo = x1;
            x1 = x2;
            y1 = y2;
            x2 = lo + ratio * (hi - lo);
            y2 = f(x2);
        } else {
            hi = x2;
            x2 = x1;
            y2 = y1;
            x1 = hi - ratio * (hi - lo);
            y1 = f(x1);
        }
    }
    if y1 < y2 { x1 } else { x2 }
}

fn frame_window(
    center: usize,
    length: usize,
    samples: usize,
    sr: f64,
) -> (usize, usize, Vec<f64>, Vec<f64>) {
    let half = length / 2;
    let a = center.saturating_sub(half);
    let b = center.saturating_add(half).saturating_add(1).min(samples);
    if b <= a {
        return (a, a, Vec::new(), Vec::new());
    }
    let mut t = Vec::with_capacity(b - a);
    let mut w = Vec::with_capacity(b - a);
    for n in a..b {
        let x = (n as f64 - center as f64) / sr;
        t.push(x);
        w.push((0.5 + 0.5 * (PI * x / (half as f64 / sr)).cos()).max(0.0));
    }
    (a, b, t, w)
}

fn render_internal(h: &Harmonics, sr: f64, samples: usize, hop: usize, scale: f64) -> Vec<f64> {
    let mut out = vec![0.; samples * h.channels];
    let mut norm = vec![0.; samples];
    let mut correction = vec![0.; h.frames];
    for j in 1..h.frames {
        correction[j] =
            correction[j - 1] + PI * (h.f0[j] + h.f0[j - 1]) * hop as f64 / sr * (scale - 1.0);
    }
    for j in 0..h.frames {
        let len = h.window_length[j].max(3) | 1;
        let (a, b, t, w) = frame_window(j * hop, len, samples, sr);
        let count = h.count[j].min(h.capacity);
        if b <= a {
            continue;
        }
        for i in a..b {
            norm[i] += w[i - a];
        }
        for n in 0..(b - a) {
            let time = t[n];
            for c in 0..h.channels {
                let env = (1.0 + time * h.amplitude_gradient[c * h.frames + j]).max(0.0);
                let mut s = 0.;
                for k in 0..count {
                    let fr = h.frequency[j * h.capacity + k];
                    let sl = h.slope[j * h.capacity + k];
                    if (fr + sl.abs() * (len / 2) as f64 / sr) * scale >= sr / 2. {
                        continue;
                    }
                    let angle = 2. * PI * scale * (fr * time + 0.5 * sl * time * time)
                        + (k + 1) as f64 * correction[j];
                    let ix = h.index(c, j, k);
                    s += h.amplitude[ix] * env * (angle + h.phase[ix]).cos();
                }
                out[(a + n) * h.channels + c] += w[n] * s;
            }
        }
    }
    for n in 0..samples {
        if norm[n] > 1e-12 {
            for c in 0..h.channels {
                out[n * h.channels + c] /= norm[n];
            }
        }
    }
    // Remove window tails inside explicit unvoiced consonant gaps; the residual
    // receives this energy. Linear interpolation avoids a hard sample jump.
    for n in 0..samples {
        let pos = n as f64 / hop as f64;
        let left = pos.floor() as usize;
        let right = (left + 1).min(h.frames.saturating_sub(1));
        let frac = pos - left as f64;
        let lv = if h.f0.get(left).copied().unwrap_or(0.0) > 0.0 {
            1.0
        } else {
            0.0
        };
        let rv = if h.f0.get(right).copied().unwrap_or(0.0) > 0.0 {
            1.0
        } else {
            0.0
        };
        let confidence = h.confidence.get(left).copied().unwrap_or(0.0)
            + (h.confidence.get(right).copied().unwrap_or(0.0)
                - h.confidence.get(left).copied().unwrap_or(0.0))
                * frac;
        let periodicity = ((confidence - 0.55) / 0.35).clamp(0.0, 1.0);
        let mask = (lv + (rv - lv) * frac) * periodicity;
        for c in 0..h.channels {
            out[n * h.channels + c] *= mask;
        }
    }
    out
}

/// Results of fitting one frame, kept independent so frames can be solved in parallel.
struct FrameFit {
    f0: f64,
    confidence: f64,
    frequency: Vec<f64>,
    slope: Vec<f64>,
    amplitude: Vec<f64>,
    phase: Vec<f64>,
    amplitude_gradient: Vec<f64>,
    count: usize,
    window_length: usize,
}

impl FrameFit {
    fn empty(channels: usize, capacity: usize, confidence: f64) -> Self {
        Self {
            f0: 0.,
            confidence,
            frequency: vec![0.; capacity],
            slope: vec![0.; capacity],
            amplitude: vec![0.; channels * capacity],
            phase: vec![0.; channels * capacity],
            amplitude_gradient: vec![0.; channels],
            count: 0,
            window_length: 0,
        }
    }
}

/// Fit center-referenced local chirps with variable-projection F0/slope refinement.
pub fn fit(
    audio: &Audio,
    f0_in: &[f64],
    confidence_in: &[f64],
    hop: usize,
    config: &Config,
) -> Result<(Harmonics, Vec<f64>)> {
    ensure!(
        audio.channels > 0 && hop > 0,
        "invalid harmonic fit dimensions"
    );
    ensure!(
        f0_in.len() == confidence_in.len(),
        "pitch/confidence length mismatch"
    );
    let channels = audio.channels;
    let samples = audio.samples();
    let frames = f0_in.len();
    let cap = config.maxnhar;
    let mut h = Harmonics::new(channels, frames, cap);
    h.f0 = f0_in.to_vec();
    h.confidence = confidence_in.to_vec();
    let sr = audio.sample_rate as f64;

    // The strongest channel is shared by every frame and is selected before the
    // parallel region, preserving the original strict-greater tie breaking.
    let mut strongest = 0;
    let mut maxe = 0.;
    for c in 0..channels {
        let e = (0..samples)
            .map(|n| {
                let x = audio.data[n * channels + c];
                x * x
            })
            .sum::<f64>();
        if e > maxe {
            maxe = e;
            strongest = c;
        }
    }

    // Every frame reads only immutable input and writes its own result.  Rayon
    // collects by source-index order, so assembly below is deterministic even
    // though frame completion order is not.
    let frame_results: Vec<FrameFit> = (0..frames)
        .into_par_iter()
        .map(|j| {
            let initial = f0_in[j];
            let mut result = FrameFit::empty(channels, cap, confidence_in[j]);
            if !initial.is_finite() || initial <= 0. {
                return result;
            }

            let length =
                ((config.rel_winsize * sr / initial).round() as usize).max(2 * hop + 1) | 1;
            let center = j * hop;
            let (a, b, t, w) = frame_window(center, length, samples, sr);
            if b - a < 8 {
                result.confidence = 0.;
                return result;
            }
            let valid = w.iter().filter(|x| **x > 0.05).count();
            let maxcount = cap
                .min(((sr / 2. - 1.) / (initial * 1.07)).floor().max(1.) as usize)
                .min((valid / 4).max(1));
            if maxcount < 1 {
                result.confidence = 0.;
                return result;
            }

            // Unlike the old serial loop, the initial slope uses the supplied
            // pitch track.  This removes a hidden dependence on the preceding
            // frame's refined value and makes each frame genuinely independent;
            // the bounded variable-projection search still performs the same
            // slope refinement for every frame.
            let mut slope0 = 0.;
            if j > 0 && j + 1 < frames && f0_in[j - 1] > 0. && f0_in[j + 1] > 0. {
                slope0 = (f0_in[j + 1] - f0_in[j - 1]) * sr / (2. * hop as f64);
            }
            let mut target = FlatMatrix::new(b - a, 1);
            for i in 0..b - a {
                target.data[i] = audio.data[(a + i) * channels + strongest];
            }
            let fitcount = maxcount.min(12);
            let span = (length / 2) as f64 / sr;
            // Every search evaluation has identical dimensions. Fill one
            // matrix in place rather than allocating a new design matrix for
            // each of the 4x10 variable-projection evaluations.
            let mut aa = FlatMatrix::with_capacity(b - a, fitcount * 2);
            aa.set_cols(fitcount * 2);
            let mut solve_scratch = SolveScratch::new(fitcount * 2, 1);
            let mut eval = |ff: f64, ss: f64| {
                fill_basis(&mut aa, &t, ff, ss, fitcount);
                solve_with_scratch(&aa, &target, &w, &mut solve_scratch);
                let mut e = 0.;
                for i in 0..aa.rows {
                    let d = w[i]
                        * (target.data[i]
                            - (0..aa.cols)
                                .map(|k| aa.at(i, k) * solve_scratch.solution[k])
                                .sum::<f64>());
                    e += d * d;
                }
                e
            };
            let energy = (0..b - a)
                .map(|i| {
                    let z = w[i] * target.data[i];
                    z * z
                })
                .sum::<f64>();
            if energy < 1e-22 {
                result.confidence = 0.;
                return result;
            }

            let mut ff = initial.clamp(config.vocal_f0_min, config.vocal_f0_max);
            let mut ss = slope0;
            let flo = config.vocal_f0_min.max(initial * 0.94);
            let fhi = config.vocal_f0_max.min(initial * 1.06);
            let slo = -initial * 0.15 / span;
            let shi = initial * 0.15 / span;
            let mut stepf = (fhi - flo) / 4.;
            let mut steps = (shi - slo) / 4.;
            for _ in 0..4 {
                let mut best = eval(ff, ss);
                let mut bf = ff;
                let mut bs = ss;
                for di in -1..=1 {
                    for ds in -1..=1 {
                        let qf = (ff + di as f64 * stepf).clamp(flo, fhi);
                        let qs = (ss + ds as f64 * steps).clamp(slo, shi);
                        let e = eval(qf, qs);
                        if e < best {
                            best = e;
                            bf = qf;
                            bs = qs;
                        }
                    }
                }
                ff = bf;
                ss = bs;
                stepf *= 0.5;
                steps *= 0.5;
            }

            let count = cap
                .min(((sr / 2. - 1.) / (ff + ss.abs() * span)).floor().max(1.) as usize)
                .min((valid / 4).max(1));
            drop(eval);
            fill_basis(&mut aa, &t, ff, ss, count);
            result.f0 = ff;
            result.frequency[..count]
                .iter_mut()
                .enumerate()
                .for_each(|(k, x)| *x = ff * (k + 1) as f64);
            result.slope[..count]
                .iter_mut()
                .enumerate()
                .for_each(|(k, x)| *x = ss * (k + 1) as f64);
            result.count = count;
            result.window_length = length;

            // The normalized time is also reused by every envelope iteration.
            let u: Vec<f64> = t.iter().map(|x| x / span).collect();
            let mut yy = FlatMatrix::new(b - a, channels);
            for i in 0..b - a {
                for c in 0..channels {
                    yy.data[i * channels + c] = audio.data[(a + i) * channels + c];
                }
            }
            let (allcoeff, allgr) = fit_envelope(&aa, &yy, &w, &u);
            for c in 0..channels {
                result.amplitude_gradient[c] = allgr[c] / span;
                for k in 0..count {
                    let coeff_cos = allcoeff[k * channels + c];
                    let coeff_sin = allcoeff[(count + k) * channels + c];
                    let ix = c * cap + k;
                    result.amplitude[ix] = (coeff_cos.powi(2) + coeff_sin.powi(2)).sqrt();
                    result.phase[ix] = (-coeff_sin).atan2(coeff_cos);
                }
            }
            result
        })
        .collect();

    for (j, result) in frame_results.into_iter().enumerate() {
        h.f0[j] = result.f0;
        h.confidence[j] = result.confidence;
        h.frequency[j * cap..(j + 1) * cap].copy_from_slice(&result.frequency);
        h.slope[j * cap..(j + 1) * cap].copy_from_slice(&result.slope);
        h.count[j] = result.count;
        h.window_length[j] = result.window_length;
        for c in 0..channels {
            h.amplitude_gradient[c * frames + j] = result.amplitude_gradient[c];
            let dst = (c * frames + j) * cap;
            let src = c * cap;
            h.amplitude[dst..dst + cap].copy_from_slice(&result.amplitude[src..src + cap]);
            h.phase[dst..dst + cap].copy_from_slice(&result.phase[src..src + cap]);
        }
    }

    stabilize(&mut h, audio, hop);
    truncate_internal_gap_windows(&mut h, hop);
    let rendered = render_internal(&h, sr, samples, hop, 1.0);
    Ok((h, rendered))
}

fn stabilize(h: &mut Harmonics, audio: &Audio, hop: usize) {
    if h.frames < 5 {
        return;
    }
    let sr = audio.sample_rate as f64;
    let raw = render_internal(h, sr, audio.samples(), hop, 1.);
    let mut residual = vec![0.; audio.samples()];
    for n in 0..audio.samples() {
        for c in 0..audio.channels {
            let d = audio.data[n * audio.channels + c] - raw[n * audio.channels + c];
            residual[n] += d * d;
        }
    }
    let old = h.amplitude.clone();
    let radius = 2;
    for j in radius..h.frames - radius {
        if (j - radius..=j + radius).any(|q| h.f0[q] <= 0.) {
            continue;
        }
        let count = (j - radius..=j + radius)
            .map(|q| h.count[q])
            .min()
            .unwrap_or(0);
        for c in 0..h.channels {
            for k in 0..count {
                let mut z = Complex64::new(0., 0.);
                for q in j - radius..=j + radius {
                    let dt = -(q as f64 - j as f64) * hop as f64 / sr;
                    let ix = h.index(c, q, k);
                    let ph = 2.
                        * PI
                        * (h.frequency[q * h.capacity + k] * dt
                            + 0.5 * h.slope[q * h.capacity + k] * dt * dt);
                    z += Complex64::from_polar(
                        h.amplitude[ix]
                            * (1. + h.amplitude_gradient[c * h.frames + q] * dt).max(0.),
                        h.phase[ix] + ph,
                    );
                }
                z /= (2 * radius + 1) as f64;
                let ix = h.index(c, j, k);
                let diff = z - Complex64::from_polar(h.amplitude[ix], h.phase[ix]);
                let p = residual[(j * hop).min(residual.len() - 1)];
                let gain = (p * 8.
                    / (h.window_length[j].max(1) as f64 * 0.5)
                    / diff.norm_sqr().max(1e-30))
                .min(1.);
                let cleaned = Complex64::from_polar(old[ix], h.phase[ix]) + diff * gain;
                h.amplitude[ix] = cleaned.norm();
                h.phase[ix] = cleaned.arg();
            }
        }
    }
}

fn truncate_internal_gap_windows(h: &mut Harmonics, hop: usize) {
    let voiced: Vec<bool> = h.f0.iter().map(|x| x.is_finite() && *x > 0.).collect();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    for i in 0..=h.frames {
        let before = i > 0 && voiced[i - 1];
        let after = i < h.frames && voiced[i];
        if after && !before {
            starts.push(i);
        }
        if before && !after {
            ends.push(i);
        }
    }
    for run in 0..starts.len().min(ends.len()) {
        let first = starts[run];
        let end = ends[run];
        if run == 0 && run + 1 == starts.len() {
            continue;
        }
        for j in first..end {
            let distance = (j - first + 1).min(end - j);
            let limit = 2 * hop.max(distance * hop) + 1;
            if h.window_length[j] > limit {
                h.window_length[j] = limit;
            }
        }
    }
}

pub fn render(
    h: &Harmonics,
    sr: u32,
    samples: usize,
    hop: usize,
    f0_scale: f64,
) -> Result<Vec<f64>> {
    ensure!(
        f0_scale.is_finite() && f0_scale > 0.,
        "f0_scale must be finite and positive"
    );
    ensure!(
        h.channels > 0 && hop > 0,
        "invalid harmonic render dimensions"
    );
    let y = render_internal(h, sr as f64, samples, hop, f0_scale);
    ensure!(
        y.iter().all(|x| x.is_finite()),
        "harmonic render produced non-finite values"
    );
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_render_sine() {
        let sr = 8000;
        let n = 4000;
        let mut d = vec![0.; n];
        for i in 0..n {
            d[i] = (2. * PI * 200. * i as f64 / sr as f64).sin();
        }
        let a = Audio {
            sample_rate: sr,
            channels: 1,
            data: d,
            encoding: "FLOAT64".into(),
        };
        let mut c = Config::default();
        c.maxnhar = 8;
        c.vocal_f0_min = 100.;
        c.vocal_f0_max = 400.;
        let f = vec![200.; 21];
        let q = vec![1.; 21];
        let (h, y) = fit(&a, &f, &q, 200, &c).unwrap();
        assert!(h.count.iter().any(|x| *x > 0));
        assert_eq!(y.len(), n);
        assert!(y.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn internal_unvoiced_gap_does_not_keep_full_harmonic_windows() {
        let sr = 8000_u32;
        let samples = 8000_usize;
        let hop = 40_usize;
        let frames = samples.div_ceil(hop) + 1;
        let mut data = vec![0.0; samples];
        let mut f0 = vec![200.0; frames];
        for i in 0..samples {
            let t = i as f64 / sr as f64;
            data[i] = (2.0 * PI * 200.0 * t).sin();
        }
        let gap_start = 3200;
        let gap_end = 3600;
        for i in gap_start..gap_end {
            f0[i / hop] = 0.0;
            data[i] += 0.25 * (i as f64 * 0.73).sin();
        }
        let confidence = f0
            .iter()
            .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let audio = Audio {
            sample_rate: sr,
            channels: 1,
            data,
            encoding: "FLOAT64".into(),
        };
        let mut cfg = Config::default();
        cfg.maxnhar = 12;
        cfg.vocal_f0_min = 100.;
        cfg.vocal_f0_max = 400.;
        let (harmonics, rendered) = fit(&audio, &f0, &confidence, hop, &cfg).unwrap();
        let interior = harmonics.window_length[(gap_start / hop) - 2];
        let edge = harmonics.window_length[(gap_start / hop) - 1];
        assert!(
            edge < interior,
            "internal gap did not shorten harmonic support"
        );
        let gap_rms = (rendered[gap_start..gap_end]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            / (gap_end - gap_start) as f64)
            .sqrt();
        assert!(
            gap_rms < 0.15,
            "harmonic layer absorbed too much gap energy: {gap_rms}"
        );
    }

    #[test]
    fn low_confidence_frame_suppresses_periodic_energy() {
        let mut h = Harmonics::new(1, 3, 1);
        h.f0 = vec![200.0, 200.0, 200.0];
        h.confidence = vec![1.0, 0.0, 1.0];
        h.count = vec![1, 1, 1];
        h.window_length = vec![81, 81, 81];
        for frame in 0..3 {
            h.frequency[frame] = 200.0;
            let ix = h.index(0, frame, 0);
            h.amplitude[ix] = 1.0;
        }
        let rendered = render(&h, 8000, 200, 40, 1.0).unwrap();
        let middle = rendered[40..44].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            middle < 0.05,
            "low-confidence frame retained periodic energy: {middle}"
        );
    }

    #[test]
    fn breathy_noise_stays_in_residual_against_periodic_reference() {
        let sr = 8000_u32;
        let n = 8000_usize;
        let hop = 40_usize;
        let frames = n.div_ceil(hop) + 1;
        let mut htrue = vec![0.; n];
        let mut data = vec![0.; n];
        for i in 0..n {
            let t = i as f64 / sr as f64;
            htrue[i] = (2. * PI * 200. * t).sin() + 0.35 * (2. * PI * 400. * t).sin();
            data[i] = htrue[i] + 0.12 * (i as f64 * 0.73).sin();
        }
        let f0 = vec![200.; frames];
        let confidence = vec![1.; frames];
        let audio = Audio {
            sample_rate: sr,
            channels: 1,
            data,
            encoding: "FLOAT64".into(),
        };
        let mut cfg = Config::default();
        cfg.maxnhar = 12;
        cfg.vocal_f0_min = 100.;
        cfg.vocal_f0_max = 400.;
        let (_, rendered) = fit(&audio, &f0, &confidence, hop, &cfg).unwrap();
        let signal_energy = htrue.iter().map(|x| x * x).sum::<f64>();
        let error = htrue
            .iter()
            .zip(&rendered)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>();
        assert!(10. * (signal_energy / error.max(1e-30)).log10() > 12.);
    }
}
