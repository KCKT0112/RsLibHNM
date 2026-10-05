use crate::config::Config;
use crate::types::{Audio, Harmonics};
use anyhow::{Result, ensure};
use nalgebra::DMatrix;
use num_complex::Complex64;
use std::f64::consts::PI;

fn basis(t: &[f64], f0: f64, slope: f64, count: usize) -> DMatrix<f64> {
    let mut a = DMatrix::zeros(t.len(), count * 2);
    for (i, &time) in t.iter().enumerate() {
        let carrier = 2.0 * PI * (f0 * time + 0.5 * slope * time * time);
        for k in 0..count {
            let ph = carrier * (k + 1) as f64;
            a[(i, k)] = ph.cos();
            a[(i, count + k)] = ph.sin();
        }
    }
    a
}

fn solve(a: &DMatrix<f64>, y: &DMatrix<f64>, weight: &[f64]) -> DMatrix<f64> {
    // Tall, narrow weighted sinusoidal design: solve the small Gram system by
    // Cholesky and retain SVD only when rank/conditioning rejects the fit.
    let rows = a.nrows();
    let cols = a.ncols();
    let outputs = y.ncols();
    let mut gram = DMatrix::zeros(cols, cols);
    let mut rhs = DMatrix::zeros(cols, outputs);
    for i in 0..rows {
        let w2 = weight[i] * weight[i];
        for p in 0..cols {
            let ap = a[(i, p)];
            for q in 0..=p {
                gram[(p, q)] += w2 * ap * a[(i, q)];
            }
            for c in 0..outputs {
                rhs[(p, c)] += w2 * ap * y[(i, c)];
            }
        }
    }
    for p in 0..cols {
        for q in 0..p {
            gram[(q, p)] = gram[(p, q)];
        }
    }
    if let Some(chol) = gram.clone().cholesky() {
        return chol.solve(&rhs);
    }
    let mut aw = a.clone();
    let mut yw = y.clone();
    for i in 0..rows {
        for q in 0..cols {
            aw[(i, q)] *= weight[i];
        }
        for c in 0..outputs {
            yw[(i, c)] *= weight[i];
        }
    }
    aw.svd(true, true)
        .solve(&yw, 1e-8)
        .unwrap_or_else(|_| DMatrix::zeros(cols, outputs))
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

fn fit_envelope(
    a: &DMatrix<f64>,
    y: &DMatrix<f64>,
    w: &[f64],
    u: &[f64],
) -> (DMatrix<f64>, Vec<f64>) {
    let channels = y.ncols();
    let mut gradient = vec![0.; channels];
    let mut coeff = solve(a, y, w);
    let rows = a.nrows();
    let cols = a.ncols();
    let mut g0 = DMatrix::zeros(cols, cols);
    let mut g1 = DMatrix::zeros(cols, cols);
    let mut g2 = DMatrix::zeros(cols, cols);
    let mut r0 = DMatrix::zeros(cols, channels);
    let mut r1 = DMatrix::zeros(cols, channels);
    for i in 0..rows {
        let w2 = w[i] * w[i];
        for p in 0..cols {
            for q in 0..=p {
                g0[(p, q)] += w2 * a[(i, p)] * a[(i, q)];
                g1[(p, q)] += w2 * a[(i, p)] * a[(i, q)] * u[i];
                g2[(p, q)] += w2 * a[(i, p)] * a[(i, q)] * u[i] * u[i];
            }
            for c in 0..channels {
                r0[(p, c)] += w2 * a[(i, p)] * y[(i, c)];
                r1[(p, c)] += w2 * a[(i, p)] * y[(i, c)] * u[i];
            }
        }
    }
    for p in 0..cols {
        for q in 0..p {
            g0[(q, p)] = g0[(p, q)];
            g1[(q, p)] = g1[(p, q)];
            g2[(q, p)] = g2[(p, q)];
        }
    }
    for c in 0..channels {
        let mut channel = coeff.column(c).into_owned();
        for _ in 0..5 {
            let carrier = a * &channel;
            let g = bounded_minimize(
                |g| {
                    let mut err = 0.;
                    for i in 0..rows {
                        let d = w[i] * (y[(i, c)] - carrier[i] * (1. + g * u[i]).max(0.));
                        err += d * d;
                    }
                    err
                },
                -4.,
                4.,
            );
            let env_min = u.iter().map(|x| 1. + g * x).fold(f64::INFINITY, f64::min);
            let solved = if env_min > 0.05 {
                let normal = &g0 + 2. * g * &g1 + g * g * &g2;
                normal.cholesky().map(|ch| {
                    ch.solve(&(r0.column(c).into_owned() + g * r1.column(c).into_owned()))
                })
            } else {
                None
            };
            channel = solved.unwrap_or_else(|| {
                let mut ae = a.clone();
                let mut target = DMatrix::zeros(rows, 1);
                for i in 0..rows {
                    let e = (1. + g * u[i]).max(0.);
                    for k in 0..cols {
                        ae[(i, k)] *= e;
                    }
                    target[(i, 0)] = y[(i, c)];
                }
                solve(&ae, &target, w).column(0).into_owned()
            });
            gradient[c] = g;
        }
        coeff.set_column(c, &channel);
    }
    (coeff, gradient)
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
    out
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
    for j in 0..frames {
        let initial = h.f0[j];
        if !initial.is_finite() || initial <= 0. {
            h.f0[j] = 0.;
            continue;
        }
        let length = ((config.rel_winsize * sr / initial).round() as usize).max(2 * hop + 1) | 1;
        let center = j * hop;
        let (a, b, t, w) = frame_window(center, length, samples, sr);
        if b - a < 8 {
            h.f0[j] = 0.;
            h.confidence[j] = 0.;
            continue;
        }
        let valid = w.iter().filter(|x| **x > 0.05).count();
        let maxcount = cap
            .min(((sr / 2. - 1.) / (initial * 1.07)).floor().max(1.) as usize)
            .min((valid / 4).max(1));
        if maxcount < 1 {
            h.f0[j] = 0.;
            h.confidence[j] = 0.;
            continue;
        }
        let mut slope0 = 0.;
        if j > 0 && j + 1 < frames && h.f0[j - 1] > 0. && h.f0[j + 1] > 0. {
            slope0 = (h.f0[j + 1] - h.f0[j - 1]) * sr / (2. * hop as f64);
        }
        let target = DMatrix::from_fn(b - a, 1, |i, _| audio.data[(a + i) * channels + strongest]);
        let fitcount = maxcount.min(12);
        let span = (length / 2) as f64 / sr;
        let eval = |ff: f64, ss: f64| {
            let aa = basis(&t, ff, ss, fitcount);
            let cc = solve(&aa, &target, &w);
            let mut e = 0.;
            for i in 0..aa.nrows() {
                let d = w[i]
                    * (target[(i, 0)]
                        - (0..aa.ncols())
                            .map(|k| aa[(i, k)] * cc[(k, 0)])
                            .sum::<f64>());
                e += d * d;
            }
            e
        };
        let energy = (0..b - a)
            .map(|i| {
                let z = w[i] * target[(i, 0)];
                z * z
            })
            .sum::<f64>();
        if energy < 1e-22 {
            h.f0[j] = 0.;
            h.confidence[j] = 0.;
            continue;
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
        let aa = basis(&t, ff, ss, count);
        h.f0[j] = ff;
        h.frequency[j * cap..j * cap + count]
            .iter_mut()
            .enumerate()
            .for_each(|(k, x)| *x = ff * (k + 1) as f64);
        h.slope[j * cap..j * cap + count]
            .iter_mut()
            .enumerate()
            .for_each(|(k, x)| *x = ss * (k + 1) as f64);
        h.count[j] = count;
        h.window_length[j] = length;
        // Fit all recording channels with the final projected frequency.
        let yy = DMatrix::from_fn(b - a, channels, |i, c| audio.data[(a + i) * channels + c]);
        let (allcoeff, allgr) = fit_envelope(
            &aa,
            &yy,
            &w,
            &t.iter().map(|x| x / span).collect::<Vec<_>>(),
        );
        for c in 0..channels {
            h.amplitude_gradient[c * frames + j] = allgr[c] / span;
            for k in 0..count {
                let ix = h.index(c, j, k);
                h.amplitude[ix] =
                    (allcoeff[(k, c)].powi(2) + allcoeff[(count + k, c)].powi(2)).sqrt();
                h.phase[ix] = (-allcoeff[(count + k, c)]).atan2(allcoeff[(k, c)]);
            }
        }
    }
    stabilize(&mut h, audio, hop);
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
}
