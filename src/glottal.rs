use anyhow::{Result, ensure};
use std::f64::consts::PI;

/// One normalized LF glottal-flow derivative period for Rd in [0.3, 2.7].
/// Root solves use deterministic bisection rather than SciPy's Brent solver;
/// tiny last-bit differences are expected and do not affect model semantics.
pub fn lf_pulse(rd: f64, size: usize) -> Result<Vec<f64>> {
    ensure!(
        rd.is_finite() && (0.3..=2.7).contains(&rd),
        "LF Rd must be in [0.3, 2.7]"
    );
    ensure!(size > 0, "pulse size must be positive");
    let ra = 0.048 * rd - 0.01;
    let rk = 0.118 * rd + 0.224;
    let rg = rk * (0.5 + 1.2 * rk) / (4.0 * (0.11 * rd - ra * (0.5 + 1.2 * rk)));
    let tp = 1.0 / (2.0 * rg);
    let te = tp * (1.0 + rk);
    let duration = 1.0 - te;
    let fun_eps = |e: f64| -> f64 {
        (if e.abs() < 1e-7 {
            duration
        } else {
            -(-e * duration).exp_m1() / e
        }) - ra
    };
    let epsilon = bisect(fun_eps, 1e-6, 1e5, 100)?;
    let omega = PI / tp;
    let q = if epsilon.abs() < 1e-8 {
        duration * duration * 0.5
    } else {
        -(-epsilon * duration).exp_m1() / epsilon - duration * (-epsilon * duration).exp()
    };
    let tail_area = -q / (epsilon * ra);
    let sin_te = (omega * te).sin();
    let balance = |alpha: f64| -> f64 {
        let e0 = -(-alpha * te).exp() / sin_te;
        let den = alpha * alpha + omega * omega;
        let opened = e0
            * (((alpha * te).exp() * (alpha * sin_te - omega * (omega * te).cos())) + omega)
            / den;
        opened + tail_area
    };
    let alpha = bisect(balance, -100.0, 100.0, 120)?;
    let e0 = -(-alpha * te).exp() / sin_te;
    let mut pulse = vec![0.0; size];
    for i in 0..size {
        let t = i as f64 / size as f64;
        pulse[i] = if t <= te {
            e0 * (alpha * t).exp() * (omega * t).sin()
        } else {
            -(((-epsilon * (t - te)).exp() - (-epsilon * duration).exp()) / (epsilon * ra))
        };
    }
    Ok(pulse)
}
fn bisect<F: Fn(f64) -> f64>(f: F, mut lo: f64, mut hi: f64, n: usize) -> Result<f64> {
    let mut flo = f(lo);
    let fhi = f(hi);
    ensure!(
        flo.is_finite() && fhi.is_finite() && flo * fhi <= 0.0,
        "LF root is not bracketed"
    );
    for _ in 0..n {
        let mid = (lo + hi) * 0.5;
        let fm = f(mid);
        if !fm.is_finite() {
            break;
        }
        if flo * fm <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            flo = fm;
        }
    }
    Ok((lo + hi) * 0.5)
}

pub fn lip_response(frequency: f64, radius_cm: f64) -> f64 {
    let ka = 2.0 * PI * frequency * radius_cm / 100.0 / 343.0;
    if ka <= 0.0 {
        0.0
    } else {
        ka / (1.0 + ka * ka).sqrt()
    }
}

/// Robust magnitude/curvature LF Rd estimate. Returned confidence is a heuristic.
pub fn estimate_rd(
    amplitudes: &[f64],
    channels: usize,
    frames: usize,
    capacity: usize,
    f0: &[f64],
    sample_rate: u32,
    radius_cm: f64,
) -> (Vec<f64>, Vec<f64>) {
    let grid: Vec<f64> = (0..49).map(|i| 0.3 + 2.4 * i as f64 / 48.0).collect();
    let mut templ = vec![vec![0.0; 32]; 49];
    for (gi, rd) in grid.iter().enumerate() {
        if let Ok(p) = lf_pulse(*rd, 4096) {
            for k in 0..32 {
                let mut re = 0.;
                let mut im = 0.;
                for (n, &v) in p.iter().enumerate() {
                    let a = -2. * PI * (k + 1) as f64 * n as f64 / 4096.;
                    re += v * a.cos();
                    im += v * a.sin();
                }
                templ[gi][k] = (re * re + im * im).sqrt().max(1e-15).ln();
            }
        }
    }
    let mut out = vec![0.; channels * frames];
    let mut conf = vec![0.; channels * frames];
    for j in 0..frames {
        let ff = f0.get(j).copied().unwrap_or(0.);
        if !(ff.is_finite() && ff > 0.) {
            continue;
        }
        let count = ((sample_rate as f64 / 2. - 1.) / ff.max(1.))
            .floor()
            .max(0.) as usize;
        let count = count.min(32).min(capacity);
        if count < 3 {
            continue;
        }
        let mut pred = vec![vec![0.; count]; 49];
        for g in 0..49 {
            for k in 0..count {
                let rad = lip_response((k + 1) as f64 * ff, radius_cm).max(1e-20);
                pred[g][k] = templ[g][k] - ((k + 1) as f64).ln() + rad.ln();
            }
        }
        for c in 0..channels {
            let base = (c * frames + j) * capacity;
            let mut peak = 0.;
            for k in 0..count {
                let v = amplitudes.get(base + k).copied().unwrap_or(0.);
                if v.is_finite() && v > peak {
                    peak = v;
                }
            }
            if peak <= 0. {
                continue;
            }
            let mut valid = vec![false; count];
            let mut logs = vec![0.; count];
            for k in 0..count {
                let v = amplitudes[base + k] / peak;
                valid[k] = v.is_finite() && v > 0.001;
                if valid[k] {
                    logs[k] = v.ln();
                }
            }
            let mut mag = vec![0.; 49];
            for g in 0..49 {
                let mut vals = Vec::new();
                for k in 0..count {
                    if valid[k] {
                        vals.push(logs[k] - pred[g][k]);
                    }
                }
                if vals.is_empty() {
                    mag[g] = 1e9;
                    continue;
                }
                vals.sort_by(|a, b| a.total_cmp(b));
                let med = vals[vals.len() / 2];
                let mut loss = 0.;
                for x in vals {
                    let z = (x - med).abs();
                    loss += if z < 0.5 {
                        0.5 * z * z
                    } else {
                        0.5 * (z - 0.25)
                    };
                }
                mag[g] = loss / (valid.iter().filter(|x| **x).count() as f64);
            }
            let mb = mag
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|x| x.0)
                .unwrap_or(0);
            let mut triples = 0;
            for k in 0..count - 2 {
                if valid[k] && valid[k + 1] && valid[k + 2] {
                    triples += 1;
                }
            }
            let mut best = mb;
            let mut losses = vec![0.; 49];
            if triples >= 6 {
                for g in 0..49 {
                    let mut l = 0.;
                    let mut n = 0.;
                    for k in 0..count - 2 {
                        if valid[k] && valid[k + 1] && valid[k + 2] {
                            let cur = 0.5
                                * ((logs[k] - pred[g][k]) - 2. * (logs[k + 1] - pred[g][k + 1])
                                    + (logs[k + 2] - pred[g][k + 2]));
                            let z = cur / 0.05;
                            l += (1. + z * z).sqrt() - 1.;
                            n += 1.;
                        }
                    }
                    losses[g] = if n > 0. { l / n } else { 1e9 };
                }
                if mag[mb] >= 0.01 {
                    best = losses
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .map(|x| x.0)
                        .unwrap_or(mb);
                }
                let alt = losses
                    .iter()
                    .enumerate()
                    .filter(|(g, _)| (grid[*g] - grid[best]).abs() >= 0.3 - 1e-8)
                    .map(|(_, v)| *v)
                    .fold(f64::INFINITY, f64::min);
                let sep = if alt > 0. {
                    (1. - losses[best] / alt).max(0.)
                } else {
                    0.
                };
                let mut q = (-4. * mag[best]).exp() * sep;
                if best == 0 || best == 48 {
                    q *= 0.5;
                }
                conf[c * frames + j] = q;
            }
            out[c * frames + j] = grid[best];
        }
    }
    (out, conf)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pulse_finite() {
        let p = lf_pulse(1., 1024).unwrap();
        assert!(p.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn lip_bounds() {
        assert_eq!(lip_response(0., 1.5), 0.);
        assert!(lip_response(1000., 1.5) > 0.);
    }
}
