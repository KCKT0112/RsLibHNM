use crate::types::{Audio, Features, Harmonics};
use crate::{
    config::Config,
    dsp::{self, Fft},
    glottal,
};
use anyhow::{Result, ensure};
use num_complex::Complex64;
use std::f64::consts::PI;

fn psd(audio: &Audio, n_fft: usize, hop: usize) -> Result<(Vec<f64>, usize, usize)> {
    let spec = dsp::stft(audio, n_fft, hop);
    let bins = spec.bins;
    let frames = spec.frames;
    let w = dsp::hann(n_fft);
    let norm = (w.iter().map(|x| x * x).sum::<f64>() * audio.sample_rate as f64).max(1e-30);
    let mut out = vec![0.; audio.channels * frames * bins];
    for c in 0..audio.channels {
        for f in 0..frames {
            for k in 0..bins {
                let z = spec.data[(c * frames + f) * bins + k];
                let mut p = z.norm_sqr() / norm;
                if k > 0 && k + 1 < bins {
                    p *= 2.;
                }
                out[(c * frames + f) * bins + k] = p.max(0.);
            }
        }
    }
    Ok((out, frames, bins))
}
fn rebin(
    src: &[f64],
    channels: usize,
    frames: usize,
    bins: usize,
    sr: u32,
    cells: usize,
    n_fft: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nyq = sr as f64 / 2.;
    let mut se = vec![0.; bins + 1];
    for k in 0..=bins {
        let center = nyq * k as f64 / (bins - 1).max(1) as f64;
        se[k] = if k == 0 {
            0.
        } else if k == bins {
            nyq
        } else {
            nyq * (k as f64 - 0.5) / (bins - 1) as f64
        };
        if k == bins - 1 {
            se[k] = nyq;
        }
        if k > 0 && k < bins - 1 {
            se[k] = nyq * (k as f64 - 0.5) / (bins - 1) as f64;
        }
        let _ = center;
    }
    // Source cells are midpoint Voronoi intervals with half-width endpoints.
    se[0] = 0.;
    for k in 1..bins {
        se[k] = nyq * (k as f64 - 0.5) / (bins - 1) as f64;
    }
    se[bins] = nyq;
    let mut te = vec![0.; cells + 1];
    for b in 0..=cells {
        te[b] = nyq * b as f64 / cells as f64;
    }
    let mut overlap_by_target: Vec<Vec<(usize, f64)>> = (0..cells).map(|_| Vec::new()).collect();
    for k in 0..bins {
        let width = (se[k + 1] - se[k]).max(1e-30);
        for b in 0..cells {
            let ov = (se[k + 1].min(te[b + 1]) - se[k].max(te[b])).max(0.);
            if ov > 0. {
                overlap_by_target[b].push((k, ov / width / (te[b + 1] - te[b])));
            }
        }
    }
    let mut out = vec![0.; channels * frames * cells];
    let df = sr as f64 / n_fft as f64;
    for c in 0..channels {
        for f in 0..frames {
            for b in 0..cells {
                let dst = (c * frames + f) * cells + b;
                let mut value = 0.;
                for &(k, coeff) in &overlap_by_target[b] {
                    value += src[(c * frames + f) * bins + k] * df * coeff;
                }
                out[dst] = value;
            }
        }
    }
    (out, te)
}
fn overlap_matrix(cells: &[f64], edges: &[f64]) -> Vec<Vec<f64>> {
    (0..cells.len() - 1)
        .map(|k| {
            (0..edges.len() - 1)
                .map(|b| (cells[k + 1].min(edges[b + 1]) - cells[k].max(edges[b])).max(0.))
                .collect()
        })
        .collect()
}
fn bands(
    psd: &[f64],
    channels: usize,
    frames: usize,
    cells: usize,
    cell_edges: &[f64],
    edges: &[f64],
) -> Vec<f64> {
    let ov = overlap_matrix(cell_edges, edges);
    let mut out = vec![0.; channels * frames * (edges.len() - 1)];
    for c in 0..channels {
        for f in 0..frames {
            for b in 0..edges.len() - 1 {
                let mut v = 0.;
                for k in 0..cells {
                    v += psd[(c * frames + f) * cells + k] * ov[k][b];
                }
                out[(c * frames + f) * (edges.len() - 1) + b] = v;
            }
        }
    }
    out
}
fn dct(v: &[f64], n: usize) -> Vec<f64> {
    let m = v.len() as f64;
    (0..n)
        .map(|k| {
            let s = (0..v.len())
                .map(|i| v[i] * (PI * (i as f64 + 0.5) * k as f64 / m).cos())
                .sum::<f64>();
            s * (if k == 0 {
                1. / m.sqrt()
            } else {
                (2. / m).sqrt()
            })
        })
        .collect()
}

fn modulation(
    residual: &Audio,
    f0: &[f64],
    sr: u32,
    hop: usize,
    edges: &[f64],
    order: usize,
) -> Vec<f64> {
    let samples = residual.samples();
    let nfft = samples.next_power_of_two().max(2);
    let mut result = vec![0.; residual.channels * f0.len() * (edges.len() - 1) * (1 + 2 * order)];
    let dim = 1 + 2 * order;
    let mut fft = Fft::new(nfft);
    let mut x = vec![0.; nfft];
    let mut base = vec![Complex64::new(0., 0.); nfft];
    let mut z = vec![Complex64::new(0., 0.); nfft];
    let mut filtered = vec![0.; nfft];
    let mut env = vec![0.; nfft];
    let mut ata = vec![0.; dim * dim];
    let mut aty = vec![0.; dim];
    let mut row = vec![0.; dim];
    let mut solver = vec![0.; dim * (dim + 1)];
    let mut solution = vec![0.; dim];
    let frequencies: Vec<f64> = (0..nfft)
        .map(|k| k.min(nfft - k) as f64 * sr as f64 / nfft as f64)
        .collect();
    let windows: Vec<(usize, usize)> = f0
        .iter()
        .enumerate()
        .map(|(j, &f)| {
            let half = (2. * sr as f64 / f.max(100.)).round().max(hop as f64) as usize;
            let a = j.saturating_mul(hop).saturating_sub(half);
            let b = (j * hop + half + 1).min(samples);
            (a, b)
        })
        .collect();
    for c in 0..residual.channels {
        for i in 0..samples {
            x[i] = residual.data[i * residual.channels + c];
        }
        x[samples..].fill(0.);
        fft.forward_into(&x, &mut base);
        for bnd in 0..edges.len() - 1 {
            z.copy_from_slice(&base);
            for k in 0..nfft {
                let fr = frequencies[k];
                if fr < edges[bnd] || fr >= edges[bnd + 1] {
                    z[k] = Complex64::new(0., 0.);
                }
            }
            fft.inverse_into(&z, &mut filtered);
            fft.analytic_envelope_into(&filtered, &mut env);
            for (j, &f0j) in f0.iter().enumerate() {
                let (a, bb) = windows[j];
                if bb <= a {
                    continue;
                }
                let count = bb - a;
                let mut rms = 0.;
                for i in a..bb {
                    rms += filtered[i] * filtered[i];
                }
                rms = (rms / count as f64).sqrt();
                let idx = ((c * f0.len() + j) * (edges.len() - 1) + bnd) * dim;
                result[idx] = rms;
                if f0j <= 0. || count <= 2 * order + 1 {
                    continue;
                }
                ata.fill(0.);
                aty.fill(0.);
                for i in a..bb {
                    let t = (i as f64 - j as f64 * hop as f64) / sr as f64;
                    row.fill(1.);
                    for k in 1..=order {
                        row[k] = (2. * PI * t * f0j * k as f64).cos();
                        row[order + k] = (2. * PI * t * f0j * k as f64).sin();
                    }
                    for q in 0..dim {
                        aty[q] += row[q] * env[i] / 2f64.sqrt();
                        for r in 0..dim {
                            ata[q * dim + r] += row[q] * row[r];
                        }
                    }
                }
                if solve_into(&ata, &aty, dim, &mut solver, &mut solution) {
                    for k in 1..=order {
                        result[idx + k] = solution[k];
                        result[idx + order + k] = solution[order + k];
                    }
                }
            }
        }
    }
    result
}
fn solve_into(a: &[f64], b: &[f64], n: usize, m: &mut [f64], out: &mut [f64]) -> bool {
    for i in 0..n {
        for j in 0..n {
            m[i * (n + 1) + j] = a[i * n + j];
        }
        m[i * (n + 1) + n] = b[i];
    }
    for i in 0..n {
        let Some(p) = (i..n).max_by(|x, y| {
            m[*x * (n + 1) + i]
                .abs()
                .total_cmp(&m[*y * (n + 1) + i].abs())
        }) else {
            return false;
        };
        if m[p * (n + 1) + i].abs() < 1e-12 {
            return false;
        }
        for j in i..=n {
            m.swap(i * (n + 1) + j, p * (n + 1) + j);
        }
        let d = m[i * (n + 1) + i];
        for j in i..=n {
            m[i * (n + 1) + j] /= d;
        }
        for q in 0..n {
            if q == i {
                continue;
            }
            let d = m[q * (n + 1) + i];
            for j in i..=n {
                m[q * (n + 1) + j] -= d * m[i * (n + 1) + j];
            }
        }
    }
    for i in 0..n {
        out[i] = m[i * (n + 1) + n];
    }
    true
}

/// Extract measured, parameter-only HNM features. PSD rebinning conserves integrated power.
pub fn extract(
    harmonic: &Audio,
    residual: &Audio,
    h: &Harmonics,
    config: &Config,
    hop: usize,
) -> Result<Features> {
    ensure!(
        harmonic.channels == residual.channels && harmonic.data.len() == residual.data.len(),
        "H and N must align"
    );
    ensure!(
        harmonic.sample_rate == residual.sample_rate,
        "sample rate mismatch"
    );
    residual.validate()?;
    config.validate(harmonic.sample_rate)?;
    let (hp, frames, bins) = psd(harmonic, config.n_fft, hop)?;
    let (np, fr2, _) = psd(residual, config.n_fft, hop)?;
    ensure!(frames == fr2 && h.frames == frames, "frame mismatch");
    let (npsd, cells) = rebin(
        &np,
        harmonic.channels,
        frames,
        bins,
        harmonic.sample_rate,
        config.npsd,
        config.n_fft,
    );
    let (hpsd, _) = rebin(
        &hp,
        harmonic.channels,
        frames,
        bins,
        harmonic.sample_rate,
        config.npsd,
        config.n_fft,
    );
    let total: Vec<f64> = npsd.iter().zip(&hpsd).map(|(a, b)| a + b).collect();
    let nyq = harmonic.sample_rate as f64 / 2.;
    let edges = vec![
        0.,
        config.chanfreq[0].min(nyq),
        config.chanfreq[1].min(nyq),
        config.chanfreq[2].min(nyq),
        nyq,
    ];
    let band_energy = bands(
        &npsd,
        harmonic.channels,
        frames,
        config.npsd,
        &cells,
        &edges,
    );
    let mut envelope = vec![0.; harmonic.channels * frames];
    for c in 0..harmonic.channels {
        for f in 0..frames {
            let mut x = 0.;
            for k in 0..config.npsd {
                x += npsd[(c * frames + f) * config.npsd + k] * (cells[k + 1] - cells[k]);
            }
            envelope[c * frames + f] = x.max(0.).sqrt();
        }
    }
    let bap_edges = (0..6).map(|i| nyq * i as f64 / 5.).collect::<Vec<_>>();
    let bn = bands(
        &npsd,
        harmonic.channels,
        frames,
        config.npsd,
        &cells,
        &bap_edges,
    );
    let bt = bands(
        &total,
        harmonic.channels,
        frames,
        config.npsd,
        &cells,
        &bap_edges,
    );
    let mut bap = vec![1.; harmonic.channels * frames * 5];
    for c in 0..harmonic.channels {
        for f in 0..frames {
            for b in 0..5 {
                let i = (c * frames + f) * 5 + b;
                bap[i] = if bt[i] > 1e-25 {
                    (bn[i] / bt[i]).clamp(0., 1.)
                } else {
                    1.
                };
                if h.f0[f] <= 0. {
                    bap[i] = 1.;
                }
            }
        }
    }
    let mut bapc = vec![0.; harmonic.channels * frames * (config.order_bap + 1)];
    let mut ratio = vec![1.; config.npsd];
    for c in 0..harmonic.channels {
        for f in 0..frames {
            for k in 0..config.npsd {
                let i = (c * frames + f) * config.npsd + k;
                ratio[k] = if total[i] > 1e-25 {
                    npsd[i] / total[i]
                } else {
                    1.
                };
            }
            let d = dct(&ratio, config.order_bap + 1);
            for q in 0..d.len() {
                bapc[(c * frames + f) * (config.order_bap + 1) + q] = d[q];
            }
        }
    }
    let mut spectral = vec![0.; harmonic.channels * frames * (config.order_spec + 1)];
    let mut loga = vec![0.; bins];
    for c in 0..harmonic.channels {
        for f in 0..frames {
            for k in 0..bins {
                loga[k] = 0.5
                    * (hp[(c * frames + f) * bins + k] + np[(c * frames + f) * bins + k])
                        .max(1e-20)
                        .ln();
            }
            let cep = dct(&loga, config.order_spec + 1);
            for q in 0..cep.len() {
                spectral[(c * frames + f) * (config.order_spec + 1) + q] = cep[q];
            }
        }
    }
    let mut mel = vec![0.; harmonic.channels * frames * 64];
    let melaxis = (0..64)
        .map(|i| 700. * ((i as f64 / 63. * (1. + nyq / 700.).ln()).exp() - 1.))
        .collect::<Vec<_>>();
    let mut smooth = vec![0.; bins];
    for c in 0..harmonic.channels {
        for f in 0..frames {
            let sigma = (h.f0[f] * config.n_fft as f64 / harmonic.sample_rate as f64 * 0.5).max(1.);
            for k in 0..bins {
                let mut a = 0.;
                let mut w = 0.;
                for q in 0..bins {
                    let z = (q as f64 - k as f64) / sigma;
                    let ww = (-0.5 * z * z).exp();
                    a += ww
                        * ((hp[(c * frames + f) * bins + q] + np[(c * frames + f) * bins + q])
                            .max(1e-20)
                            .ln()
                            * 0.5);
                    w += ww;
                }
                smooth[k] = a / w;
            }
            for m in 0..64 {
                let pos = (melaxis[m] / nyq * (bins - 1) as f64).clamp(0., (bins - 1) as f64);
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(bins - 1);
                mel[(c * frames + f) * 64 + m] =
                    smooth[lo] + (smooth[hi] - smooth[lo]) * (pos - lo as f64);
            }
        }
    }
    let (rd, rd_conf) = glottal::estimate_rd(
        &h.amplitude,
        h.channels,
        frames,
        h.capacity,
        &h.f0,
        harmonic.sample_rate,
        config.lip_radius,
    );
    let noise_modulation = modulation(
        residual,
        &h.f0,
        harmonic.sample_rate,
        hop,
        &edges,
        config.maxnhar_e,
    );
    let lip_response = cells
        .windows(2)
        .map(|w| glottal::lip_response((w[0] + w[1]) * 0.5, config.lip_radius))
        .collect();
    let mut vector = vec![0.; harmonic.channels * frames * 72];
    for c in 0..harmonic.channels {
        for f in 0..frames {
            let i = (c * frames + f) * 72;
            vector[i] = if h.f0[f] > 0. { 1. } else { 0. };
            vector[i + 1] = h.f0[f];
            vector[i + 2] = rd[c * frames + f];
            vector[i + 3..i + 67]
                .copy_from_slice(&mel[(c * frames + f) * 64..(c * frames + f) * 64 + 64]);
            vector[i + 67..i + 72]
                .copy_from_slice(&bap[(c * frames + f) * 5..(c * frames + f) * 5 + 5]);
        }
    }
    Ok(Features {
        noise_envelope: envelope,
        noise_psd: npsd,
        noise_psd_edges: cells,
        noise_band_energy: band_energy,
        noise_band_edges: edges,
        noise_modulation: noise_modulation,
        spectral_cepstrum: spectral,
        bap_cepstrum: bapc,
        lip_response,
        mel_log_spectrum: mel,
        bap,
        rd,
        rd_confidence: rd_conf,
        vector,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dct_shape() {
        assert_eq!(dct(&[1.; 8], 6).len(), 6);
    }
}
