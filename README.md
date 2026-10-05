# RsLibHNM

Independent Rust implementation of a clean-vocal Harmonic + Noise Model (HNM).

Status: core port compiles, unit tests pass, and `example/1_01.wav` completes the analysis/separation and saved-model synthesis path. Rust numerical output is not ABI-compatible with the Python package or any other implementation; it uses `f64`, rustfft, nalgebra, and ChaCha8 noise generation.

## Build and test

```bash
cargo check
cargo test
cargo build --release
```

## CLI

```bash
cargo run --release --bin rsLibHNM -- separate input.wav -o output
cargo run --release --bin rsLibHNM -- synthesize output/analysis.rs-hnm \
  -o output/synthesized.wav --f0-scale 1.25
cargo run --release --bin rsLibHNM -- synthesize output/analysis.rs-hnm \
  -o output/preserved.wav --f0-scale 1.25 --preserve-spectral-envelope
```

`separate` writes a compact versioned `analysis.rs-hnm` model containing active parameter data, synthesis-required noise features, and the 72-D vector, plus float WAV H/N/reconstructed components. Derived cepstrum/Mel/BAP/Rd helper arrays are not duplicated in the compact file; they remain available in memory during analysis. Replay spectra are omitted from this first Rust format, so H/N WAV files are exported from memory before the spectra are discarded. `synthesize` uses only saved parameters.

## Fixed contract

Defaults:

```text
n_fft:       2048       maxnhar:    120
maxnhar_e:   5          npsd:       256
nchannel:    4          chanfreq:   3000, 6000, 10000 Hz
order_spec:  64         order_bap:  5
rel_winsize: 4.0        lip_radius: 1.5 cm
```

The 72-D vector is project-defined:

```text
[0]       VUV
[1]       F0 Hz
[2]       LF Rd
[3..66]   64 natural-log Mel amplitude values
[67..71]  5 BAP values
```

No hidden resampling is performed. WAV input is interleaved sample-major `f64`; output is float32 WAV. The compact model stores independent parameters and synthesis features, not original replay spectra; it cannot provide exact STFT replay. A 1-second 16 kHz smoke model is about 90 KB; the full 44.1 kHz example compact model is about **6.7 MB** because it stores roughly 2903 frames of 120-partial harmonic parameters plus noise PSD/modulation and the 72-D vector.

## Current evidence and limitations

The Rust port is structurally complete for compact parameter analysis, H/N export, compressed model save/load, and parameter synthesis, but has not reached Python quality/performance parity. On the current Apple Silicon workstation, the second-round release analysis of `example/1_01.wav` took approximately **39 seconds of post-build runtime**; the measured command including a 14.8-second rebuild took 53.94 seconds. Stage timing was: harmonic fit 35.70 s, feature extraction 3.07 s, F0 0.11 s, STFT 0.06 s. This is not real-time. The compact model is the default because replay spectra dominate file size. The Rust RNG and FFT differ numerically from Python, so parameter noise is compared statistically rather than sample-for-sample.

The compact full-example model is approximately 6.7 MB. Replay RMS error is `1.13e-19`; H+N versus reconstructed WAV differs by `1.49e-8` after float32 output. The Rust unit suite currently contains **15 tests**. A sample-level VUV mask removes harmonic tails in explicit unvoiced gaps, and a confidence gate reduces periodic energy in weakly periodic frames. These gates improve the targeted synthetic consonant cases but do not constitute broad recorded-vocal SDR validation; real voiced consonants and breathy vocals still require an external labeled evaluation set.

The default backend is portable Rust + Rayon; it does not require Accelerate, OpenBLAS, MKL, Metal, or platform-specific native libraries. Set `RSLIBHNM_THREADS` to select a fixed Rayon worker count for reproducible runs; unset uses Rayon’s platform default:

```bash
RSLIBHNM_THREADS=8 cargo run --release --bin rsLibHNM -- separate vocal.wav -o output
```

The published timing is an Apple Silicon measurement of the portable backend, not a cross-platform guarantee. Optional platform-specific BLAS/GPU backends are not enabled in this release; future backends must pass the same H/N, feature and synthesis regression thresholds as the portable path.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
