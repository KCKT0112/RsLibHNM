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

No hidden resampling is performed. WAV input is interleaved sample-major `f64`; output is float32 WAV. The compact model stores independent parameters and synthesis features, not original replay spectra; it cannot provide exact STFT replay. A 1-second 16 kHz smoke model is about 90 KB; the full 44.1 kHz example compact model is about 7.8 MB because it stores roughly 2903 frames of 120-partial f64-derived parameter data plus noise PSD/modulation and the 72-D vector.

## Current evidence and limitations

The Rust port is structurally complete for compact parameter analysis, H/N export, compressed model save/load, and parameter synthesis, but has not reached Python quality/performance parity. On the current Apple Silicon workstation, release analysis of `example/1_01.wav` completed in approximately 64 seconds after Rayon frame parallelization (previous Rust version approximately 186 seconds); this is not real-time. The Rust RNG and FFT differ numerically from Python, so parameter noise is compared statistically rather than sample-for-sample. The internal consonant-gap regression passes, but broad recorded-vocal separation validation remains incomplete.

The Rust unit suite currently contains 13 tests. Before production use, add cross-language golden fixtures for F0, harmonic parameters, 72-D features, replay error, H/N metrics, and a controlled performance target. Short low-pitch transitions and source/tract Rd ambiguity remain model limitations in both implementations.
