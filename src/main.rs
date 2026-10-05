use anyhow::Result;
use clap::{Parser, Subcommand};
use rs_libhnm::{
    analyze,
    audio::{read_wav, write_wav},
    config::Config,
    model, synthesis,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "rsLibHNM",
    about = "Clean-vocal harmonic-plus-noise analysis and synthesis"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Analyze {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    Separate {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    Synthesize {
        analysis: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1.0)]
        f0_scale: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        preserve_spectral_envelope: bool,
    },
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Analyze { input, output } => {
            let audio = read_wav(&input)?;
            let result = analyze(&audio, Config::default())?;
            model::save_compact(&result, &output)?;
            println!(
                "saved {} frames at {} Hz",
                result.harmonics.frames, result.metadata.sample_rate
            );
        }
        Command::Separate { input, output } => {
            let audio = read_wav(&input)?;
            let result = analyze(&audio, Config::default())?;
            std::fs::create_dir_all(&output)?;
            // Components are written from the in-memory spectra before discarding replay spectra.
            for (name, component) in [
                ("harmonic", "harmonic"),
                ("noise", "noise"),
                ("reconstructed", "mixture"),
            ] {
                write_wav(
                    &output.join(format!("{name}.wav")),
                    &result.reconstruct(component)?,
                )?;
            }
            model::save_compact(&result, &output.join("analysis.rs-hnm"))?;
            println!(
                "saved version-{} analysis and H/N components",
                result.metadata.analysis_version
            );
        }
        Command::Synthesize {
            analysis,
            output,
            f0_scale,
            seed,
            preserve_spectral_envelope,
        } => {
            let result = model::load(&analysis)?;
            let (wave, harmonic, noise) = synthesis::synthesize(
                &result.harmonics,
                &result.features,
                &result.metadata.config,
                result.metadata.sample_rate,
                result.metadata.num_samples,
                result.metadata.hop_length,
                f0_scale,
                seed,
                preserve_spectral_envelope,
            )?;
            let directory = output.parent().unwrap_or(std::path::Path::new("."));
            std::fs::create_dir_all(directory)?;
            let audio = |data| rs_libhnm::types::Audio {
                sample_rate: result.metadata.sample_rate,
                channels: result.metadata.channels,
                data,
                encoding: "FLOAT32".into(),
            };
            write_wav(&output, &audio(wave))?;
            let stem = output
                .file_stem()
                .and_then(|v| v.to_str())
                .unwrap_or("synthesized");
            write_wav(
                &directory.join(format!("{stem}.harmonic.wav")),
                &audio(harmonic),
            )?;
            write_wav(&directory.join(format!("{stem}.noise.wav")), &audio(noise))?;
        }
    }
    Ok(())
}
