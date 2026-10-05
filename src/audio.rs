use crate::types::Audio;
use anyhow::{Result, ensure};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

pub fn read_wav(path: &std::path::Path) -> Result<Audio> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let mut data = Vec::new();
    match spec.sample_format {
        SampleFormat::Float => {
            for x in reader.samples::<f32>() {
                data.push(x? as f64);
            }
        }
        SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample.saturating_sub(1))) as f64;
            for x in reader.samples::<i32>() {
                data.push(x? as f64 / scale);
            }
        }
    }
    let channels = spec.channels as usize;
    ensure!(
        channels > 0 && data.len() % channels == 0,
        "invalid WAV channels"
    );
    let audio = Audio {
        sample_rate: spec.sample_rate,
        channels,
        data,
        encoding: format!("{:?}", spec.sample_format),
    };
    audio.validate()?;
    Ok(audio)
}
pub fn write_wav(path: &std::path::Path, audio: &Audio) -> Result<()> {
    audio.validate()?;
    ensure!(
        audio.data.iter().all(|x| *x >= -1. && *x <= 1.),
        "WAV would clip"
    );
    let spec = WavSpec {
        channels: audio.channels as u16,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for x in &audio.data {
        writer.write_sample(*x as f32)?;
    }
    writer.finalize()?;
    Ok(())
}
