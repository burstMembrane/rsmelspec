use crate::colors::Colormap;
use audio::resample_audio;
use audio::split;
use clap::Parser;
use clio::Input;
use rayon::prelude::*;
use std::time::Instant;
use wavers::Wav;
mod audio;
mod colors;
mod mel;
#[derive(Parser)]
struct Cli {
    #[clap(short, long)]
    input_file: Input,

    #[clap(short, long, default_value = "128")]
    mels: usize,

    #[clap(short, long, default_value = "2048")]
    fft_size: usize,

    #[clap(long, default_value = "512")]
    hop_size: usize,

    #[clap(short, long, default_value = "2048")]
    win_size: usize,

    #[clap(long, default_value = "0.0")]
    f_min: f32,

    #[clap(long, default_value = "8000.0")]
    f_max: f32,

    #[clap(short, long, default_value = "80.0")]
    top_db: f32,

    #[clap(long, default_value = "true")]
    onesided: bool,

    #[clap(short, long, default_value = "22050.0")]
    sampling_rate: f64,

    #[clap(short, long, default_value = "output.png")]
    output: String,
    // width and height
    #[clap(short, long, default_value = "0")]
    width: u32,
    #[clap(short, long, default_value = "0")]
    height: u32,
    #[clap(long, default_value = "10.0")]
    chunk_duration: f32,

    #[clap(long, default_value = "cpu")]
    device: String,
    // colormap
    #[clap(long, default_value = "magma")]
    colormap: String,
}

fn main() {
    let args = Cli::parse();
    let input_file = args.input_file;
    let output_path = &args.output;
    let fft_size = args.fft_size;
    let hop_size = args.hop_size;
    let n_mels = args.mels;
    let sampling_rate = args.sampling_rate;
    let top_db = args.top_db;
    let f_min = args.f_min;
    let f_max = args.f_max;
    let win_size = args.win_size;

    let spectrogram_config = mel::SpectrogramConfig::new(args.onesided);
    let mel_config = mel::MelConfig::new(
        sampling_rate as f32,
        fft_size,
        win_size,
        hop_size,
        f_min,
        f_max,
        n_mels,
        top_db,
        spectrogram_config,
    );

    let colormap = Colormap::from_name(&args.colormap).unwrap_or(Colormap::Magma);
    // read the input file to a vec of f32
    let start = Instant::now();
    let mut wav: Wav<f32> = Wav::from_path(input_file.path().as_ref() as &std::path::Path).unwrap();
    // read the audio data into a Vec<f32>
    // get audio metadata
    let duration = wav.duration();
    let sample_rate = wav.sample_rate();
    println!("duration: {:?}", duration);
    println!("sample rate: {:?}", sample_rate);
    let num_samples = duration as f32 * sample_rate as f32;
    println!("num samples: {:?}", num_samples);

    let max_val = wav.read().unwrap().iter().cloned().fold(0.0, f32::max);
    let normalized_buffer: Vec<f32> = wav
        .read()
        .unwrap()
        .iter()
        .cloned()
        .map(|x| x / max_val)
        .collect();
    // Resample if WAV rate differs from desired sampling_rate
    let target_sr = sampling_rate as usize;
    let input_sr = sample_rate as usize;
    let buffer = if input_sr != target_sr {
        eprintln!("Resampling from {} Hz to {} Hz", input_sr, target_sr);
        resample_audio(&normalized_buffer, input_sr, target_sr)
    } else {
        normalized_buffer.clone()
    };
    // Compute overlap in seconds and in frames for chunking
    let overlap_secs = (win_size as f32 - hop_size as f32) / sampling_rate as f32;
    let overlap_frames = (overlap_secs * sampling_rate as f32 / hop_size as f32).round() as usize;
    let mel_specs: Vec<Vec<Vec<f32>>> = if args.chunk_duration > 0.0 {
        let (chunks, _padding) = split(
            &buffer,
            args.chunk_duration,
            overlap_secs,
            sampling_rate as u32,
        );
        // Overlap-aware chunk processing: drop overlap frames except for first chunk
        chunks
            .into_par_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut spec = mel::mel_spectrogram_db(mel_config.clone(), chunk.to_vec());
                if i > 0 {
                    spec.drain(0..overlap_frames.min(spec.len()));
                }
                spec
            })
            .collect()
    } else {
        vec![mel::mel_spectrogram_db(mel_config.clone(), buffer.clone())]
    };

    // Stitch the time frames back together
    let full_spec: Vec<Vec<f32>> = mel_specs.into_iter().flatten().collect();
    // debug any zero values
    for spec in full_spec.iter() {
        for val in spec.iter() {
            if *val == 0.0 {
                println!("zero value: {:?}", val);
            }
        }
    }

    // Determine output resolution: if width or height is zero, use spec dimensions
    let spec_frames = full_spec.len() as u32;
    let spec_bins = full_spec.get(0).map(|row| row.len() as u32).unwrap_or(0);
    let width_px = if args.width == 0 {
        spec_frames
    } else {
        args.width
    };
    let height_px = if args.height == 0 {
        spec_bins
    } else {
        args.height
    };

    // Plot the full spectrogram
    let image = mel::plot_mel_spec(full_spec, colormap, width_px, height_px);
    image.save(output_path).unwrap();
    println!("elapsed: {:?}", start.elapsed());
    // realtime factor -- the audio duration divided by the elapsed time
    let realtime_factor = num_samples as f32 / start.elapsed().as_secs_f32();
    println!("realtime factor: {:?}x", realtime_factor);
}
