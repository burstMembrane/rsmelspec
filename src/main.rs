use clap::Parser;
use clio::Input;
use rayon::prelude::*;
use wavers::{read, Wav};
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

    #[clap(long, default_value = "10.0")]
    chunk_duration: f32,
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
    // read the input file to a vec of f32
    let mut wav: Wav<f32> = Wav::from_path(input_file.path().as_ref() as &std::path::Path).unwrap();
    // read the audio data into a Vec<f32>
    let buffer: Vec<f32> = wav.read().unwrap().to_vec();

    // Determine chunk size in samples
    let chunk_samples = (args.chunk_duration * sampling_rate as f32) as usize;
    // Split buffer into chunks
    let chunks: Vec<&[f32]> = buffer.chunks(chunk_samples).collect();
    // Parallel compute mel specs for each chunk
    let mel_specs: Vec<Vec<Vec<f32>>> = chunks
        .into_par_iter()
        .map(|chunk| mel::mel_spectrogram_db(mel_config.clone(), chunk.to_vec()))
        .collect();
    // Stitch the time frames back together
    let full_spec: Vec<Vec<f32>> = mel_specs.into_iter().flatten().collect();
    // Plot the full spectrogram
    let image = mel::plot_mel_spec(full_spec, colors::Colormap::Magma, 1920, 256);
    image.save(output_path).unwrap();
}
