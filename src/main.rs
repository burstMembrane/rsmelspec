use clap::Parser;
use clio::Input;
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

    let mel_spec = mel::mel_spectrogram_db(mel_config, buffer);
    let image = mel::plot_mel_spec(mel_spec, colors::Colormap::Inferno);
    image.save(output_path).unwrap();
}
