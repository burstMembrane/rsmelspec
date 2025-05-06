use image::ImageBuffer;
use image::Rgb;
use num::complex::Complex;
use rayon::prelude::*;
use std::time::Instant;

use crate::colors;
pub struct SpectrogramConfig {
    onesided: bool,
}
impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self { onesided: true }
    }
}
impl SpectrogramConfig {
    pub fn new(onesided: bool) -> Self {
        Self { onesided }
    }
}
impl Clone for SpectrogramConfig {
    fn clone(&self) -> Self {
        Self {
            onesided: self.onesided,
        }
    }
}
pub struct MelConfig {
    sample_rate: f32,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    top_db: f32,
    spectrogram_config: SpectrogramConfig,
}
impl Clone for MelConfig {
    fn clone(&self) -> Self {
        Self {
            sample_rate: self.sample_rate,
            n_fft: self.n_fft,
            win_length: self.win_length,
            hop_length: self.hop_length,
            f_min: self.f_min,
            f_max: self.f_max,
            n_mels: self.n_mels,
            top_db: self.top_db,
            spectrogram_config: self.spectrogram_config.clone(),
        }
    }
}

impl MelConfig {
    pub fn new(
        sample_rate: f32,
        n_fft: usize,
        win_length: usize,
        hop_length: usize,
        f_min: f32,
        f_max: f32,
        n_mels: usize,
        top_db: f32,
        spectrogram_config: SpectrogramConfig,
    ) -> Self {
        Self {
            sample_rate,
            n_fft,
            win_length,
            hop_length,
            f_min,
            f_max,
            n_mels,
            top_db,
            spectrogram_config,
        }
    }
}
pub fn mel_spectrogram_db(config: MelConfig, waveform: Vec<f32>) -> Vec<Vec<f32>> {
    let start = Instant::now();
    let top_db = config.top_db;
    let mel_spec: Vec<Vec<f32>> = mel_spectrogram(config, waveform);
    let result: Vec<Vec<f32>> = amplitude_to_db(mel_spec, top_db);
    println!("mel_spectrogram_db elapsed: {:?}", start.elapsed());
    result
}

fn mel_spectrogram(config: MelConfig, waveform: Vec<f32>) -> Vec<Vec<f32>> {
    let start = Instant::now();
    let spectrogram = spectrogram(
        waveform,
        config.n_fft,
        config.win_length,
        config.hop_length,
        config.spectrogram_config.onesided,
    );
    let n_freqs = spectrogram[0].len();
    let fbanks = mel_filter_bank(
        n_freqs,
        config.f_min,
        config.f_max,
        config.n_mels,
        config.sample_rate,
    );

    let mel_spec: Vec<Vec<f32>> = spectrogram
        .into_par_iter()
        .map(|spec_row| {
            (0..config.n_mels)
                .map(|j| {
                    spec_row
                        .iter()
                        .zip(fbanks.iter())
                        .map(|(&s, fbank)| s * fbank[j])
                        .sum()
                })
                .collect()
        })
        .collect();

    println!("mel_spectrogram time: {:?}", start.elapsed());
    mel_spec
}

fn spectrogram(
    waveform: Vec<f32>,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    onesided: bool,
) -> Vec<Vec<f32>> {
    let start = Instant::now();
    println!("n_fft: {}", n_fft);
    println!("win_length: {}", win_length);
    println!("hop_length: {}", hop_length);
    assert!(n_fft >= win_length);
    assert!(win_length >= hop_length);

    let pad_size = (win_length - hop_length) / 2;

    let padded_waveform: Vec<f32> = vec![0.0; pad_size]
        .into_iter()
        .chain(waveform)
        .chain(vec![0.0; pad_size])
        .collect();

    let window_fn = hann_window(win_length);
    let indices: Vec<usize> = (0..padded_waveform.len() - win_length - pad_size)
        .step_by(hop_length)
        .collect();

    let full_spec: Vec<Vec<f32>> = indices
        .into_par_iter()
        .map(|i| {
            // window + FFT + magnitude
            let mut windowed_input = Vec::with_capacity(win_length);
            for (a, w) in padded_waveform[i..i + win_length]
                .iter()
                .zip(window_fn.iter())
            {
                windowed_input.push(a * w);
            }
            fft(windowed_input, n_fft)
                .into_iter()
                .map(|c| c.norm())
                .collect()
        })
        .collect();

    if onesided {
        let result = full_spec
            .iter()
            .map(|row| row.iter().take(n_fft / 2 + 1).cloned().collect())
            .collect();
        println!("spectrogram time: {:?}", start.elapsed());
        result
    } else {
        println!("spectrogram time: {:?}", start.elapsed());
        full_spec
    }
}

fn fft(input: Vec<f32>, n_fft: usize) -> Vec<Complex<f32>> {
    let num_samples = input.len();
    assert!(n_fft.is_power_of_two(), "n_fft must be a power of 2");
    assert!(
        num_samples <= n_fft,
        "n must be less than or equal to n_fft"
    );

    if num_samples <= 1 {
        return input.into_iter().map(|x| Complex::new(x, 0.0)).collect();
    }

    let padded_input = if num_samples < n_fft {
        let padding = vec![0.0; n_fft - num_samples];
        input.clone().into_iter().chain(padding).collect()
    } else {
        input.clone()
    };

    // Split into even and odd parts
    let even: Vec<f32> = padded_input.iter().step_by(2).cloned().collect();
    let odd: Vec<f32> = padded_input.iter().skip(1).step_by(2).cloned().collect();

    // Recursive FFT on even and odd parts
    let even_fft = fft(even, n_fft / 2);
    let odd_fft = fft(odd, n_fft / 2);

    // Combine results
    let mut output = vec![Complex::new(0.0, 0.0); n_fft];
    for k in 0..(n_fft / 2) {
        let t = odd_fft[k]
            * Complex::from_polar(1.0, -2.0 * std::f32::consts::PI * k as f32 / n_fft as f32);
        output[k] = even_fft[k] + t;
        output[k + n_fft / 2] = even_fft[k] - t; // Exploit symmetry
    }
    output
}

fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (length - 1) as f32).cos()))
        .collect()
}

fn mel_filter_bank(
    n_freqs: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    sample_rate: f32,
) -> Vec<Vec<f32>> {
    let f_nyquist = sample_rate / 2.0;

    let all_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| f_nyquist * i as f32 / (n_freqs - 1) as f32)
        .collect(); // (n_freqs,)

    let m_min = hz_to_mel(f_min);
    let m_max = hz_to_mel(f_max);

    let m_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32)
        .collect(); // (n_mels + 2,)

    let f_points: Vec<f32> = m_points.iter().map(|&mel| mel_to_hz(mel)).collect();

    let f_diff: Vec<f32> = f_points
        .iter()
        .skip(1)
        .zip(f_points.iter().take(f_points.len() - 1))
        .map(|(f2, f1)| f2 - f1)
        .collect(); // (n_mels + 1,)

    let slopes: Vec<Vec<f32>> = all_freqs
        .iter()
        .map(|&f| f_points.iter().map(|&fp| fp - f).collect())
        .collect(); // (n_freqs, n_mels + 2)

    let down_slopes: Vec<Vec<f32>> = slopes
        .iter()
        .map(|slope_slice| {
            slope_slice
                .iter()
                .take(n_mels)
                .zip(f_diff.iter().take(n_mels))
                .map(|(slope, &diff)| -1.0 * slope / diff)
                .collect()
        })
        .collect(); // (n_freqs, n_mels)

    let up_slopes: Vec<Vec<f32>> = slopes
        .iter()
        .map(|slope_slice| {
            slope_slice
                .iter()
                .skip(2)
                .take(n_mels)
                .zip(f_diff.iter().skip(1).take(n_mels))
                .map(|(slope, &diff)| slope / diff)
                .collect()
        })
        .collect();

    let mut fbanks: Vec<Vec<f32>> = up_slopes
        .iter()
        .zip(down_slopes.iter())
        .map(|(up, down)| {
            let row = down
                .iter()
                .zip(up.iter())
                .map(|(&d, &u)| d.min(u).max(0.0)) // Use both up and down slopes
                .collect::<Vec<f32>>();
            row
        })
        .collect();

    // Apply Slaney normalization
    for i in 0..n_mels {
        let enorm = 2.0 / (f_points[i + 2] - f_points[i]);
        for fbank in fbanks.iter_mut() {
            fbank[i] *= enorm;
        }
    }

    fbanks
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}
pub fn plot_mel_spec(
    mel_spec: Vec<Vec<f32>>,
    cmap: colors::Colormap,
    width: usize,
    height: usize,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let start = Instant::now();
    let time_steps = mel_spec.len(); // X-axis
    let mel_bands = mel_spec[0].len(); // Y-axis

    println!("width: {}", time_steps);
    println!("height: {}", mel_bands);

    let flat_vals: Vec<f32> = mel_spec
        .iter()
        .flat_map(|row| row.iter().cloned())
        .collect();
    let smin = flat_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let smax = flat_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Precompute the colormap lookup table (256 entries)
    let color_map = colors::precompute_colormap(&cmap);

    let mut image = ImageBuffer::new(time_steps as u32, mel_bands as u32);
    for t in 0..time_steps {
        for m in 0..mel_bands {
            let norm = (mel_spec[t][m] - smin) / (smax - smin + f32::EPSILON);
            let index = (norm * 255.0).round().clamp(0.0, 255.0) as usize;
            let color = color_map[index];
            // Flip vertically so lowest freq is at bottom
            image.put_pixel(t as u32, (mel_bands - 1 - m) as u32, Rgb(color));
        }
    }
    println!("original width: {}", image.width());
    println!("original height: {}", image.height());
    // resize the image to the given width and height
    image = image::imageops::resize(
        &image,
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );

    println!("resized width: {}", image.width());
    println!("resized height: {}", image.height());

    println!("plot_mel_spec elapsed: {:?}", start.elapsed());
    image
}

fn amplitude_to_db(amplitudes: Vec<Vec<f32>>, top_db: f32) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    use std::time::Instant;
    let start = Instant::now();
    // Stage 1: power -> dB conversion (parallel)
    let dbs: Vec<Vec<f32>> = amplitudes
        .into_par_iter()
        .map(|row| {
            row.into_iter()
                .map(|amp| 20.0 * amp.max(1e-10).log10())
                .collect()
        })
        .collect();

    // Compute global max (serial, since flattening and max is fast)
    let max_db = dbs
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Stage 2: apply top_db clipping (parallel)
    let clipped: Vec<Vec<f32>> = dbs
        .into_par_iter()
        .map(|row| row.into_iter().map(|db| db.max(max_db - top_db)).collect())
        .collect();

    println!("amplitude_to_db elapsed: {:?}", start.elapsed());
    clipped
}

fn _assert_complex_eq(left: Complex<f32>, right: Complex<f32>) {
    // tolerance for floating-point comparison
    const EPSILON: f32 = 1e-5;

    assert!(
        (left.re - right.re).abs() < EPSILON,
        "left: {:?}, right: {:?}",
        left,
        right
    );
    assert!(
        (left.im - right.im).abs() < EPSILON,
        "left: {:?}, right: {:?}",
        left,
        right
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::complex::Complex;

    #[test]
    fn test_fft_constant() {
        let input = vec![1.0, 0.0, 0.0, 0.0]; // Changed to real input
        let output = fft(input, 4);
        _assert_complex_eq(output[0], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[2], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[3], Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_fft_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // Changed to real input
        let output = fft(input, 4);
        _assert_complex_eq(output[0], Complex::new(10.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(-2.0, 2.0));
        _assert_complex_eq(output[2], Complex::new(-2.0, 0.0));
        _assert_complex_eq(output[3], Complex::new(-2.0, -2.0));
    }

    #[test]
    fn test_fft_with_length_eight() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]; // Changed to real input
        let output = fft(input, 8);
        _assert_complex_eq(output[0], Complex::new(10.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(-0.41421356, -7.24264069));
        _assert_complex_eq(output[2], Complex::new(-2.0, 2.0));
        _assert_complex_eq(output[3], Complex::new(2.41421356, -1.24264069));
        _assert_complex_eq(output[4], Complex::new(-2.0, 0.0));
        _assert_complex_eq(output[5], Complex::new(2.41421356, 1.24264069));
        _assert_complex_eq(output[6], Complex::new(-2.0, -2.0));
        _assert_complex_eq(output[7], Complex::new(-0.41421356, 7.24264069));
    }
}
