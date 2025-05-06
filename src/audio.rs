use rubato::{FftFixedInOut, Resampler};

pub fn split(
    audio: &[f32],
    segment_length_secs: f32,
    overlap_secs: f32,
    sample_rate: u32,
) -> (Vec<Vec<f32>>, usize) {
    let total_samples = audio.len();
    let segment_samples = (segment_length_secs * sample_rate as f32) as usize;
    let overlap_samples = (overlap_secs * sample_rate as f32) as usize;
    let step = segment_samples.saturating_sub(overlap_samples);
    let num_segments = if step > 0 {
        (total_samples + step - 1) / step
    } else {
        1
    };

    let mut segments = Vec::with_capacity(num_segments);
    let mut padding = 0;
    for i in 0..num_segments {
        let start = i * step;
        let end = (start + segment_samples).min(total_samples);
        let mut segment: Vec<f32> = audio[start..end].to_vec();
        if segment.len() < segment_samples {
            padding = segment_samples - segment.len();
            segment.extend(std::iter::repeat(0.0).take(padding));
        }
        segments.push(segment);
    }
    (segments, padding)
}

#[allow(dead_code)]
/// Merges overlapping segments back into a single audio vector, using overlap-add.
pub fn merge(
    segments: &[Vec<f32>],
    overlap_secs: f32,
    padding: usize,
    sample_rate: u32,
) -> Vec<f32> {
    if segments.is_empty() {
        return Vec::new();
    }
    let segment_samples = segments[0].len();
    let overlap_samples = (overlap_secs * sample_rate as f32) as usize;
    let step = segment_samples.saturating_sub(overlap_samples);
    let total_len = step * (segments.len() - 1) + segment_samples;
    let mut audio = vec![0.0f32; total_len];
    let mut counts = vec![0.0f32; total_len];
    for (i, segment) in segments.iter().enumerate() {
        let start = i * step;
        for j in 0..segment_samples {
            if start + j < audio.len() {
                audio[start + j] += segment[j];
                counts[start + j] += 1.0;
            }
        }
    }
    // Normalize overlap-add
    for (a, c) in audio.iter_mut().zip(counts.iter()) {
        if *c > 0.0 {
            *a /= *c;
        }
    }
    // Remove padding from end if needed
    if padding > 0 && audio.len() >= padding {
        audio.truncate(audio.len() - padding);
    }
    audio
}

/// Resamples a mono audio buffer from `fs_in` to `fs_out` using FFT-based fixed-in/out resampling.
pub fn resample_audio(input: &[f32], fs_in: usize, fs_out: usize) -> Vec<f32> {
    // Choose a power-of-two chunk size for FFT
    let nbr_frames_in = 1024;
    // Build the resampler: (in_rate, out_rate, chunk_size, channels)
    let mut resampler = FftFixedInOut::<f32>::new(fs_in, fs_out, nbr_frames_in, 1)
        .expect("Failed to create FFT resampler");
    // How many frames the filter delays
    let delay = resampler.output_delay();
    // Compute exact target length
    let new_length = ((input.len() as f64) * (fs_out as f64) / (fs_in as f64)).round() as usize;

    let mut output: Vec<f32> = Vec::with_capacity(new_length + delay);
    let mut in_buf = Vec::with_capacity(nbr_frames_in);
    let mut pos = 0;

    // Process all full-sized chunks
    while pos + nbr_frames_in <= input.len() {
        in_buf.clear();
        in_buf.extend_from_slice(&input[pos..pos + nbr_frames_in]);
        let processed = resampler
            .process(&[in_buf.clone()], None)
            .expect("Resampling failed");
        output.extend_from_slice(&processed[0]);
        pos += nbr_frames_in;
    }

    // Process any remaining tail frames
    if pos < input.len() {
        in_buf.clear();
        in_buf.extend_from_slice(&input[pos..]);
        let processed = resampler
            .process_partial(Some(&[in_buf.clone()]), None)
            .expect("Resampling failed");
        output.extend_from_slice(&processed[0]);
    }

    // Flush internal filter delay
    while output.len() < new_length + delay {
        let tail = resampler
            .process_partial(None::<&[Vec<f32>]>, None)
            .expect("Resampling failed");
        output.extend_from_slice(&tail[0]);
    }

    // Drop the initial delay and truncate to the exact length
    let mut final_out = if output.len() > delay {
        output.split_off(delay)
    } else {
        Vec::new()
    };
    final_out.truncate(new_length);
    final_out
}
