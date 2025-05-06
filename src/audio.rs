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
