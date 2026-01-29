use rayon::{prelude::*};
// For computing taste vectors
const POS_WEIGHT: f32 = 1.0;
const NEG_WEIGHT: f32 = 1.0;
const CLIP_VALUE: f32 = 10.0;


pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn norm(value: f32, min: f32, max: f32) -> f32 {
    if max <= min {
        0.0
    } else {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }
}

#[inline]
pub fn log_norm(value: u32, min_log: f32, max_log: f32) -> f32 {
    let v = (value as f32 + 1.0).ln();
    norm(v, min_log, max_log)
}


pub fn clip(v: &mut [f32], clip: f32) {
    // Clip values to avoid dominance
    for x in v.iter_mut() {
        *x = x.clamp(-clip, clip);
    }
}


pub fn search_similarity(
    query: &[f32],
    embeddings: &[Vec<f32>],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = embeddings
        .par_iter()
        .enumerate()
        .map(|(i, emb)| (i, dot(query, emb)))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(k);
    scores
}


pub fn weighted_centroid(pairs: &[(&[f32], f32)]) -> Option<Vec<f32>> {
    if pairs.is_empty() {
        return None;
    }

    let dim = pairs[0].0.len();
    let mut acc = vec![0.0f32; dim];
    let mut weight_sum = 0.0f32;

    for (emb, weight) in pairs {
        // Defensive: ignore broken data
        if emb.len() != dim {
            continue;
        }

        let w = weight.abs();
        if w == 0.0 {
            continue;
        }

        weight_sum += w;
        for i in 0..dim {
            acc[i] += emb[i] * w;
        }
    }

    if weight_sum == 0.0 {
        return None;
    }

    for v in &mut acc {
        *v /= weight_sum;
    }

    // Keep centroid stable for similarity search
    clip(&mut acc, 10.0);

    Some(acc)
}


pub fn build_taste_query(
    positive: Option<Vec<f32>>,
    negative: Option<Vec<f32>>,
) -> Option<Vec<f32>> {
    match (positive, negative) {
        // No signal
        (None, None) => None,

        // Only positive taste
        (Some(mut pos), None) => {
            clip(&mut pos, CLIP_VALUE);
            Some(pos)
        }

        // Only negative taste (invert)
        (None, Some(mut neg)) => {
            for v in &mut neg {
                *v = -*v;
            }
            clip(&mut neg, CLIP_VALUE);
            Some(neg)
        }

        // Positive + negative
        (Some(mut pos), Some(neg)) => {
            if pos.len() != neg.len() {
                // Fallback: trust positives
                clip(&mut pos, CLIP_VALUE);
                return Some(pos);
            }

            for i in 0..pos.len() {
                pos[i] = POS_WEIGHT * pos[i] - NEG_WEIGHT * neg[i];
            }

            clip(&mut pos, CLIP_VALUE);
            Some(pos)
        }
    }
}