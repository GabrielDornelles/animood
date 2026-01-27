use anyhow::Result;
use rayon::{prelude::*};
use crate::types::{AnimeResult};
use crate::embedder::embed;
use crate::AppState;

// Dataset normalization constants (computed offline)
pub const SCORE_MIN: f32 = 5.04;
pub const SCORE_MAX: f32 = 9.29;

pub const MEMBERS_LOG_MIN: f32 = 9.211;
pub const MEMBERS_LOG_MAX: f32 = 15.267;

pub const FAVORITES_LOG_MIN: f32 = 0.693;
pub const FAVORITES_LOG_MAX: f32 = 12.413;


fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm(value: f32, min: f32, max: f32) -> f32 {
    if max <= min {
        0.0
    } else {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }
}

#[inline]
fn log_norm(value: u32, min_log: f32, max_log: f32) -> f32 {
    let v = (value as f32 + 1.0).ln();
    norm(v, min_log, max_log)
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


// pub fn query_anime_with_user_taste(
//     state: &AppState,
//     taste_query: Option<Vec<f32>>,
//     k: usize,
// ) -> Result<Vec<AnimeResult>> {

//     if let Some(query_vec) = taste_query {
//         let top = search_similarity(
//             &query_vec,
//             &embeddings.embeddings,
//             20 * 2,
//         )
//     }

// }

pub fn query_anime(
    state: &AppState,
    query: &str,
    k: usize,
) -> Result<Vec<AnimeResult>> {
    let query_emb = &embed(
        &state.model,
        &state.tokenizer,
        &[query.to_string()],
    )?[0];

    let top = search_similarity(query_emb, &state.embeddings.embeddings, k * 2);

    let mut results: Vec<AnimeResult> = top
        .into_iter()
        .map(|(idx, embedding_score)| {
            let final_score =
                0.6 * embedding_score +
                0.20 * norm(state.embeddings.scores[idx], SCORE_MIN, SCORE_MAX) +
                0.10 * log_norm(
                    state.embeddings.members[idx],
                    MEMBERS_LOG_MIN,
                    MEMBERS_LOG_MAX,
                ) +
                0.10 * log_norm(
                    state.embeddings.favorites[idx],
                    FAVORITES_LOG_MIN,
                    FAVORITES_LOG_MAX,
                );

            AnimeResult {
                title: state.embeddings.names[idx].clone(),
                score: final_score,
                image_url: state.embeddings.picture_urls[idx].clone(),
                llm_description: state.embeddings.llm_description[idx].clone(),
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.truncate(20);

    Ok(results)
}


// pub fn query_anime(query: &str, k: usize) -> Result<Vec<AnimeResult>> {
//     let loaded = AnimeEmbeddings::load_bin("embeddings.bin")?;

//     let query_emb = &embed(vec![query.to_string()])?[0];
//     let top = search_similarity(&query_emb, &loaded.embeddings, k * 2);

//     let mut results: Vec<AnimeResult> = top
//         .into_iter()
//         .map(|(idx, embedding_score)| {
//             let norm_score = norm(
//                 loaded.scores[idx],
//                 SCORE_MIN,
//                 SCORE_MAX,
//             );

//             let norm_members = log_norm(
//                 loaded.members[idx],
//                 MEMBERS_LOG_MIN,
//                 MEMBERS_LOG_MAX,
//             );

//             let norm_favorites = log_norm(
//                 loaded.favorites[idx],
//                 FAVORITES_LOG_MIN,
//                 FAVORITES_LOG_MAX,
//             );

//             let final_score =
//                 0.55 * embedding_score +
//                 0.20 * norm_score +
//                 0.15 * norm_members +
//                 0.10 * norm_favorites;

//             AnimeResult {
//                 title: loaded.names[idx].clone(),
//                 score: final_score,
//                 image_url: loaded.picture_urls[idx].clone(),
//                 llm_description: loaded.llm_description[idx].clone()
//             }
//         })
//         .collect();
    
//     results.sort_by(|a, b| {
//         b.score
//             .partial_cmp(&a.score)
//             .unwrap()
//     });
//     results.truncate(k);

//     Ok(results)
// }

