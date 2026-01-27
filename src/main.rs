// use std::sync::Arc;

// use anyhow::Result;
// use axum::{Router, routing::{post, get}};
// use tower_http::cors::{CorsLayer, Any};
// use animood::{
//     AppState,
//     api::query_handler,
//     model::build_model_and_tokenizer_from_disk,
//     types::AnimeEmbeddings,
// };

// use axum::{Json};
// use serde_json::json;

// async fn health() -> Json<serde_json::Value> {
//     Json(json!({"status": "ok"}))
// }


// #[tokio::main]
// async fn main() -> Result<()> {
//     println!("Loading model and tokenizer...");
//     let model_dir = std::env::var("MODEL_DIR")
//     .unwrap_or_else(|_| "./app/models/jina-embeddings-v2-small-en".into());
//     let (model, tokenizer) = build_model_and_tokenizer_from_disk(&model_dir)?;

//     println!("Loading embeddings.bin...");
//     let embeddings = AnimeEmbeddings::load_bin("embeddings.bin")?;

//     let state = Arc::new(AppState {
//         model,
//         tokenizer,
//         embeddings,
//     });

//     let cors = CorsLayer::new()
//         .allow_origin(Any) 
//         .allow_methods(Any)
//         .allow_headers(Any);


//     let app = Router::new()
//         .route("/query", post(query_handler))
//         .route("/health", get(health))
//         .with_state(state)
//         .layer(cors);

//     println!("Server running at http://0.0.0.0:3000");
//     use tokio::net::TcpListener;

//     let listener = TcpListener::bind("0.0.0.0:3000").await?;
//     axum::serve(listener, app).await?;


//     Ok(())
// }

use anyhow::{Result};
// use animood::{
//     build_bin_struct_from_json, 
//     //query_anime,
// };
use reqwest::header::{ACCEPT, USER_AGENT};
use animood::mal_types::parse_mal_list;
use animood::{AnimeEmbeddings, search_similarity};
use animood::types::{AnimeResult};

// =============================
// Vector utilities
// =============================

fn clip_and_normalize(v: &mut [f32], clip: f32) {
    // Clip values to avoid dominance
    for x in v.iter_mut() {
        *x = x.clamp(-clip, clip);
    }

    // L2 normalize (cosine space)
    // let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    // if norm > 0.0 {
    //     for x in v.iter_mut() {
    //         *x /= norm;
    //     }
    // }
}

// =============================
// Weighted centroid
// =============================

fn weighted_centroid(pairs: &[(&[f32], f32)]) -> Option<Vec<f32>> {
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
    clip_and_normalize(&mut acc, 10.0);

    Some(acc)
}

// =============================
// Taste query builder
// =============================

const POS_WEIGHT: f32 = 1.0;
const NEG_WEIGHT: f32 = 1.0;
const CLIP_VALUE: f32 = 10.0;

fn build_taste_query(
    positive: Option<Vec<f32>>,
    negative: Option<Vec<f32>>,
) -> Option<Vec<f32>> {
    match (positive, negative) {
        // No signal
        (None, None) => None,

        // Only positive taste
        (Some(mut pos), None) => {
            clip_and_normalize(&mut pos, CLIP_VALUE);
            Some(pos)
        }

        // Only negative taste (invert)
        (None, Some(mut neg)) => {
            for v in &mut neg {
                *v = -*v;
            }
            clip_and_normalize(&mut neg, CLIP_VALUE);
            Some(neg)
        }

        // Positive + negative
        (Some(mut pos), Some(neg)) => {
            if pos.len() != neg.len() {
                // Fallback: trust positives
                clip_and_normalize(&mut pos, CLIP_VALUE);
                return Some(pos);
            }

            for i in 0..pos.len() {
                pos[i] = POS_WEIGHT * pos[i] - NEG_WEIGHT * neg[i];
            }

            clip_and_normalize(&mut pos, CLIP_VALUE);
            Some(pos)
        }
    }
}


pub const SCORE_MIN: f32 = 5.04;
pub const SCORE_MAX: f32 = 9.29;

pub const MEMBERS_LOG_MIN: f32 = 9.211;
pub const MEMBERS_LOG_MAX: f32 = 15.267;

pub const FAVORITES_LOG_MIN: f32 = 0.693;
pub const FAVORITES_LOG_MAX: f32 = 12.413;

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

#[tokio::main]
async fn main() -> Result<()> { 
    //build_bin_struct_from_json("./llm_enriched.json")?;
    let embeddings = AnimeEmbeddings::load_bin("embeddings.bin")?;
    let client = reqwest::Client::new();

    let res = client
        .get("https://myanimelist.net/animelist/Dornelles/load.json?status=7&offset=0")
        .header(USER_AGENT, "Mozilla/5.0")
        .header(ACCEPT, "application/json")
        .send()
        .await?;

    let body = res.text().await?;
    let entries = parse_mal_list(&body)?;
    // for e in entries.iter() {
    //     println!("{} ({})", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id);
    // }

    let mut personal_favorites = Vec::new();
    let mut unliked = Vec::new();

    let mut watched = Vec::new();

    let mut positive_pairs: Vec<(&[f32], f32)> = Vec::new();
    let mut negative_pairs: Vec<(&[f32], f32)> = Vec::new();

    for item in entries.iter() {
        if item.status == Some(2) {
            watched.push(item.anime_id);
            if let Some(diff) = item.anime_score_diff {

                if diff > 0.5 && diff.abs() < 99.0 {
                    personal_favorites.push(item);
                    let embedding = embeddings.get_embedding(item.anime_id)?;
                    if let Some(embedding_vec) = embedding {
                         positive_pairs.push((embedding_vec, diff));
                    } 
                }
                
                if diff < - 0.5 && diff.abs() < 99.0{
                    unliked.push(item);
                    let embedding = embeddings.get_embedding(item.anime_id)?;
                    if let Some(embedding_vec) = embedding {
                         negative_pairs.push((embedding_vec, diff));
                    } 
                }
            }
        }
    }

    let positive_taste = weighted_centroid(&positive_pairs);
    let negative_taste = weighted_centroid(&negative_pairs);

    let taste_query = build_taste_query(positive_taste, negative_taste).unwrap();

    let top = search_similarity(
        &taste_query,
        &embeddings.embeddings,
        100 * 2,
    );


    let mut results: Vec<AnimeResult> = top
        .into_iter()
        .map(|(idx, embedding_score)| {
            let final_score =
                0.8 * embedding_score +
                0.05 * norm(embeddings.scores[idx], SCORE_MIN, SCORE_MAX) +
                0.05 * log_norm(
                    embeddings.members[idx],
                    MEMBERS_LOG_MIN,
                    MEMBERS_LOG_MAX,
                ) +
                0.1 * log_norm(
                    embeddings.favorites[idx],
                    FAVORITES_LOG_MIN,
                    FAVORITES_LOG_MAX,
                );

            AnimeResult {
                title: embeddings.names[idx].clone(),
                score: final_score,
                image_url: embeddings.picture_urls[idx].clone(),
                llm_description: embeddings.llm_description[idx].clone(),
                mal_id: embeddings.ids[idx]
            }
        })
        .collect();
    
    
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.truncate(50);
    results.retain(|item| !watched.contains(&item.mal_id));

    // clear animes that have been watched

    // for item in results {
    //     if item.mal_id in watched {
    //         // drop the item from results
    //     }
    // }

  

        // same post-processing as before
    

    //println!("{}", body);
    println!("Personal favorites:");
    for e in personal_favorites.iter() {
        println!("{} ({}) - Score diff: {:?}", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id, e.anime_score_diff);
    }

    println!("\nPeople like it, but you didn't:");
    for e in unliked.iter() {
         println!("{} ({}) - Score diff: {:?}", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id, e.anime_score_diff);
    }

    println!("\n Recommendations for you:");

    for item in results {
        let title = item.title;
        println!("{title}")
    }
    // for e in &results.iter(){
    //     println!("{e.title}");
    // }


    Ok(())
}
