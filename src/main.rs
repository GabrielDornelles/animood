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
use animood::{AnimeEmbeddings};

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

    let mut positive_pairs: Vec<(f32, f32)> = Vec::new();
    let mut negative_pairs: Vec<(f32, f32)> = Vec::new();

    for item in entries.iter() {
        if item.status == Some(2) {
            if let Some(diff) = item.anime_score_diff {

                if diff > 1.0 {
                    personal_favorites.push(item);
                    //EmbeddingScorePair{embedding: 3.2, score: 2.5};
                    positive_pairs.push((3.2, 3.1));

                }
                
                if diff < - 1.0 {
                    unliked.push(item);
                }
            }
        }
    }

    //println!("{}", body);
    println!("Personal favorites:");
    for e in personal_favorites.iter() {
        println!("{} ({}) - Score diff: {:?}", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id, e.anime_score_diff);
    }

    println!("\nPeople like it, but you didn't:");
    for e in unliked.iter() {
         println!("{} ({}) - Score diff: {:?}", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id, e.anime_score_diff);
    }


    Ok(())
}
