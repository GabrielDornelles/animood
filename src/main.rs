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
use animood::{AnimeEmbeddings};
use animood::query_anime_with_user_mal;

#[tokio::main]
async fn main() -> Result<()> { 
    //build_bin_struct_from_json("./llm_enriched.json")?;
    let embeddings = AnimeEmbeddings::load_bin("embeddings.bin")?;
    let username = "Dornelles";
    let recommendations = query_anime_with_user_mal(embeddings, username).await;
    Ok(())
}
