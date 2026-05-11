use std::sync::Arc;
use tracing_subscriber::EnvFilter;
use anyhow::Result;
use axum::{Router, routing::{post, get}};
use tower_http::cors::{CorsLayer, Any};
use animood::{
    AppState,
    api::{query_handler, compatibility_handler, archetype_handler},
    model::build_model_and_tokenizer_from_disk,
    types::AnimeEmbeddings,
    social::ProfileCache,
};
use tracing::info;
use axum::Json;
use serde_json::json;

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    info!("Loading model and tokenizer...");
    let model_dir = std::env::var("MODEL_DIR")
        .unwrap_or_else(|_| "./app/models/jina-embeddings-v2-small-en".into());
    let (model, tokenizer) = build_model_and_tokenizer_from_disk(&model_dir)?;

    info!("Loading embeddings.bin...");
    let embeddings = AnimeEmbeddings::load_bin("embeddings.bin")?;

    let state = Arc::new(AppState {
        model,
        tokenizer,
        embeddings,
        profile_cache: ProfileCache::new(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/query",         post(query_handler))
        .route("/compatibility", post(compatibility_handler))
        .route("/archetype",     post(archetype_handler))
        .route("/health",        get(health))
        .with_state(state)
        .layer(cors);

    info!("Server running at http://0.0.0.0:3000");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}
