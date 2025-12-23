use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    Json,
};

use crate::{
    AppState,
    query_anime,
    types::{QueryRequest, AnimeResult},
};

pub async fn query_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<Vec<AnimeResult>>, StatusCode> {
    println!("200: Received query: {}", req.query);
    let k = req.k.unwrap_or(100);

    query_anime(&state, &req.query, k)
        .map(Json)
        .map_err(|e| {
            eprintln!("query error: {e:?}");
            StatusCode::INTERNAL_SERVER_ERROR
        })
}
