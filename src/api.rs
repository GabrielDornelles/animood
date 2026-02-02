use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    Json,
};

use crate::{
    AppState,
    query_anime,
    query_anime_with_user_mal,
    types::{QueryRequest, AnimeResult},
};

use crate::search::RecommendationResponse;

pub async fn query_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<RecommendationResponse>, StatusCode> {
    println!("200: Received query: {}", req.query);
    let k = req.k.unwrap_or(100);
    let username = req.username.unwrap_or("".to_string());

    query_anime_with_user_mal(
        &state, 
        username
    ).await
    .map(Json)
        .map_err(|e| {
            eprintln!("query error: {e:?}");
            StatusCode::INTERNAL_SERVER_ERROR
        })


    // query_anime(&state, &req.query, k)
    //     .map(Json)
    //     .map_err(|e| {
    //         eprintln!("query error: {e:?}");
    //         StatusCode::INTERNAL_SERVER_ERROR
    //     })
}
