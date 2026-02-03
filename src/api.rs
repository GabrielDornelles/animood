use std::sync::Arc;
use tracing::{info, error};

use axum::{
    extract::State,
    http::StatusCode,
    Json,
};

use crate::{
    AppState,
    query_anime,
    query_anime_with_user_mal,
    types::{QueryRequest},
};

use crate::search::RecommendationResponse;

pub async fn query_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<RecommendationResponse>, StatusCode> {
    
    info!(
        query = %req.query,
        k = req.k,
        user = ?req.username,
        "received query request"
    );
    let k = req.k.unwrap_or(100);
  
    if let Some(username) = req.username {
         query_anime_with_user_mal(
            &state, 
            username
        ).await
        .map(Json)
            .map_err(|e| {
                error!("query error: {e:?}");
                StatusCode::INTERNAL_SERVER_ERROR
            })

    }
    else {
        query_anime(&state, &req.query, k)
        .map(Json)
        .map_err(|e| {
            error!("query error: {e:?}");
            StatusCode::INTERNAL_SERVER_ERROR
        })

    }
}
