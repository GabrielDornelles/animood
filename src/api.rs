use std::sync::Arc;
use tracing::{info, error};
use serde::Deserialize;

use axum::{
    extract::State,
    http::StatusCode,
    Json,
};

use crate::{
    AppState,
    query_anime,
    query_anime_with_user_mal,
    types::QueryRequest,
    search::RecommendationResponse,
    social::{
        compute_compatibility, compute_archetype,
        CompatibilityResponse, ArchetypeResponse,
    },
};

// ---------------------------------------------------------------------------
// Existing recommendation handler (unchanged)
// ---------------------------------------------------------------------------

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
        query_anime_with_user_mal(&state, username)
            .await
            .map(Json)
            .map_err(|e| {
                error!("query error: {e:?}");
                StatusCode::INTERNAL_SERVER_ERROR
            })
    } else {
        query_anime(&state, &req.query, k)
            .map(Json)
            .map_err(|e| {
                error!("query error: {e:?}");
                StatusCode::INTERNAL_SERVER_ERROR
            })
    }
}

// ---------------------------------------------------------------------------
// POST /compatibility
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CompatibilityRequest {
    pub username_a: String,
    pub username_b: String,
}

pub async fn compatibility_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompatibilityRequest>,
) -> Result<Json<CompatibilityResponse>, StatusCode> {
    info!(
        user_a = %req.username_a,
        user_b = %req.username_b,
        "received compatibility request"
    );

    compute_compatibility(
        &state.profile_cache,
        &state.embeddings,
        &req.username_a,
        &req.username_b,
    )
    .await
    .map(Json)
    .map_err(|e| {
        error!("compatibility error: {e:?}");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

// ---------------------------------------------------------------------------
// POST /archetype
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ArchetypeRequest {
    pub username: String,
}

pub async fn archetype_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ArchetypeRequest>,
) -> Result<Json<ArchetypeResponse>, StatusCode> {
    info!(user = %req.username, "received archetype request");

    compute_archetype(
        &state.profile_cache,
        &state.embeddings,
        &req.username,
    )
    .await
    .map(Json)
    .map_err(|e| {
        error!("archetype error: {e:?}");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}
