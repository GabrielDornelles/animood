use anyhow::Result;
use serde::Serialize;

use crate::mal_functions::get_anime_list;
use crate::types::AnimeEmbeddings;
use crate::types::{AnimeResult};
use crate::embedder::embed;
use crate::AppState;
use crate::vec_ops::{search_similarity, norm, log_norm, build_taste_query, weighted_centroid};
use crate::mal_functions::{gather_mal_user_data, genre_reason_map, MalUserData};

// Dataset normalization constants (computed offline)
pub const SCORE_MIN: f32 = 5.04;
pub const SCORE_MAX: f32 = 9.29;

pub const MEMBERS_LOG_MIN: f32 = 9.211;
pub const MEMBERS_LOG_MAX: f32 = 15.267;

pub const FAVORITES_LOG_MIN: f32 = 0.693;
pub const FAVORITES_LOG_MAX: f32 = 12.413;

pub fn query_anime(
    state: &AppState,
    query: &str,
    k: usize,
) -> Result<RecommendationResponse> {
    let query_emb = &embed(
        &state.model,
        &state.tokenizer,
        &[query.to_string()],
    )?[0];

    let top = search_similarity(query_emb, &state.embeddings.embeddings, k * 2);

    let mut global_recommendations: Vec<AnimeResult> = top
        .into_iter()
        .map(|(idx, embedding_score)| {
            let final_score =
                0.7 * embedding_score +
                0.10 * norm(state.embeddings.scores[idx], SCORE_MIN, SCORE_MAX) +
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
                mal_id: state.embeddings.ids[idx]
            }
        })
        .collect();

    global_recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    global_recommendations.truncate(20);

    
    //Ok(results)
    Ok(
        RecommendationResponse {
            global_recommendations: global_recommendations,
            genre_recommendations: Vec::new(),
            global_genres: Vec::new(),
            favorite_genres: Vec::new(),
            favorites: Vec::new(),
            unpreferred: Vec::new(),
        }
    )
}


fn build_ranked_results(
    top: Vec<(usize, f32)>,
    embeddings: &AnimeEmbeddings,
    user_data: &MalUserData,
    limit: usize,
) -> Result<Vec<AnimeResult>> {
    let mut results: Vec<AnimeResult> = top
        .into_iter()
        .map(|(idx, embedding_score)| {
            let final_score =
                0.8 * embedding_score +
                0.02 * norm(embeddings.scores[idx], SCORE_MIN, SCORE_MAX) +
                0.08 * log_norm(
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
    results.retain(|item| !user_data.watched.contains(&item.mal_id) && !user_data.dropped.contains(&item.mal_id));
    results.truncate(limit);
    Ok(results)
}


pub async fn query_anime_with_user_mal(
    state: &AppState,
    // embeddings: AnimeEmbeddings,
    username: String,
) -> Result<RecommendationResponse> {
    let embeddings = &state.embeddings;//AnimeEmbeddings::load_bin("embeddings.bin")?;

    let entries = get_anime_list(&username).await?;
    let user_data = gather_mal_user_data(&entries, &embeddings)?;

    let positive_taste = weighted_centroid(&user_data.higher_than_avg_scored_anime);
    let negative_taste = weighted_centroid(&user_data.lower_than_avg_scored_anime);

    let taste_query = build_taste_query(positive_taste, negative_taste).unwrap();

    let top = search_similarity(
        &taste_query,
        &embeddings.embeddings,
        100 * 2,
    );
    let global_recommendations = build_ranked_results(
        top, &embeddings, &user_data, 20
    )?;
  
    // println!("\nWatched Genres:");
    // for item in &user_data.global_genres_sorted {
    //     println!("{} - appears: {}", item.1.name, item.1.count)
    // }

    // println!("\nPrefered Genres:");
    // for item in &user_data.favorite_genres_sorted{
    //     println!("{} - appears: {}", item.1.name, item.1.count)
    // }

    let mut genre_recommendations = Vec::new();
    let genre_reasons = genre_reason_map();
    for top_genre in &user_data.top_5_genres_ratio {
       
        let genre_pairs: Vec<_> = user_data.higher_than_avg_scored_anime
            .iter()
            .filter(|item| item.genres.contains(&top_genre.id))
            .collect();

        let genre_taste = weighted_centroid(genre_pairs).unwrap();

        let top = search_similarity(
            &genre_taste,
            &embeddings.embeddings,
            100 * 2,
        );

        let recommendations = build_ranked_results(top, &embeddings, &user_data, 20)?;
 
        let reason = genre_reasons
            .get(&top_genre.id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Recommendations for: {}", top_genre.name));
        
        genre_recommendations.push(
            GenreRecommendation {
                genre_id: top_genre.id,
                genre_name: top_genre.name.clone(),
                reason,
                recommendations,
            });

        // println!("\n{}", reason);

        // for item in &results {
        //     let title = &item.title;
        //     println!("{title}")
        // }

    }

    let global_genres = user_data
    .global_genres_sorted
    .iter()
    .map(|(id, stat)| GenreCount {
        id: *id,
        name: stat.name.clone(),
        count: stat.count,
    })
    .collect();

    let favorite_genres = user_data
        .favorite_genres_sorted
        .iter()
        .map(|(id, stat)| GenreCount {
            id: *id,
            name: stat.name.clone(),
            count: stat.count,
        })
        .collect();
    
    let favorites: Vec<AnimeScoreDiff> = user_data.favorites
        .iter()
        .map(|e| AnimeScoreDiff {
            anime_id: e.anime_id,
            title: e.anime_title.clone(),
            score_diff: e.anime_score_diff,
            image_url: e.anime_image_path.clone()
            
        })
        .collect();

    let unpreferred: Vec<AnimeScoreDiff> = user_data.unpreferred
        .iter()
        .map(|e| AnimeScoreDiff {
            anime_id: e.anime_id,
            title: e.anime_title.clone(),
            score_diff: e.anime_score_diff,
            image_url: e.anime_image_path.clone()
            //image_url: e.image_url,
            // picutre: e.images
        })
        .collect();
    
    Ok(RecommendationResponse {
        global_recommendations,
        genre_recommendations,
        global_genres,
        favorite_genres,
        favorites,
        unpreferred,
    })


    // println!("\nYou liked more than most people:");
    // for e in user_data.favorites.iter() {
    //     println!("{} ({}) - Score diff: {:?}", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id, e.anime_score_diff.unwrap());
    //     // println!("genres: {:?}", e.genres);
    // }

    // println!("\nPeople like it, but you didn't:");
    // for e in user_data.unpreferred.iter() {
    //      println!("{} ({}) - Score diff: {:?}", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id, e.anime_score_diff.unwrap());
    // }

    // println!("\nRecommendations for you:");

    // for item in &results {
    //     let title = &item.title;
    //     println!("{title}")
    // }
  
    // Ok(results)

}

#[derive(Serialize)]
#[derive(Debug)]
pub struct AnimeScoreDiff {
    pub anime_id: u32,
    pub title: Option<String>,
    pub score_diff: Option<f32>,
    pub image_url: Option<String>
}


#[derive(Serialize)]
#[derive(Debug)]
pub struct GenreCount {
    pub id: u32,
    pub name: String,
    pub count: usize,
}


#[derive(Serialize)]
#[derive(Debug)]
pub struct GenreRecommendation {
    pub genre_id: u32,
    pub genre_name: String,
    pub reason: String,
    pub recommendations: Vec<AnimeResult>,
}

#[derive(Serialize)]
#[derive(Debug)]
pub struct RecommendationResponse {
    pub global_recommendations: Vec<AnimeResult>,

    pub genre_recommendations: Vec<GenreRecommendation>,

    pub global_genres: Vec<GenreCount>,
    pub favorite_genres: Vec<GenreCount>,

    pub favorites: Vec<AnimeScoreDiff>,
    pub unpreferred: Vec<AnimeScoreDiff>,
}
