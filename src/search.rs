use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::mal_functions::get_anime_list;
use crate::types::AnimeEmbeddings;
use crate::types::{AnimeResult};
use crate::embedder::embed;
use crate::AppState;
use crate::vec_ops::{search_similarity, norm, log_norm, build_taste_query, weighted_centroid};

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
                mal_id: state.embeddings.ids[idx]
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.truncate(20);

    Ok(results)
}


struct GenreStat {
    name: String,
    count: usize,
}

pub async fn query_anime_with_user_mal(
    //state: &AppState,
    embeddings: AnimeEmbeddings,
    username: &str,
) -> Result<Vec<AnimeResult>> {
    //let embeddings = &state.embeddings;//AnimeEmbeddings::load_bin("embeddings.bin")?;
    let entries = get_anime_list(username).await?;

    let mut personal_favorites = Vec::new();
    let mut unliked = Vec::new();

    let mut watched = Vec::new();
    let mut dropped = Vec::new();

    // embedding, diff_score, 
    let mut positive_pairs: Vec<(&[f32], f32)> = Vec::new();
    let mut negative_pairs: Vec<(&[f32], f32)> = Vec::new();

    let mut genre_hashmap: HashMap<u32, GenreStat> = HashMap::new();

    let mut genre_hashmap_favorites: HashMap<u32, GenreStat> = HashMap::new();


    for item in entries.iter() {

        if item.status == Some(4) { // 4 == dropped
            dropped.push(item.anime_id);
        }

        if item.status == Some(2) { // 2 == completed
            watched.push(item.anime_id);

            for genre in item.genres.iter().flatten() {
                genre_hashmap
                .entry(genre.id)
                .and_modify(|stat| stat.count += 1)
                .or_insert(
                    GenreStat {
                        name: genre.name.clone(),
                        count: 1,
                    }
                );
                // look for genre.id, modify if exists, or insert if it doesnt
            }

            if let Some(diff) = item.anime_score_diff {

                if diff > 1.0 && diff.abs() < 99.0 {
                    personal_favorites.push(item);
                    let embedding = embeddings.get_embedding(item.anime_id)?;
                    if let Some(embedding_vec) = embedding {
                         positive_pairs.push((embedding_vec, diff));
                    }
                    
                    for genre in item.genres.iter().flatten() {
                        genre_hashmap_favorites
                        .entry(genre.id)
                        .and_modify(|stat| stat.count += 1)
                        .or_insert(
                            GenreStat {
                                name: genre.name.clone(),
                                count: 1,
                            }
                        );
                        // look for genre.id, modify if exists, or insert if it doesnt
                    }

                }
                
                if diff < - 1.0 && diff.abs() < 99.0{
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
 
        
    // faster lookups with .contains by using a hashset, vecs would be o(n)
    let watched: HashSet<_> = watched.into_iter().collect();
    let dropped: HashSet<_> = dropped.into_iter().collect();
    results.retain(|item| !watched.contains(&item.mal_id) && !dropped.contains(&item.mal_id));
    results.truncate(50);
    // Can return here
  
    let mut genres_vec: Vec<(&u32, &GenreStat)> = genre_hashmap.iter().collect();
    genres_vec.sort_by_key(|(_, stat)| std::cmp::Reverse(stat.count));

    let mut genres_vec_favorites: Vec<(&u32, &GenreStat)> = genre_hashmap_favorites.iter().collect();
    genres_vec_favorites.sort_by_key(|(_, stat)| std::cmp::Reverse(stat.count));

    println!("\nWatched Genres:");
    for item in &genres_vec {
        println!("{} - appears: {}", item.1.name, item.1.count)
    }

    println!("\nPrefered Genres:");
    for item in &genres_vec_favorites {
        println!("{} - appears: {}", item.1.name, item.1.count)
    }
  
    println!("\nYou liked more than most people:");
    for e in personal_favorites.iter() {
        println!("{} ({}) - Score diff: {:?}", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id, e.anime_score_diff.unwrap());
        // println!("genres: {:?}", e.genres);
    }

    println!("\nPeople like it, but you didn't:");
    for e in unliked.iter() {
         println!("{} ({}) - Score diff: {:?}", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id, e.anime_score_diff.unwrap());
    }

    println!("\nRecommendations for you:");

    for item in &results {
        let title = &item.title;
        println!("{title}")
    }
    // for e in &results.iter(){
    //     println!("{e.title}");
    // }
    Ok(results)


}