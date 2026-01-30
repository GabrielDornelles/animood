use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::mal_functions::get_anime_list;
use crate::mal_types::MalAnimeEntry;
use crate::types::AnimeEmbeddings;
use crate::types::{AnimeResult};
use crate::embedder::embed;
use crate::AppState;
use crate::vec_ops::{search_similarity, norm, log_norm, build_taste_query, weighted_centroid, PreferenceSignal};

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

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.truncate(20);

    Ok(results)
}

struct GenreStat {
    name: String,
    count: usize,
}

struct GenreRatio {
    id: u32,
    name: String,
    ratio: f32
}


fn genre_reason_map() -> HashMap<u32, &'static str> {
    HashMap::from([
        (1,  "you enjoy **Action** stories with high energy, intense conflicts, and momentum that never slows down"),
        (8,  "you gravitate toward **Drama**, emotionally driven stories that explore people, relationships, and inner struggles"),
        (22, "you like **Romance** stories where emotions, bonds, and intimacy take center stage"),
        (10, "you enjoy **Fantasy** with immersive worlds, imaginative settings, and stories beyond everyday reality"),
        (37, "you're drawn to **Supernatural** stories that blur the line between the real and the unseen"),
        (2,  "you enjoy **Adventure** journeys, exploration, and characters growing through challenges"),
        (4,  "you appreciate **Comedy** and humor as part of storytelling, whether lighthearted or clever"),
        (7,  "you like **Mystery** narratives that keep you guessing and reward attention to detail"),
        (46, "you tend to enjoy **Award Winning** works with strong artistic or narrative ambition"),
        (41, "you enjoy **Suspense** stories that are tension-driven and keep you on edge"),
        (24, "you're interested in **Sci-Fi** with speculative ideas, futuristic themes, and thought-provoking concepts"),
        (9,  "you don't shy away from **Ecchi** with provocative or playful elements mixed into the story"),
        (30, "you appreciate **Sports** stories centered around competition, discipline, and personal growth"),
        (14, "you enjoy **Horror** with darker atmospheres designed to unsettle or disturb"),
        (5,  "you're open to **Avant Garde** storytelling that's experimental and unconventional"),
        (47, "you enjoy **Gourmet** stories that are cozy and celebrate food, craft, and everyday pleasures"),
        (36, "you appreciate **Slice of Life** stories that are quiet, grounded, and focus on daily life"),
    ])
}

// Rule of thumb according to llms:
//      Data ingestion / parsing → own
//      Derived views / analysis → borrow
//      Algorithms → take iterators / slices

// A snapshot of all user-specific derived data, tied to the lifetime of the MAL entries and embeddings
struct MalUserData <'a> {
    global_genre_hashmap: HashMap<u32, GenreStat>,
    favorite_genre_hashmap: HashMap<u32, GenreStat>,

    higher_than_avg_scored_anime: Vec<PreferenceSignal<'a>>,
    lower_than_avg_scored_anime: Vec<PreferenceSignal<'a>>,

    watched: HashSet<u32>,
    dropped: HashSet<u32>,

    favorites: Vec<&'a MalAnimeEntry>,
    unpreferred: Vec<&'a MalAnimeEntry>,
}

fn gather_mal_user_data<'a>(
    entries: &'a [MalAnimeEntry],
    embeddings: &'a AnimeEmbeddings,
) -> Result<MalUserData<'a>> {
    let mut personal_favorites = Vec::new();
    let mut unliked = Vec::new();

    let mut watched = Vec::new();
    let mut dropped = Vec::new();

    let mut positive_anime: Vec<PreferenceSignal> = Vec::new();
    let mut negative_anime: Vec<PreferenceSignal> = Vec::new();

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
                        let genre_ids: Vec<u32> = item.genres.iter().flatten().map(|genre| genre.id).collect();
                        positive_anime.push(
                                PreferenceSignal {
                                embedding: embedding_vec,
                                diff: diff,
                                genres: genre_ids
                            }
                        )
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
                        let genre_ids: Vec<u32> = item.genres.iter().flatten().map(|genre| genre.id).collect();
                         negative_anime.push(
                                PreferenceSignal {
                                embedding: embedding_vec,
                                diff: diff,
                                genres: genre_ids
                            }
                        )
                    }
                }
            }
        }
    }
    let watched: HashSet<_> = watched.into_iter().collect();
    let dropped: HashSet<_> = dropped.into_iter().collect();
    Ok(
        MalUserData { 
            global_genre_hashmap: genre_hashmap, 
            favorite_genre_hashmap: genre_hashmap_favorites, 
            higher_than_avg_scored_anime: positive_anime, 
            lower_than_avg_scored_anime: negative_anime, 
            watched: watched, 
            dropped: dropped, 
            favorites: personal_favorites, 
            unpreferred: unliked 
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
    //state: &AppState,
    embeddings: AnimeEmbeddings,
    username: &str,
) -> Result<Vec<AnimeResult>> {
    //let embeddings = &state.embeddings;//AnimeEmbeddings::load_bin("embeddings.bin")?;
    let entries = get_anime_list(username).await?;
    let user_data = gather_mal_user_data(&entries, &embeddings)?;

    let positive_taste = weighted_centroid(&user_data.higher_than_avg_scored_anime);
    let negative_taste = weighted_centroid(&user_data.lower_than_avg_scored_anime);

    let taste_query = build_taste_query(positive_taste, negative_taste).unwrap();

    let top = search_similarity(
        &taste_query,
        &embeddings.embeddings,
        100 * 2,
    );
    let results = build_ranked_results(
        top, &embeddings, &user_data, 20
    )?;
    
    let mut genres_ratio = Vec::new();

    for (genre_id, stat) in &user_data.favorite_genre_hashmap {
        if let Some(global_stat) = user_data.global_genre_hashmap.get(genre_id) {
            genres_ratio.push(
                GenreRatio {
                    id: *genre_id,
                    name: stat.name.clone(),
                    ratio: (stat.count as f32 / global_stat.count as f32) * (1.0 + stat.count as f32).ln()
                }
            ); 
        }
    }
    genres_ratio.sort_by(|a, b| {
        b.ratio.partial_cmp(&a.ratio).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut genres_vec: Vec<(&u32, &GenreStat)> = user_data.global_genre_hashmap.iter().collect();
    genres_vec.sort_by_key(|(_, stat)| std::cmp::Reverse(stat.count));

    let mut genres_vec_favorites: Vec<(&u32, &GenreStat)> = user_data.favorite_genre_hashmap.iter().collect();
    genres_vec_favorites.sort_by_key(|(_, stat)| std::cmp::Reverse(stat.count));

    println!("\nWatched Genres:");
    for item in &genres_vec {
        println!("{} - appears: {}", item.1.name, item.1.count)
    }

    println!("\nPrefered Genres:");
    for item in &genres_vec_favorites {
        println!("{} - appears: {}", item.1.name, item.1.count)
    }

    println!("\nPrefered Genres Ratio:");
    for item in &genres_ratio {
        println!("{} - appears: {}", item.name, item.ratio)
    }

    let top_5_genres_ratio = &genres_ratio[..genres_ratio.len().min(5)];


    let genre_reasons = genre_reason_map();
    for top_genre in top_5_genres_ratio {
       
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
        results.truncate(20);

        // println!("\nRecommendations for you for {}:", top_genre.name);
        let reason = genre_reasons
            .get(&top_genre.id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Recommendations for: {}", top_genre.name));

        
        //.unwrap_or_default(top_genre.name);
        println!("\n{}", reason);

        for item in &results {
            let title = &item.title;
            println!("{title}")
        }

    }


  
    println!("\nYou liked more than most people:");
    for e in user_data.favorites.iter() {
        println!("{} ({}) - Score diff: {:?}", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id, e.anime_score_diff.unwrap());
        // println!("genres: {:?}", e.genres);
    }

    println!("\nPeople like it, but you didn't:");
    for e in user_data.unpreferred.iter() {
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