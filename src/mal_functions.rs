use anyhow::{Result};
use reqwest::header::{ACCEPT, USER_AGENT};
use crate::mal_types::{MalAnimeEntry, parse_mal_list};
use std::collections::{HashMap, HashSet};

use crate::vec_ops::{PreferenceSignal};
use crate::types::AnimeEmbeddings;

const PAGE_SIZE: usize = 300;

pub async fn get_anime_list(username: &str) -> Result<Vec<MalAnimeEntry>> {
    let client = reqwest::Client::new();
    let mut offset = 0;
    let mut all_entries = Vec::new();
    
    loop {
        let url = format!(
            "https://myanimelist.net/animelist/{}/load.json?status=7&offset={}",
            username,
            offset
        );

        let res = client
            .get(&url)
            .header(USER_AGENT, "Mozilla/5.0")
            .header(ACCEPT, "application/json")
            .send()
            .await?;

        let body = res.text().await?;

        let entries = parse_mal_list(&body)?;

        if entries.is_empty() {
            break; // 🚪 no more pages
        }

        all_entries.extend(entries);
        offset += PAGE_SIZE;
    }

    println!("Total anime fetched: {} for {}", all_entries.len(), username);
    Ok(all_entries)

}

pub struct GenreStat {
    pub name: String,
    pub count: usize,
}

pub struct GenreRatio {
    pub id: u32,
    pub name: String,
    pub ratio: f32
}

pub fn genre_reason_map() -> HashMap<u32, &'static str> {
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
pub struct MalUserData <'a> {
    pub global_genres_sorted: Vec<(u32, GenreStat)>,
    pub favorite_genres_sorted: Vec<(u32, GenreStat)>,
    pub top_5_genres_ratio: Vec<GenreRatio>,

    pub higher_than_avg_scored_anime: Vec<PreferenceSignal<'a>>,
    pub lower_than_avg_scored_anime: Vec<PreferenceSignal<'a>>,

    pub watched: HashSet<u32>,
    pub dropped: HashSet<u32>,

    pub favorites: Vec<&'a MalAnimeEntry>,
    pub unpreferred: Vec<&'a MalAnimeEntry>
}

pub fn gather_mal_user_data<'a>(
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

    let mut genres_ratio = Vec::new();

    for (genre_id, stat) in &genre_hashmap_favorites {
        if let Some(global_stat) = genre_hashmap.get(genre_id) {
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

    let mut genres_vec: Vec<(u32, GenreStat)> = genre_hashmap.into_iter().collect();
    genres_vec.sort_by_key(|(_, stat)| std::cmp::Reverse(stat.count));
    
    let mut genres_vec_favorites: Vec<(u32, GenreStat)> = genre_hashmap_favorites.into_iter().collect();
    genres_vec_favorites.sort_by_key(|(_, stat)| std::cmp::Reverse(stat.count));

    let top_5_genres_ratio: Vec<GenreRatio> = genres_ratio.into_iter().take(5).collect();

    Ok(
        MalUserData {
            global_genres_sorted: genres_vec,
            favorite_genres_sorted : genres_vec_favorites,
            higher_than_avg_scored_anime: positive_anime, 
            lower_than_avg_scored_anime: negative_anime, 
            watched: watched, 
            dropped: dropped, 
            favorites: personal_favorites, 
            unpreferred: unliked,
            top_5_genres_ratio: top_5_genres_ratio
        }
    )
}

