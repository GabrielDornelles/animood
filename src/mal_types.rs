// Rust structs to deserialize MyAnimeList `/load.json` entries.
// Designed for serde_json deserialization. Fields chosen to be
// conservative (Option<...>) because MAL sometimes returns null/missing.

use serde::{Deserialize, Deserializer};
use serde_json::Value;

fn deserialize_string_or_number<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;

    Ok(match value {
        Some(Value::String(s)) => Some(s),
        Some(Value::Number(n)) => Some(n.to_string()),
        Some(Value::Bool(b)) => Some(b.to_string()),
        _ => None,
    })
}


#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Genre {
    pub id: u32,
    pub name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Demographic {
    pub id: u32,
    pub name: String,
}

/// A single entry from the MAL `load.json` anime list endpoint.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MalAnimeEntry {
    // user-level fields
    pub status: Option<u8>,                 // user's list status (1=watching, 2=completed, ...)
    pub score: Option<u8>,                  // user's score for this anime (0-10)
    pub tags: Option<String>,
    // pub is_rewatching: Option<u8>,
    // pub num_watched_episodes: Option<u32>,
    // pub created_at: Option<u64>,            // unix timestamp
    // pub updated_at: Option<u64>,            // unix timestamp

    // anime metadata
    #[serde(deserialize_with = "deserialize_string_or_number")] // turns out there is anime named 86, and MAL returns it as an integer :D
    pub anime_title: Option<String>,
    // pub anime_title_eng: Option<String>,
    // pub anime_num_episodes: Option<u32>,
    // pub anime_airing_status: Option<u8>,
    pub anime_id: u32,
    // pub anime_studios: Option<String>,
    // pub anime_licensors: Option<String>,
    // pub anime_season: Option<String>,

    // popularity / scoring
    // pub anime_total_members: Option<u64>,
    // pub anime_total_scores: Option<u64>,
    // pub anime_score_val: Option<f32>,      // global MAL score (e.g. 8.21)
    pub anime_score_diff: Option<f32>,     // the per-user score diff field 
    // pub anime_popularity: Option<u32>,

    // video / flags
    // pub has_episode_video: Option<bool>,
    // pub has_promotion_video: Option<bool>,
    // pub has_video: Option<bool>,

    // pub video_url: Option<String>,

    // // categories
    pub genres: Option<Vec<Genre>>,
    // pub demographics: Option<Vec<Demographic>>,

    // // urls, images
    // pub title_localized: Option<String>,
    // pub anime_url: Option<String>,
    // pub anime_image_path: Option<String>,

    // // list UI / misc
    // pub is_added_to_list: Option<bool>,
    // pub anime_media_type_string: Option<String>,
    // pub anime_mpaa_rating_string: Option<String>,

    // // date strings (sometimes null)
    // pub start_date_string: Option<String>,
    // pub finish_date_string: Option<String>,
    // pub anime_start_date_string: Option<String>,
    // pub anime_end_date_string: Option<String>,

    // other small fields
    // pub days_string: Option<String>,
    // pub storage_string: Option<String>,
    // pub priority_string: Option<String>,
    // pub notes: Option<String>,
    // pub editable_notes: Option<String>,
}

// Convenience parse function
pub fn parse_mal_list(json: &str) -> Result<Vec<MalAnimeEntry>, serde_json::Error> {
    serde_json::from_str(json)
}

// Usage example (not compiled into library):
// let entries: Vec<MalAnimeEntry> = parse_mal_list(&body)?;
// for e in entries.iter() { println!("{} ({})", e.anime_title.as_deref().unwrap_or("<nil>"), e.anime_id); }

// Notes / next steps:
// - If you prefer to convert 0/1 integer flags into bools, we can add custom
//   deserializers (e.g. `#[serde(deserialize_with = "int_bool")]`).
// - We kept many fields optional to avoid panics when MAL returns null or
//   when fields vary between endpoints. If you want stricter types, we can
//   make them non-optional and provide fallback defaults.
// - If you want mapping into your internal `Anime`/`AnimeMetadata` types,
//   I can add `impl From<MalAnimeEntry> for Anime` helpers that compute
//   affinity-relevant values (genre vector, completion fraction, taste signal, etc.).
