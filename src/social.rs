use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::Serialize;
use anyhow::Result;

use crate::vec_ops::{weighted_centroid, build_taste_query, dot};
use crate::mal_functions::{gather_mal_user_data, MalUserData};
use crate::mal_functions::get_anime_list;
use crate::types::AnimeEmbeddings;

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct CachedProfile {
    pub taste_vector: Vec<f32>,
    pub top_genres: Vec<(u32, String)>,
    pub archetypes: Vec<ScoredArchetype>,   // ranked, primary first
    pub badges: Vec<Badge>,
    pub fetched_at: u64,
}

pub struct ProfileCache {
    inner: RwLock<HashMap<String, CachedProfile>>,
}

impl ProfileCache {
    pub fn new() -> Self {
        Self { inner: RwLock::new(HashMap::new()) }
    }

    pub fn get(&self, username: &str) -> Option<CachedProfile> {
        self.inner.read().unwrap().get(username).cloned()
    }

    pub fn set(&self, username: String, profile: CachedProfile) {
        self.inner.write().unwrap().insert(username, profile);
    }

    pub fn is_fresh(profile: &CachedProfile) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(profile.fetched_at) < 86_400
    }
}

// ---------------------------------------------------------------------------
// Archetypes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum Archetype {
    CerebralMelancholic,
    CozyEnjoyer,
    EdgeLord,
    HypeChaser,
    RomanceBrain,
    WorldBuilder,
    NarrativeJunkie,
    CultClassicHead,
    SportsManiac,
    HorrorHead,
    SciFiDreamer,
    SlapstickLover,
    MoeSoftie,
    ActionJunkie,
    CasualDrifter,
}

impl Archetype {
    pub fn label(&self) -> &'static str {
        match self {
            Self::CerebralMelancholic => "The Cerebral Melancholic",
            Self::CozyEnjoyer        => "The Cozy Enjoyer",
            Self::EdgeLord           => "The Edge Lord",
            Self::HypeChaser         => "The Hype Chaser",
            Self::RomanceBrain       => "The Romance Brain",
            Self::WorldBuilder       => "The World Builder",
            Self::NarrativeJunkie    => "The Narrative Junkie",
            Self::CultClassicHead    => "The Cult Classic Head",
            Self::SportsManiac       => "The Sports Maniac",
            Self::HorrorHead         => "The Horror Head",
            Self::SciFiDreamer       => "The Sci-Fi Dreamer",
            Self::SlapstickLover     => "The Slapstick Lover",
            Self::MoeSoftie          => "The Moe Softie",
            Self::ActionJunkie       => "The Action Junkie",
            Self::CasualDrifter      => "The Casual Drifter",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::CerebralMelancholic =>
                "You seek anime that lingers — slow burns, philosophical weight, and emotional complexity. You rate things other people find 'boring' as masterpieces.",
            Self::CozyEnjoyer =>
                "Slice of life, wholesome bonds, food anime at 11pm. You watch anime to feel something warm. Peak taste, no notes.",
            Self::EdgeLord =>
                "Psychological horror, moral ambiguity, characters who do terrible things for understandable reasons. You respect craft over comfort.",
            Self::HypeChaser =>
                "You're in it for the rush. Big battles, hype moments, sakuga cuts. You know a banger OP when you hear one.",
            Self::RomanceBrain =>
                "Will-they-won't-they has you in a chokehold. You've yelled at a screen more than once. You are normal about this.",
            Self::WorldBuilder =>
                "Magic systems, lore dumps, fictional maps — you eat it all. The isekai pipeline was made for you, but you have standards.",
            Self::NarrativeJunkie =>
                "Plot twists, unreliable narrators, mysteries that actually pay off. You rewatch episodes to catch what you missed the first time.",
            Self::CultClassicHead =>
                "Experimental storytelling, award-winning prestige anime, things that weren't made for the algorithm. Your watchlist is a museum.",
            Self::SportsManiac =>
                "The underdog arc. The training montage. The rival who becomes a friend. You cry at volleyball and you're proud of it.",
            Self::HorrorHead =>
                "You actively seek out the ones that disturb. Gore, dread, existential terror — it's all fair game. Sleep is for the boring.",
            Self::SciFiDreamer =>
                "Hard sci-fi concepts, speculative futures, AIs with feelings. You're here for the ideas as much as the characters.",
            Self::SlapstickLover =>
                "Timing, absurdity, chaos. You can appreciate a well-placed comedic beat more than most people appreciate drama.",
            Self::MoeSoftie =>
                "Cute things done cutely. Character dynamics and soft aesthetics matter to you. No shame in the comfort watch.",
            Self::ActionJunkie =>
                "Non-stop momentum, kinetic animation, big power systems. You want to feel the impact through the screen.",
            Self::CasualDrifter =>
                "Eclectic taste, no fixed genre loyalties. You'll watch anything once and have surprisingly strong opinions about all of it.",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            Self::CerebralMelancholic => "🌧️",
            Self::CozyEnjoyer        => "☕",
            Self::EdgeLord           => "🩸",
            Self::HypeChaser         => "⚡",
            Self::RomanceBrain       => "💘",
            Self::WorldBuilder       => "🗺️",
            Self::NarrativeJunkie    => "🔍",
            Self::CultClassicHead    => "🎭",
            Self::SportsManiac       => "🏆",
            Self::HorrorHead         => "💀",
            Self::SciFiDreamer       => "🚀",
            Self::SlapstickLover     => "😂",
            Self::MoeSoftie          => "🌸",
            Self::ActionJunkie       => "🔥",
            Self::CasualDrifter      => "🎲",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoredArchetype {
    pub label: String,
    pub description: String,
    pub emoji: String,
    #[serde(skip)]
    pub signal: f32,
}

impl ScoredArchetype {
    fn from(archetype: Archetype, signal: f32) -> Self {
        Self {
            label: archetype.label().to_string(),
            description: archetype.description().to_string(),
            emoji: archetype.emoji().to_string(),
            signal,
        }
    }
}

/// Score each archetype using a weighted blend of fav_ratio and global_presence,
/// then return the top 3 that clear the minimum threshold.
///
/// MAL genre IDs used:
/// 1=Action, 2=Adventure, 4=Comedy, 5=Avant Garde, 7=Mystery, 8=Drama,
/// 9=Ecchi, 10=Fantasy, 14=Horror, 22=Romance, 24=Sci-Fi,
/// 30=Sports, 36=Slice of Life, 37=Supernatural,
/// 41=Suspense, 46=Award Winning, 47=Gourmet
pub fn rank_archetypes(user_data: &MalUserData) -> Vec<ScoredArchetype> {
    let fav_map: HashMap<u32, usize> = user_data
        .favorite_genres_sorted
        .iter()
        .map(|(id, stat)| (*id, stat.count))
        .collect();

    let global_map: HashMap<u32, usize> = user_data
        .global_genres_sorted
        .iter()
        .map(|(id, stat)| (*id, stat.count))
        .collect();

    let fav_total   = fav_map.values().sum::<usize>().max(1) as f32;
    let global_total = global_map.values().sum::<usize>().max(1) as f32;

    let score = |ids: &[u32]| -> f32 {
        let fav: f32    = ids.iter().map(|id| *fav_map.get(id).unwrap_or(&0) as f32).sum();
        let global: f32 = ids.iter().map(|id| *global_map.get(id).unwrap_or(&0) as f32).sum();
        (fav / fav_total) * 0.7 + (global / global_total) * 0.3
    };

    let mut candidates = vec![
        ScoredArchetype::from(Archetype::CerebralMelancholic, score(&[8, 41, 46])),
        ScoredArchetype::from(Archetype::CozyEnjoyer,         score(&[36, 47])),
        ScoredArchetype::from(Archetype::EdgeLord,            score(&[14, 5, 41])),
        ScoredArchetype::from(Archetype::HypeChaser,          score(&[1, 2, 37])),
        ScoredArchetype::from(Archetype::RomanceBrain,        score(&[22, 26, 28])),
        ScoredArchetype::from(Archetype::WorldBuilder,        score(&[10, 37, 2])),
        ScoredArchetype::from(Archetype::NarrativeJunkie,     score(&[7, 24, 46])),
        ScoredArchetype::from(Archetype::CultClassicHead,     score(&[5, 46])),
        ScoredArchetype::from(Archetype::SportsManiac,        score(&[30])),
        ScoredArchetype::from(Archetype::HorrorHead,          score(&[14])),
        ScoredArchetype::from(Archetype::SciFiDreamer,        score(&[24])),
        ScoredArchetype::from(Archetype::SlapstickLover,      score(&[4])),
        ScoredArchetype::from(Archetype::MoeSoftie,           score(&[9, 36])),
        ScoredArchetype::from(Archetype::ActionJunkie,        score(&[1, 2])),
    ];

    candidates.sort_by(|a, b| b.signal.partial_cmp(&a.signal).unwrap());

    const MIN_SIGNAL: f32 = 0.05;
    let mut ranked: Vec<ScoredArchetype> = candidates
        .into_iter()
        .filter(|a| a.signal >= MIN_SIGNAL)
        .take(3)
        .collect();

    if ranked.is_empty() {
        ranked.push(ScoredArchetype::from(Archetype::CasualDrifter, 0.0));
    }

    ranked
}

// ---------------------------------------------------------------------------
// Behavioral badges
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Badge {
    pub id: String,
    pub label: String,
    pub description: String,
    pub emoji: String,
}

impl Badge {
    fn new(id: &str, label: &str, description: &str, emoji: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            description: description.to_string(),
            emoji: emoji.to_string(),
        }
    }
}

pub fn compute_badges(user_data: &MalUserData) -> Vec<Badge> {
    let mut badges = Vec::new();

    let watched_count = user_data.watched.len();
    let dropped_count = user_data.dropped.len();
    let total         = (watched_count + dropped_count).max(1) as f32;
    let drop_rate     = dropped_count as f32 / total;
    let favs          = &user_data.favorites;
    let unpreferred   = &user_data.unpreferred;

    // Completionist vs Quitter
    if drop_rate < 0.03 && watched_count >= 30 {
        badges.push(Badge::new(
            "completionist",
            "The Completionist",
            "Drop rate under 3%. You finish what you start, even when it gets bad.",
            "✅",
        ));
    } else if drop_rate > 0.20 {
        badges.push(Badge::new(
            "quitter",
            "The Quitter",
            "High drop rate. Life's too short, and episode 1 tells you everything.",
            "🚪",
        ));
    }

    // Contrarian — rates popular anime below average more than most
    let contrarian_ratio = unpreferred.len() as f32 / watched_count.max(1) as f32;
    if contrarian_ratio > 0.25 && unpreferred.len() >= 10 {
        badges.push(Badge::new(
            "contrarian",
            "The Contrarian",
            "You rate popular anime lower than MAL average more than most. You have opinions and they're not the safe ones.",
            "🔻",
        ));
    }

    // Hidden Gem Hunter — broad favorite genre spread + low contrarian signal
    let fav_genre_spread = user_data.favorite_genres_sorted.len();
    if favs.len() >= 10 && fav_genre_spread >= 8 && contrarian_ratio < 0.15 {
        badges.push(Badge::new(
            "hidden_gem_hunter",
            "The Hidden Gem Hunter",
            "Wide genre spread in your favorites and low contrarian signal — you dig into overlooked corners and find things worth loving.",
            "💎",
        ));
    }

    // Loyalist — top fav genre dominates at a high ratio
    if let (Some((_, top_fav)), Some((_, top_global))) = (
        user_data.favorite_genres_sorted.first(),
        user_data.global_genres_sorted.first(),
    ) {
        let loyalty = top_fav.count as f32 / top_global.count.max(1) as f32;
        if loyalty > 0.55 && top_fav.count >= 10 {
            badges.push(Badge::new(
                "loyalist",
                "The Loyalist",
                &format!(
                    "Over half your favorites are {}. You know what you like and you're not apologizing for it.",
                    top_fav.name
                ),
                "🎖️",
            ));
        }
    }

    // Veteran / Newcomer
    if watched_count >= 200 {
        badges.push(Badge::new(
            "veteran",
            "The Veteran",
            &format!(
                "{} completed anime. You've seen things. You have context. Newbies fear your recommendations.",
                watched_count
            ),
            "🏅",
        ));
    } else if watched_count >= 5 && watched_count <= 20 {
        badges.push(Badge::new(
            "newcomer",
            "The Newcomer",
            "Still early in the journey. The best discoveries are still ahead of you.",
            "🌱",
        ));
    }

    badges.truncate(3);
    badges
}

// ---------------------------------------------------------------------------
// Defender badges
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct DefenderBadge {
    pub anime_id: u32,
    pub anime_title: String,
    pub score_diff: f32,
    pub label: String,
    pub emoji: String,
    pub image_url: Option<String>,
}

pub fn compute_defender_badges(user_data: &MalUserData) -> Vec<DefenderBadge> {
    let mut favs: Vec<_> = user_data.favorites.iter().collect();
    favs.sort_by(|a, b| {
        b.anime_score_diff
            .unwrap_or(0.0)
            .partial_cmp(&a.anime_score_diff.unwrap_or(0.0))
            .unwrap()
    });

    favs.iter()
        .take(3)
        .filter_map(|entry| {
            let title = entry.anime_title.clone()?;
            let diff  = entry.anime_score_diff?;
            Some(DefenderBadge {
                anime_id:    entry.anime_id,
                anime_title: title.clone(),
                score_diff:  diff,
                label:       format!("{} Defender", title),
                emoji:       "🛡️".to_string(),
                image_url:   entry.anime_image_path.clone(),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Build or load profile from cache
// ---------------------------------------------------------------------------

async fn build_full_profile(
    username: &str,
    embeddings: &AnimeEmbeddings,
) -> Result<(CachedProfile, Vec<DefenderBadge>)> {
    let entries   = get_anime_list(username).await?;
    let user_data = gather_mal_user_data(&entries, embeddings)?;

    let positive = weighted_centroid(&user_data.higher_than_avg_scored_anime);
    let negative = weighted_centroid(&user_data.lower_than_avg_scored_anime);
    let taste_vector = build_taste_query(positive, negative)
        .ok_or_else(|| anyhow::anyhow!(
            "Not enough rated anime to build a taste profile for '{}'", username
        ))?;

    let top_genres: Vec<(u32, String)> = user_data
        .favorite_genres_sorted
        .iter()
        .take(5)
        .map(|(id, stat)| (*id, stat.name.clone()))
        .collect();

    let archetypes      = rank_archetypes(&user_data);
    let badges          = compute_badges(&user_data);
    let defender_badges = compute_defender_badges(&user_data);

    let fetched_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let profile = CachedProfile { taste_vector, top_genres, archetypes, badges, fetched_at };
    Ok((profile, defender_badges))
}

// ---------------------------------------------------------------------------
// Archetype endpoint
// ---------------------------------------------------------------------------

#[derive(Serialize, Debug)]
pub struct ArchetypeResponse {
    pub username: String,
    /// index 0 = primary, 1 = secondary, 2 = tertiary
    pub archetypes: Vec<ScoredArchetype>,
    pub badges: Vec<Badge>,
    pub defender_badges: Vec<DefenderBadge>,
    pub top_genres: Vec<String>,
    pub profile_age_secs: u64,
}

pub async fn compute_archetype(
    cache: &ProfileCache,
    embeddings: &AnimeEmbeddings,
    username: &str,
) -> Result<ArchetypeResponse> {
    let (profile, defender_badges) = build_full_profile(username, embeddings).await?;
    let top_genres = profile.top_genres.iter().map(|(_, n)| n.clone()).collect();

    cache.set(username.to_string(), profile.clone());

    Ok(ArchetypeResponse {
        username: username.to_string(),
        archetypes: profile.archetypes,
        badges: profile.badges,
        defender_badges,
        top_genres,
        profile_age_secs: 0,
    })
}

// ---------------------------------------------------------------------------
// Compatibility endpoint
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_ab = dot(a, b);
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    (dot_ab / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

fn shared_genres(a: &[(u32, String)], b: &[(u32, String)]) -> Vec<String> {
    let b_ids: std::collections::HashSet<u32> = b.iter().map(|(id, _)| *id).collect();
    a.iter().filter(|(id, _)| b_ids.contains(id)).map(|(_, n)| n.clone()).collect()
}

fn exclusive_genres(a: &[(u32, String)], b: &[(u32, String)]) -> Vec<String> {
    let b_ids: std::collections::HashSet<u32> = b.iter().map(|(id, _)| *id).collect();
    a.iter().filter(|(id, _)| !b_ids.contains(id)).map(|(_, n)| n.clone()).collect()
}

fn verdict(score: u8) -> &'static str {
    match score {
        90..=100 => "Soulmates. Probably already watch together.",
        75..=89  => "High compatibility. You clearly have taste.",
        60..=74  => "Solid overlap with interesting differences.",
        45..=59  => "Different flavors, but you can make it work.",
        30..=44  => "Polar opposites — could be fun or disastrous.",
        _        => "Completely different taste universes. Respect.",
    }
}

#[derive(Serialize, Debug)]
pub struct CompatibilityUserInfo {
    pub username: String,
    pub archetypes: Vec<ScoredArchetype>,
    pub badges: Vec<Badge>,
    pub defender_badges: Vec<DefenderBadge>,
    pub top_genres: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct CompatibilityResponse {
    pub compatibility_score: u8,
    pub verdict: String,
    pub shared_genres: Vec<String>,
    pub user_a_exclusive: Vec<String>,
    pub user_b_exclusive: Vec<String>,
    pub user_a: CompatibilityUserInfo,
    pub user_b: CompatibilityUserInfo,
    pub user_a_profile_age_secs: u64,
    pub user_b_profile_age_secs: u64,
}

pub async fn compute_compatibility(
    cache: &ProfileCache,
    embeddings: &AnimeEmbeddings,
    username_a: &str,
    username_b: &str,
) -> Result<CompatibilityResponse> {
    let ((profile_a, defenders_a), (profile_b, defenders_b)) = tokio::try_join!(
        build_full_profile(username_a, embeddings),
        build_full_profile(username_b, embeddings),
    )?;

    let raw   = cosine_similarity(&profile_a.taste_vector, &profile_b.taste_vector);
    let score = (((raw + 1.0) / 2.0) * 100.0).round() as u8;

    let shared = shared_genres(&profile_a.top_genres, &profile_b.top_genres);
    let a_excl = exclusive_genres(&profile_a.top_genres, &profile_b.top_genres);
    let b_excl = exclusive_genres(&profile_b.top_genres, &profile_a.top_genres);

    let age_a = profile_a.fetched_at;
    let age_b = profile_b.fetched_at;

    cache.set(username_a.to_string(), profile_a.clone());
    cache.set(username_b.to_string(), profile_b.clone());

    Ok(CompatibilityResponse {
        compatibility_score: score,
        verdict: verdict(score).to_string(),
        shared_genres: shared,
        user_a_exclusive: a_excl,
        user_b_exclusive: b_excl,
        user_a: CompatibilityUserInfo {
            username:        username_a.to_string(),
            archetypes:      profile_a.archetypes,
            badges:          profile_a.badges,
            defender_badges: defenders_a,
            top_genres:      profile_a.top_genres.iter().map(|(_, n)| n.clone()).collect(),
        },
        user_b: CompatibilityUserInfo {
            username:        username_b.to_string(),
            archetypes:      profile_b.archetypes,
            badges:          profile_b.badges,
            defender_badges: defenders_b,
            top_genres:      profile_b.top_genres.iter().map(|(_, n)| n.clone()).collect(),
        },
        user_a_profile_age_secs: age_a,
        user_b_profile_age_secs: age_b,
    })
}
