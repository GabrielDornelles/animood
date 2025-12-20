use anyhow::Result;
use rayon::{prelude::*};
use crate::types::{AnimeEmbeddings, AnimeResult};
use crate::embedder::embed;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn search_similarity(
    query: &[f32],
    embeddings: &[Vec<f32>],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = embeddings
        .par_iter()
        .enumerate()
        .map(|(i, emb)| (i, dot(query, emb)))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(k);
    scores
}

pub fn query_anime(query: &str) -> Result<Vec<AnimeResult>> {
    let loaded = AnimeEmbeddings::load_bin("embeddings.bin")?;

    let query_emb = &embed(vec![query.to_string()])?[0];
    let top = search_similarity(&query_emb, &loaded.embeddings, 20);

    let results = top
        .into_iter()
        .map(|(idx, score)| AnimeResult {
            title: loaded.names[idx].clone(),
            score,
            image_url: loaded.picture_urls[idx].clone(),
        })
        .collect();
    Ok(results)
}

