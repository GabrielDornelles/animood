use anyhow::{Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use crate::types::AnimeData;
use crate::embedder::embed;
use crate::types::{AnimeEmbeddings, AnimeFilteredData};

// The JSON structure is: [{"Title": AnimeData}, {"Title2": AnimeData}, ...]
pub fn extract_synopses_and_pic_from_json(file_path: &str) -> Result<Vec<AnimeFilteredData>> {
    // Deserialize as Vec<HashMap<String, AnimeData>>
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let anime_entries: Vec<HashMap<String, AnimeData>> = serde_json::from_reader(reader)?;
    let results: Vec<AnimeFilteredData> = anime_entries
        .into_iter()
        .flat_map(|entry| {
            entry.into_iter()  // into_iter() gives (key, value) pairs
                .filter_map(|(title, anime)| {
                    anime.synopsis.and_then(|s| {
                        if !s.trim().is_empty() {

                            let picture = anime
                            .images?
                            .webp?
                            .large_image_url?;
                            
                            let mut rich_synopsis = s;

                            let description = anime.llm_description.clone()?;

                            if let Some(enriched) = anime.llm_description {
                                rich_synopsis.push_str("\n\n");
                                rich_synopsis.push_str(&enriched);
                            }
                            Some(
                                AnimeFilteredData{
                                    title: title,
                                    rich_synopsis: rich_synopsis,
                                    llm_description: description,
                                    picture: picture,
                                    score: anime.score,
                                    members: anime.members,
                                    favorites: anime.favorites
                                }
                            )
                            //Some((title, rich_synopsis, picture))
                        } else {
                            None
                        }
                    })
                })
        })
        .collect();
    Ok(results)
}


pub fn build_bin_struct_from_json(file_path: &str) -> Result<()> {
    let anime_data = extract_synopses_and_pic_from_json(file_path)?;
    
    let mut titles = Vec::new();
    let mut synopses = Vec::new();
    let mut picture_urls = Vec::new();

    let mut scores = Vec::new();
    let mut members = Vec::new();
    let mut favorites = Vec::new();

    let mut llm_descriptions = Vec::new();

    for item in anime_data {
        titles.push(item.title);
        synopses.push(item.rich_synopsis);
        picture_urls.push(item.picture);

        scores.push(item.score);
        members.push(item.members);
        favorites.push(item.favorites);

        llm_descriptions.push(item.llm_description);

    }
    
    let embeddings = embed(synopses)?;
    let anime_embeddings = AnimeEmbeddings{
        names: titles,
        embeddings: embeddings,
        picture_urls: picture_urls,
        scores: scores,
        members: members,
        favorites: favorites,
        llm_description: llm_descriptions
    };

    //Step 7: Save with bincode
    anime_embeddings.save_bin("embeddings.bin")?;
    Ok(())
}