use anyhow::{Result};
use reqwest::header::{ACCEPT, USER_AGENT};
use crate::mal_types::{MalAnimeEntry, parse_mal_list};

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
