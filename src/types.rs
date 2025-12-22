use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::fmt;

#[derive(Debug, Deserialize)]
pub struct ImageFormat {
    // image_url: Option<String>,
    // small_image_url: Option<String>,
    pub large_image_url: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Images {
    // jpg: Option<ImageFormat>,
    pub webp: Option<ImageFormat>,
}


#[derive(Debug, Deserialize)]
pub struct AnimeData {
    pub synopsis: Option<String>,
    pub llm_description: Option<String>,
    pub images: Option<Images>,
    pub score: f32,
    pub members: u32,
    pub favorites: u32
}

// ---

pub struct AnimeFilteredData {
    pub title: String,
    pub rich_synopsis: String,
    pub llm_description: String,
    pub picture: String,
    pub score: f32,
    pub members: u32,
    pub favorites: u32
}


#[derive(Debug, Serialize, Deserialize)]
pub struct AnimeEmbeddings {
    pub names: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
    pub picture_urls: Vec<String>,
    pub scores: Vec<f32>,
    pub members: Vec<u32>,
    pub favorites: Vec<u32>,
    pub llm_description: Vec<String>
}

impl AnimeEmbeddings {
    
    pub fn save_bin(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::BufWriter;
        
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }
    
    pub fn load_bin(path: &str) -> Result<Self> {
        use std::fs::File;
        use std::io::BufReader;
        
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let embeddings: Self = bincode::deserialize_from(reader)?;
        Ok(embeddings)
    }
}

// for the frontend to consume
#[derive(Debug, Serialize, Deserialize)]
pub struct AnimeResult {
    pub title: String,
    pub score: f32,
    pub image_url: String,
    pub llm_description: String
}

impl fmt::Display for AnimeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Title: {}", self.title)?;
        writeln!(f, "Score: {:.3}", self.score)?;
        writeln!(f, "Image: {}", self.image_url)?;
        Ok(())
    }
}