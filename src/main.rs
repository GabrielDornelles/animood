use anyhow::{Result};
use anime_recommender::{
    //build_bin_struct_from_json, 
    query_anime,
};

fn main() -> Result<()> { 
    //build_bin_struct_from_json("./anime_filtered.json")?;
    let query = "philosofical and existential anime";
    let recommendations = query_anime(query)?;
    for (idx, item) in recommendations.iter().enumerate() {
        println!("Top {idx} \n{item}");
    }
    Ok(())
}
