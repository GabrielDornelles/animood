use anyhow::{Result};
use anime_recommender::{
    //build_bin_struct_from_json, 
    query_anime,
};

fn main() -> Result<()> { 
    //build_bin_struct_from_json("./anime_filtered.json")?;
    let query = "an adventure with elfs and dwarfs after beating the demon king";
    //let query = "dark fantasy about surviving a brutal world battling demons of the past";
    //let query = "a protagonist is transported into a new world. exploring a fantasy world slowly and thoughtfully";

    println!("Query: {query}");
    let recommendations = query_anime(query, 20)?;
    for (idx, item) in recommendations.iter().enumerate() {
        println!("Top {idx} \n{item}");
    }
    Ok(())
}
