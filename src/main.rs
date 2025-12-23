use std::sync::Arc;

use anyhow::Result;
use axum::{Router, routing::post};

use anime_recommender::{
    AppState,
    api::query_handler,
    model::build_model_and_tokenizer,
    types::AnimeEmbeddings,
};


use tower_http::cors::{CorsLayer, Any};



#[tokio::main]
async fn main() -> Result<()> {
    println!("Loading model and tokenizer...");
    let (model, tokenizer) = build_model_and_tokenizer()?;

    println!("Loading embeddings.bin...");
    let embeddings = AnimeEmbeddings::load_bin("embeddings.bin")?;

    let state = Arc::new(AppState {
        model,
        tokenizer,
        embeddings,
    });

    let cors = CorsLayer::new()
        .allow_origin(Any) // ⚠️ for development; in production, specify exact origin
        .allow_methods(Any)
        .allow_headers(Any);


    let app = Router::new()
        .route("/query", post(query_handler))
        .with_state(state)
        .layer(cors);

    println!("Server running at http://0.0.0.0:3005");

    // axum::Server::bind(&"0.0.0.0:3000".parse()?)
    //     .serve(app.into_make_service())
    //     .await?;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("0.0.0.0:3005").await?;
    axum::serve(listener, app).await?;


    Ok(())
}


// -------------------


// use anyhow::{Result};
// use anime_recommender::{
//     build_bin_struct_from_json, 
//     //query_anime,
// };
// fn main() -> Result<()> { 
//     build_bin_struct_from_json("./llm_enriched.json")?;
//     //let query = "an adventure with elfs and dwarfs after beating the demon king";
//     //let query = "dark fantasy about surviving a brutal world battling demons of the past";
//     //let query = "a protagonist is transported into a new world. exploring a fantasy world slowly and thoughtfully";

//     // println!("Query: {query}");
//     // let recommendations = query_anime(query, 20)?;
//     // for (idx, item) in recommendations.iter().enumerate() {
//     //     println!("Top {idx} \n{item}");
//     // }
//     Ok(())
// }
