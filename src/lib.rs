pub mod model;
pub mod embedder;
pub mod load_dataset;
pub mod types;
pub mod mal_types;
pub mod search;

pub use embedder::embed;
pub use embedder::bin_embed_creation;
pub use load_dataset::build_bin_struct_from_json;
pub use search::query_anime;
pub use types::AnimeEmbeddings;

pub mod state;
pub use state::AppState;
pub mod api;
