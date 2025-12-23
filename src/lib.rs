pub mod model;
pub mod embedder;
pub mod load_dataset;
pub mod types;
pub mod search;

pub use embedder::embed;
pub use embedder::embed_at_build;
pub use load_dataset::build_bin_struct_from_json;
pub use search::query_anime;
pub use types::AnimeEmbeddings;

pub mod state; // 👈 ADD THIS
pub use state::AppState; // 👈 EXPORT
pub mod api; // 👈 ADD
