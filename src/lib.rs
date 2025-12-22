pub mod model;
pub mod embedder;
pub mod load_dataset;
pub mod types;
pub mod search;

pub use embedder::embed;
pub use load_dataset::build_bin_struct_from_json;
pub use search::query_anime;
pub use types::AnimeEmbeddings;


