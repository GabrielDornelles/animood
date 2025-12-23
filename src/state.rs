use candle_transformers::models::jina_bert::BertModel;
use tokenizers::Tokenizer;

use crate::types::AnimeEmbeddings;

/// Shared application state, loaded once at startup
pub struct AppState {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
    pub embeddings: AnimeEmbeddings,
}
