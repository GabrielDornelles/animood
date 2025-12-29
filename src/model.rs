use anyhow::{Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::jina_bert::{
    BertModel, Config, PositionEmbeddingType
};
use std::path::Path;

pub fn build_model_and_tokenizer_from_disk(
    model_dir: impl AsRef<Path>,
) -> Result<(BertModel, tokenizers::Tokenizer)> {

    let model_dir = model_dir.as_ref();

    let model_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");

    // ---- sanity checks (fail early, fail loud) ----
    if !model_path.exists() {
        anyhow::bail!("Missing model.safetensors at {:?}", model_path);
    }
    if !tokenizer_path.exists() {
        anyhow::bail!("Missing tokenizer.json at {:?}", tokenizer_path);
    }

    println!("Loading model from: {:?}", model_path);
    println!("Loading tokenizer from: {:?}", tokenizer_path);

    // ---- tokenizer ----
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
    .map_err(|e| anyhow::anyhow!(
        "Failed to load tokenizer from {:?}: {}",
        tokenizer_path,
        e
    ))?;


    // ---- device ----
    let device = Device::Cpu;

    // ---- model config (hardcoded to match jina small v2) ----
    let config = Config::new(
        tokenizer.get_vocab_size(true),
        512,      // hidden size
        4,        // layers
        8,        // heads
        2048,     // intermediate size
        candle_nn::Activation::Gelu,
        1024,     // max seq len
        2,
        0.02,
        1e-12,
        0,
        PositionEmbeddingType::Alibi,
    );

    // ---- mmap weights (zero-copy, low RAM) ----
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_path],
            DType::F32,
            &device,
        )?
    };

    let model = BertModel::new(vb, &config)?;

    Ok((model, tokenizer))
}



// --------------------

// use anyhow::{Result};
// use candle_core::{DType, Device};
// use candle_nn::VarBuilder;
// use candle_transformers::models::jina_bert::{BertModel, Config, PositionEmbeddingType};

pub fn build_model_and_tokenizer() -> Result<(BertModel, tokenizers::Tokenizer)> {
    use hf_hub::{api::sync::Api, Repo, RepoType};

    let model_name = "jinaai/jina-embeddings-v2-small-en".to_string();        
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_name.clone(), RepoType::Model));
    let model_path = repo.get("model.safetensors")?;
    println!("Model path: {:?}", model_path);
    
    let tokenizer_path = repo.get("tokenizer.json")?;
    
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!(e))?;

    // let device: Device = Device::new_metal(0)?;
    let device = Device::Cpu;

    let config = Config::new(
        tokenizer.get_vocab_size(true),
        512,      // hidden size
        4,        // layers (actual for small)
        8,        // heads
        2048,     // intermediate size
        candle_nn::Activation::Gelu,
        1024,     // max seq len (model_max_length in config)
        2,
        0.02,
        1e-12,
        0,
        PositionEmbeddingType::Alibi,
    );



    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)?
    };

    let model = BertModel::new(vb, &config)?;

    Ok((model, tokenizer))
}
