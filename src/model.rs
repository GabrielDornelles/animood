
use anyhow::{Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::jina_bert::{BertModel, Config, PositionEmbeddingType};

pub fn build_model_and_tokenizer() -> Result<(BertModel, tokenizers::Tokenizer)> {
    use hf_hub::{api::sync::Api, Repo, RepoType};

    let model_name = "jinaai/jina-embeddings-v2-small-en".to_string();        
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_name.clone(), RepoType::Model));
    let model_path = repo.get("model.safetensors")?;
    
    let tokenizer_path = repo.get("tokenizer.json")?;
    
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!(e))?;

    let device: Device = Device::new_metal(0)?;

    // base model
    // let config = Config::new(
    //     tokenizer.get_vocab_size(true),
    //     768,      
    //     12,       
    //     12,       
    //     3072,    
    //     candle_nn::Activation::Gelu,
    //     8192,    
    //     2,
    //     0.02,
    //     1e-12,
    //     0,
    //     PositionEmbeddingType::Alibi,
    // );
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
