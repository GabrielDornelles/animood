use anyhow::{Result};
use candle_core::{Module, Tensor};
use candle_transformers::models::jina_bert::BertModel;
use tokenizers::Tokenizer;
use std::{time::Instant};
use crate::model::build_model_and_tokenizer;

fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

pub fn embed(
    model: &BertModel,
    tokenizer: &Tokenizer,
    synopsis: &[String],
) -> Result<Vec<Vec<f32>>> {
    let device = &model.device;

    // let tokenizer = tokenizer
    //     .with_padding(None)
    //     .with_truncation(None)
    //     .map_err(|e| anyhow::anyhow!(e))?;

    let mut embedding_vec = Vec::with_capacity(synopsis.len());

    for item in synopsis {
        if item.trim().is_empty() {
            embedding_vec.push(Vec::new());
            continue;
        }

        let tokens = tokenizer
            .encode(item.as_str(), true)
            .map_err(|e| anyhow::anyhow!(e))?

            .get_ids()
            .to_vec();

        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;

        let embeddings = model.forward(&token_ids)?;
        let (_, n_tokens, _) = embeddings.dims3()?;

        let embeddings = embeddings.sum(1)? / (n_tokens as f64);
        let embeddings = normalize_l2(&embeddings?)?;

        embedding_vec.push(embeddings.squeeze(0)?.to_vec1()?);
    }

    Ok(embedding_vec)
}


pub fn embed_at_build(synopsis: Vec<String>) -> Result<Vec<Vec<f32>>> {
    let (model, mut tokenizer) = build_model_and_tokenizer()?;
    let device = &model.device;

    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!(e))?;

    let mut embedding_vec = Vec::new();
    let array_size = &synopsis.len(); 
    for (idx, item) in synopsis.iter().enumerate() {
        if item.trim().is_empty() {
            // Push empty vector for consistency
            println!("empty synopsis found");
            embedding_vec.push(Vec::new());
            continue;
        }
        println!("{idx}/{array_size}");
    
        let tokens = tokenizer
        .encode(item.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?
        .get_ids()
        .to_vec();

        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;

        let t1 = Instant::now();
        let embeddings = model.forward(&token_ids)?;
        println!("Forward pass: {:?}", t1.elapsed());
        let (_, n_tokens, _) = embeddings.dims3()?;

        let embeddings = embeddings.sum(1)? / (n_tokens as f64);

        // unwrap Result<Tensor> into Tensor
        let embeddings = embeddings?;

        let embeddings = normalize_l2(&embeddings)?;
        let embedding_f32: Vec<f32> = embeddings.squeeze(0)?.to_vec1()?;
        embedding_vec.push(embedding_f32);
        //embedding_vec.push(embeddings);
        
    }
    Ok(embedding_vec)
    // Ok(())
}