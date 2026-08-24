use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Wrapper for Gemma tokenizer (SentencePiece-based)
pub struct GemmaTokenizer {
    inner: Tokenizer,
}

impl GemmaTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tok = Tokenizer::from_file(path.as_ref()).map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self { inner: tok })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(enc.get_ids().iter().map(|&x| x as i64).collect())
    }

    pub fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        let u32_ids: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        self.inner
            .decode(&u32_ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

pub fn load_tokenizer<P: AsRef<Path>>(path: P) -> Result<GemmaTokenizer> {
    GemmaTokenizer::from_file(path.as_ref())
        .with_context(|| format!("failed to load tokenizer at {:?}", path.as_ref()))
}

/// Gemma chat template helper (simple version)
/// Gemma instruct models expect: <bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
pub fn apply_gemma_chat_template(prompt: &str) -> String {
    format!(
        "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        prompt
    )
}

/// Fallback mock tokenizer when model not present (for validation without download)
#[allow(dead_code)]
pub fn mock_tokenize(text: &str) -> Vec<i64> {
    // Simple whitespace mock - not accurate but allows pipeline validation
    text.split_whitespace()
        .enumerate()
        .map(|(i, _)| 1000 + i as i64)
        .collect()
}

#[allow(dead_code)]
pub fn mock_detokenize(ids: &[i64]) -> String {
    format!("[mock detokenize: {} tokens]", ids.len())
}
