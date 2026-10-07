use anyhow::Result;

pub const DEFAULT_CONTEXT_LENGTH: usize = 4096;
pub const DEFAULT_MAX_TOKENS: usize = 2048;
pub const MAX_TOKENS: usize = 4096;

pub fn generation_budget(
    prompt_tokens: usize,
    max_tokens: Option<usize>,
    context_length: Option<usize>,
    model_limit: Option<usize>,
) -> Result<usize> {
    let requested = max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let context = context_length.unwrap_or(DEFAULT_CONTEXT_LENGTH);
    anyhow::ensure!(
        (1..=MAX_TOKENS).contains(&requested),
        "生成トークン数は1〜4096で指定してください"
    );
    anyhow::ensure!(context > 0, "コンテキスト長は1以上で指定してください");
    let limit = model_limit.map_or(context, |model| context.min(model));
    anyhow::ensure!(prompt_tokens < limit,
        "入力は{prompt_tokens}トークンあり、コンテキスト上限{limit}を使い切っています。入力を短くするかコンテキスト長を増やしてください");
    Ok(requested.min(limit - prompt_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_generation_and_context_budgets() {
        assert_eq!(
            generation_budget(600, None, None, Some(32768)).unwrap(),
            2048
        );
        assert_eq!(
            generation_budget(100, Some(4096), Some(4096), None).unwrap(),
            3996
        );
        assert_eq!(
            generation_budget(1800, Some(2048), Some(2048), None).unwrap(),
            248
        );
        assert_eq!(
            generation_budget(2000, Some(2048), Some(4096), Some(2048)).unwrap(),
            48
        );
    }

    #[test]
    fn exhausted_or_invalid_limits_are_rejected() {
        assert!(generation_budget(4096, Some(2048), Some(4096), None).is_err());
        assert!(generation_budget(1024, None, None, Some(1024)).is_err());
        assert!(generation_budget(1, Some(0), None, None).is_err());
        assert!(generation_budget(1, Some(4097), None, None).is_err());
        assert!(generation_budget(1, None, Some(0), None).is_err());
    }

    #[test]
    fn context_can_exceed_4096_and_respects_model_capacity() {
        assert_eq!(
            generation_budget(12000, Some(2048), Some(32768), Some(32768)).unwrap(),
            2048
        );
        assert_eq!(
            generation_budget(32000, Some(2048), Some(131072), Some(32768)).unwrap(),
            768
        );
        assert!(generation_budget(32768, None, Some(131072), Some(32768)).is_err());
        assert_eq!(
            generation_budget(100000, None, Some(131072), None).unwrap(),
            2048
        );
    }
}
