# Weighted Sentiment Implementation Report

## 1. Objective

The sentiment pipeline previously treated every collected X/Twitter post equally after FinBERT scoring. This meant a post from a highly credible source, such as an institutional crypto news account or official protocol account, had the same impact as an unknown user post.

This implementation adds a source credibility weighting layer while preserving the existing sentiment analyzer and keeping unknown accounts included in the calculation.

## 2. Design Summary

The update introduces a separate source credibility resolver instead of hardcoding weights inside ingestion or scoring logic. Each scored tweet/post now receives:

- `source_tier`
- `source_weight`
- `source_focus`
- `source_reason`
- `raw_sentiment_score`
- `weighted_sentiment_contribution`

The original raw sentiment score is still preserved. The 5-minute sentiment table keeps `avg_score` as the unweighted FinBERT average for backward compatibility and adds `weighted_avg_score` as the new credibility-weighted metric.

## 3. Source Weight Configuration

Source tiers are configured in:

```text
src/core/config/twitter_source_tiers.json
```

Default weights:

| Tier | Weight | Purpose |
| :--- | ---: | :--- |
| `tier1` | `3.0` | Institutional crypto news, official protocols, founders, major sector leaders |
| `economy_news_sources` | `2.75` | Major finance/economy news sources |
| `tier2` | `2.0` | Strong crypto analysis and news sources |
| `turkish_economy_sources` | `1.75` | Turkish economy and finance sources |
| `tier3` | `1.25` | Useful supporting sources with higher hype risk |
| `unknown` | `1.0` | Missing, ordinary, or non-listed accounts |

The maximum default source weight is capped at `3.0` to avoid over-dominance by any source class.

## 4. Credibility Resolver

Implemented in:

```text
src/feature_engineering/source_credibility.py
```

Main responsibilities:

- Normalizes handles case-insensitively.
- Accepts handles with or without `@`.
- Handles URL-style profile strings when possible.
- Looks up configured source tier metadata.
- Returns safe fallback metadata for missing or unknown handles.
- Provides shared weighted sentiment calculation helpers.

Unknown accounts are never filtered out. If the author handle is missing or not in the config, the resolver returns:

```json
{
  "source_tier": "unknown",
  "source_weight": 1.0,
  "source_focus": null,
  "source_reason": "Default unknown source"
}
```

## 5. Weighted Sentiment Formula

Raw average:

```text
avg_score = sum(raw_sentiment_score) / tweet_count
```

Weighted average:

```text
weighted_avg_score =
    sum(raw_sentiment_score * source_weight) / sum(source_weight)
```

Example from the validation script:

| Source | Raw Score | Weight | Weighted Contribution |
| :--- | ---: | ---: | ---: |
| `@CoinDesk` | `0.8` | `3.0` | `2.4` |
| `@VitalikButerin` | `0.6` | `3.0` | `1.8` |
| `@randomuser123` | `-0.5` | `1.0` | `-0.5` |

Results:

```text
unweighted_avg = (0.8 + 0.6 - 0.5) / 3 = 0.3000
weighted_avg = (2.4 + 1.8 - 0.5) / 7.0 = 0.5286
```

## 6. Pipeline Integration

### XQuik Live Sentiment Path

Updated file:

```text
src/data_sources/xquik/xquik_ingestion.py
```

After tweets are scored by FinBERT, the ingestion path:

1. Extracts the best available author handle from the event payload.
2. Resolves the source tier and source weight.
3. Adds credibility metadata to the scored tweet payload.
4. Sends `source_weight` and `source_tier` into the 5-minute sentiment aggregator.
5. Logs per-cycle tier counts for debugging.

The existing event polling, retweet filtering, and FinBERT scoring behavior were left intact.

### Aggregation Path

Updated file:

```text
src/feature_engineering/sentiment_aggregator.py
```

The accumulator now tracks:

- Raw score sum
- Weighted score sum
- Total source weight
- Tier counts
- Existing positive/negative/neutral counts
- Existing min/max/sample tweet fields

It writes both raw and weighted metrics to `tweet_sentiment_5m`.

### Tavily/Timescale Sink Compatibility Path

Updated file:

```text
src/feature_engineering/sentiment_scorer.py
```

The older `score_and_store` path now uses the same source credibility resolver and writes the same weighted columns. This keeps backward compatibility for records routed through `TimescaleSink`.

## 7. Database Changes

Updated file:

```text
src/db/db_schema.sql
```

New additive columns:

- `weighted_avg_score`
- `total_source_weight`
- `tier1_count`
- `tier2_count`
- `tier3_count`
- `economy_news_count`
- `turkish_economy_count`
- `unknown_count`

The migration uses `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so existing databases can be upgraded without dropping historical data.

Existing consumers that read `avg_score` continue to work because `avg_score` remains the unweighted average.

## 8. Validation

Validation script:

```text
scripts/validate_weighted_sentiment.py
```

Run:

```bash
python3 scripts/validate_weighted_sentiment.py
```

Observed result:

```text
unweighted_avg=0.3000
weighted_avg=0.5286
total_source_weight=7.0
OK
```

Additional checks performed:

```bash
python3 -m py_compile src/feature_engineering/source_credibility.py \
  src/feature_engineering/sentiment_aggregator.py \
  src/feature_engineering/sentiment_scorer.py \
  src/data_sources/xquik/xquik_ingestion.py \
  scripts/validate_weighted_sentiment.py
```

The tier JSON was also checked with:

```bash
python3 -m json.tool src/core/config/twitter_source_tiers.json
```

## 9. Assumptions and Safeguards

- Author handle field names may vary by provider, so extraction checks common direct and nested keys.
- Missing handles are treated as `unknown`, not as errors.
- Unknown accounts remain part of both raw and weighted sentiment.
- The sentiment model logic was not changed.
- `avg_score` was intentionally preserved for existing anomaly model queries.
- Per-author caps were not implemented because the current aggregator does not retain author-level event windows, and adding that would increase state complexity.

## 10. Capstone Explanation

This implementation improves sentiment quality by giving moderately higher influence to credible institutional or official sources while retaining broader community sentiment from ordinary users. The system now supports direct comparison between raw public sentiment (`avg_score`) and credibility-adjusted sentiment (`weighted_avg_score`), which is useful for debugging, model analysis, and presentation of design tradeoffs.
