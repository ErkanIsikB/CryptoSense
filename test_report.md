# CryptoSense Automated Test Suite Execution Report

**Execution Status**: 🟢 PASS
**Date/Time**: 2026-06-12 00:49:14
**Total Duration**: 2m 40.3s
**Python Version**: 3.14.0

## Executive Summary

> **27/27** tests passed across **3** test suites

| Test Suite | Total Tests | Passed | Failed | Skipped | Duration |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Unit Tests | 24 | 24 | 0 | 0 | 3.31s |
| Database Integration | 2 | 2 | 0 | 0 | 3.42s |
| XQuik Live API | 1 | 1 | 0 | 0 | 2m 9.1s |
| **TOTAL** | **27** | **27** | **0** | **0** | **2m 40.3s** |

## 📋 Detailed Per-Test Results

### 📊 Data Processing
> **3/3** tests passed — total duration: **2ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_align_labels_to_sequences` | 2ms | Verifies that ground-truth labels are correctly aligned to windowed feature sequences. |
| ✅ | `test_slice_continuous_windows` | <1ms | Validates sliding-window slicing on continuous time-series data without gaps. |
| ✅ | `test_slice_continuous_windows_with_gaps` | <1ms | Ensures the window slicer correctly handles temporal gaps in the input data. |

### 🧠 LSTM Autoencoder
> **2/2** tests passed — total duration: **24ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_dimensions_and_forward_pass` | 18ms | Checks that the LSTM Autoencoder output tensor dimensions match the input. |
| ✅ | `test_parameters_gradients` | 6ms | Confirms that gradients flow through all trainable parameters during backprop. |

### 🔄 Retraining Scheduler
> **2/2** tests passed — total duration: **10ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_retrain_job` | 7ms | Verifies that the periodic retrain job invokes the training pipeline correctly. |
| ✅ | `test_start_and_shutdown_scheduler` | 3ms | Tests that the APScheduler-based retraining loop starts and shuts down cleanly. |

### 💬 Sentiment Scorer (FinBERT)
> **7/7** tests passed — total duration: **3.26s**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_backward_compat_alias` | <1ms | Verifies score_texts_batched is a backward-compatible alias for score_news_batched. |
| ✅ | `test_compound_score` | <1ms | Validates the compound score formula: score = P(positive) − P(negative). |
| ✅ | `test_cryptobert_fallback_if_unavailable` | 1.42s | Ensures CryptoBERT fallback returns neutral scores when the pipeline is unavailable. |
| ✅ | `test_cryptobert_model_f1_score` | 354ms | Validates CryptoBERT classification performance (Macro F1 ≥ 0.70) on a 18-sample crypto tweet dataset. |
| ✅ | `test_finbert_fallback_if_unavailable` | 695ms | Ensures FinBERT fallback returns neutral scores when the pipeline is unavailable. |
| ✅ | `test_finbert_model_f1_score` | 474ms | Validates FinBERT classification performance (Macro F1 ≥ 0.75) on a 24-sample news dataset. |
| ✅ | `test_is_english` | 312ms | Validates English language detection for non-English tweet filtering. |

### 🛑 Signal Handling
> **1/1** tests passed — total duration: **10ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_setup_signals_sets_event` | 10ms | Confirms that SIGTERM/SIGINT handlers are wired to set the shutdown event. |

### 🗄️ TimescaleDB Sink
> **1/1** tests passed — total duration: **2ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_write_routing` | 2ms | Verifies that incoming data is routed to the correct DB table (trades vs orderbook). |

### 🔍 XQuik Off-Topic Filter
> **8/8** tests passed — total duration: **<1ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_drops_news_about_another_coin_with_our_tag_appended` | <1ms | Drops tweets primarily about another coin with our coin's tag appended as spam. |
| ✅ | `test_drops_pure_tag_and_link_spam` | <1ms | Drops tweets that consist solely of hashtags and links with no real content. |
| ✅ | `test_drops_tag_blast_news_tweets` | <1ms | Drops mass-tagged news-blast tweets that mention many unrelated coins. |
| ✅ | `test_generic_tags_do_not_count_as_coin_tags` | <1ms | Ensures generic tags like #crypto or #blockchain are not counted as coin-specific. |
| ✅ | `test_keeps_small_multi_coin_comparisons` | <1ms | Keeps tweets comparing a small number of coins (legitimate discussion). |
| ✅ | `test_keeps_tweets_whose_prose_mentions_the_coin` | <1ms | Keeps tweets where the coin name appears in the prose body text. |
| ✅ | `test_keeps_tweets_with_coin_only_in_tag_but_real_prose` | <1ms | Keeps tweets that have the coin only in a hashtag but contain real prose content. |
| ✅ | `test_unknown_symbol_is_never_filtered` | <1ms | Ensures tweets for unknown/untracked symbols bypass the off-topic filter. |

### 🗃️ Database Integration (Transaction-Isolated)
> **2/2** tests passed — total duration: **3.42s**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_trade_aggregator` | 1.71s | Pushes mock trade data through the aggregator and verifies correct 5-min candle insertion. |
| ✅ | `test_orderbook_aggregator` | 1.71s | Pushes mock orderbook snapshots through the aggregator and verifies 5-min summary insertion. |

### 🌐 XQuik Live API
> **1/1** tests passed — total duration: **2m 9.1s**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_live_tweets` | 2m 9.1s | End-to-end live API call to XQuik to verify connectivity and response parsing. |

## 🐦 Live Tweet Sentiment Analysis

> **56** tweets collected from XQuik were scored by FinBERT

### Summary Statistics

| Metric | Value |
| :--- | :--- |
| **Overall Mood** | 🟢 Bullish |
| **Average Compound Score** | +0.3452 |
| **Positive Tweets** | 47 (84%) |
| **Negative Tweets** | 1 (2%) |
| **Neutral Tweets** | 8 (14%) |

### Individual Tweet Scores

| # | Sentiment | Compound | Pos | Neg | Neu | Tweet |
| :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | 🟢 | +0.8658 | 0.866 | 0.000 | 0.134 | On @solana, every second counts. In @SolgunGG, every battle matters. With $LOBO, every wolf in the pack has a role to… |
| 2 | 🟢 | +0.7418 | 0.743 | 0.001 | 0.255 | They’ve been trying to cover it up for years, diverting attention, trying to tarnish its image and none of that has w… |
| 3 | 🟢 | +0.6836 | 0.687 | 0.003 | 0.310 | Finally the US Bitcoin Reserve gets mentioned, it's been a long drought.   Let's get to work and make it happen in th… |
| 4 | 🟢 | +0.6725 | 0.673 | 0.000 | 0.327 | solana:32CdQdBUxbCsLy5AUHWmyidfwhgGUr9N573NBUrDpump 🚀 |
| 5 | 🟢 | +0.6521 | 0.654 | 0.002 | 0.343 | PillSwap deep dive 🔄🔥  The safest, fastest way to trade across chains. No more juggling DEXs, bridges, or guessing if… |
| 6 | 🟢 | +0.6521 | 0.654 | 0.002 | 0.343 | PillSwap deep dive 🔄🔥  The safest, fastest way to trade across chains. No more juggling DEXs, bridges, or guessing if… |
| 7 | 🟢 | +0.6104 | 0.611 | 0.000 | 0.389 | pumped to work with the chads at SF and Sunrise, the launch of $SPCX on @Solana is a great ecosystem moment, glad to … |
| 8 | 🟢 | +0.5677 | 0.639 | 0.072 | 0.289 | Swan is the band kids of Bitcoin but they don’t play any instruments |
| 9 | 🟢 | +0.5636 | 0.564 | 0.000 | 0.436 | great news seeing a patron of arts collecting pixelart  the strategy is indeed to build bridges with digital art coll… |
| 10 | 🟢 | +0.5635 | 0.570 | 0.006 | 0.424 | @WhiteHouse  #Does cryptocurrency have any meaning? There is only one way: banks can raise interest rates. Does it ma… |
| 11 | 🟢 | +0.5629 | 0.604 | 0.041 | 0.354 | Retail Ry says Bitcoin doesn’t need to fit anyone’s definition of what it should be  "Everyone has their idea of what… |
| 12 | 🟢 | +0.5495 | 0.550 | 0.001 | 0.449 | Bitcoin 1H Outlook 👀  The trend remains your friend until proven otherwise.  No need to chase candles—wait for confir… |
| 13 | 🟢 | +0.5464 | 0.547 | 0.000 | 0.453 | "Our general strategy is just keep acquiring Bitcoin, never sell the Bitcoin." https://t.co/sPwSh2smUL |
| 14 | 🟢 | +0.5433 | 0.543 | 0.000 | 0.457 | Prediction markets are redefining how modern quantitative funds consume alternative data. Platforms built on @solana … |
| 15 | 🟢 | +0.5432 | 0.544 | 0.000 | 0.456 | @mascots2026 let’s show them solana:8GxLxKA8tf3h8JUkXFfP4dNyn6D2vvwyGif5wanRpump is the running meta. Let’s pump |
| 16 | 🟢 | +0.5364 | 0.536 | 0.000 | 0.463 | The SEC abolishing the PDT rule is a massive macro catalyst. TradFi is now in direct competition with crypto for aggr… |
| 17 | 🟢 | +0.5230 | 0.523 | 0.000 | 0.477 | After a long time working on Yaoming on BNB running it to 2 million with the creator onboard, we decided to relaunch … |
| 18 | 🟢 | +0.5023 | 0.503 | 0.001 | 0.496 | The best decision I ever made had nothing to do with bitcoin. It was deciding to pursue Jesus and let that be the fil… |
| 19 | 🟢 | +0.4832 | 0.544 | 0.061 | 0.395 | This is where real assets are tokenized, for example: you can swap Bitcoin for gold, in a DEX space, I want to introd… |
| 20 | 🟢 | +0.4779 | 0.480 | 0.003 | 0.517 | Caution,,,,, c,,, #CexCrypto is a fake platform preying on investors! ❌ Stuck with blocked withdrawals? Don’t give up… |
| 21 | 🟢 | +0.4442 | 0.447 | 0.002 | 0.551 | Do not invest,,,,,,#CexCrypto is a fake platform preying on investors! ❌ Stuck with blocked withdrawals? Don’t give u… |
| 22 | 🟢 | +0.4442 | 0.447 | 0.002 | 0.551 | Do not invest,,,,,,#CexCrypto is a fake platform preying on investors! ❌ Stuck with blocked withdrawals? Don’t give u… |
| 23 | 🟢 | +0.4332 | 0.439 | 0.005 | 0.556 | 📊 TRAMDY Market Snapshot — June 11  ₿ BTC: $63,465 (+3.4% / 24h) Ξ ETH: $1,673 (+3.8% / 24h)  ⚙️ Derivatives check: •… |
| 24 | 🟢 | +0.4059 | 0.407 | 0.001 | 0.592 | 🔏 Top 10 Protocols by TVL: - Variations, June 11, 2026  1. Binance CEX $BNB - Chain: Multi-Chain, Category: CEX ▪ TVL… |
| 25 | 🟢 | +0.3943 | 0.427 | 0.032 | 0.541 | ''pls ignore the musical chairs ending and buy the token I bought 3% of at 20k marketcap''  If you're believing in co… |
| 26 | 🟢 | +0.3892 | 0.427 | 0.038 | 0.536 | Bitcoin Holders: This Is Hard To Ignore Now  Timestamps: 0:00 - tides are turning 1:36 - Long-term holders 2:09 - 81%… |
| 27 | 🟢 | +0.3742 | 0.375 | 0.001 | 0.623 | i’m giving away a free 1gb esım for estonia, bought via @nadanada_me with bitcoin lightning through @nuri   free give… |
| 28 | 🟢 | +0.3712 | 0.372 | 0.000 | 0.628 | https://t.co/0xTxG4vLXV - 🚀 Deposit Crypto &amp; Claim Up to $1,000 at Fair Go! 💰🎰 Play pokies with Bitcoin &amp; oth… |
| 29 | 🟢 | +0.3678 | 0.371 | 0.003 | 0.626 | @satsukikatayama  #Does cryptocurrency have any meaning? There is only one way: banks can raise interest rates. Does … |
| 30 | 🟢 | +0.3445 | 0.345 | 0.001 | 0.654 | Bid on more solana:9UuLsJ3jf8ViBNeRcwXD53re5G3ypgfKK3s2EiMMpump   aixbt knows...  https://t.co/Vc1bdbvGSY |
| 31 | 🟢 | +0.3256 | 0.332 | 0.006 | 0.662 | Privacy-preserving web search paid with Bitcoin + Cashu.  Under the hood: - Top up SEARCH tokens (Cashu Ecash) with L… |
| 32 | 🟢 | +0.3009 | 0.364 | 0.064 | 0.572 | 🚨 $BTC W-PATTERN BREAKOUT ATTEMPT 🚨  From 63.5k to 69.5k.  Next level: 70k+. Let's go. #Bitcoin https://t.co/HjioK2HPoF |
| 33 | 🟢 | +0.2971 | 0.351 | 0.054 | 0.595 | $btc signal of the day:  bear phase, 28 days in a row  price ~$63.2k  stress is high with 9.8m btc (48.8% of supply) … |
| 34 | 🟢 | +0.2900 | 0.291 | 0.001 | 0.709 | $BTC planning to make my first 50rr So help me God😔🤲🏽 Na red I wan dey see everywhere abeg😅💪🏾  #BTC https://t.co/jz1D… |
| 35 | 🟢 | +0.2819 | 0.386 | 0.104 | 0.510 | bitcoin:native   Most talk about cycles from a time perspective, arguing Bitcoin has rigid 4-year cycles, but I have … |
| 36 | 🟢 | +0.2583 | 0.264 | 0.006 | 0.731 | Ca: EbXdnYXgHGqMkcbjbtj2PKzi21oNaDYmqG45vpDnpump  Posted on the official Yaoming BNB page too |
| 37 | 🟢 | +0.2315 | 0.232 | 0.000 | 0.768 | Even in worst bearish scenario where $BTC dumps to $50k, I think $SOL is more likely to visit the $45-$50 zone than $… |
| 38 | 🟢 | +0.2310 | 0.232 | 0.001 | 0.768 | Crypto Market Update: total cap $2.259T, +3.4% in 24h on $81.285B volume. BTC dominance 56.3%, ETH 8.96%. Relief boun… |
| 39 | 🟢 | +0.2310 | 0.232 | 0.001 | 0.768 | Crypto Market Update: total cap $2.259T, +3.4% in 24h on $81.285B volume. BTC dominance 56.3%, ETH 8.96%. Relief boun… |
| 40 | 🟢 | +0.2060 | 0.206 | 0.000 | 0.794 | Bitcoin climbs 2.52% to $62,857 after recent volatility. Glassnode's 'The Bitcoin Vector #59' offers insights into cu… |
| 41 | 🟢 | +0.1955 | 0.196 | 0.000 | 0.804 | Bitcoin has been one of the biggest talking points this week.  A lot of people are looking for a single explanation b… |
| 42 | 🟢 | +0.1904 | 0.193 | 0.002 | 0.805 | 🚨 Options Expiry Alert 🚨  At 08:00 UTC tomorrow, ~$2.51B in crypto options are set to expire on Deribit.  $BTC: $2.23… |
| 43 | 🟢 | +0.1904 | 0.193 | 0.002 | 0.805 | 🚨 Options Expiry Alert 🚨  At 08:00 UTC tomorrow, ~$2.51B in crypto options are set to expire on Deribit.  $BTC: $2.23… |
| 44 | 🟢 | +0.1863 | 0.191 | 0.005 | 0.804 | This is the guy @KryptoFynn who claimed oil would breakout next week, and there hasn't been a single mention of oil s… |
| 45 | 🟢 | +0.1822 | 0.433 | 0.250 | 0.317 | The next phase of Solana will not be judged by TPS.  Institutions do not come onchain because a chain is cool.  They … |
| 46 | 🟢 | +0.1799 | 0.181 | 0.001 | 0.818 | 🎁 Daily Case claimed! 🚀  Use my link and unbox rewards worth up to $1,250 daily 👇 https://t.co/VbCDp4g2V4  #SOLPump #… |
| 47 | 🟢 | +0.1614 | 0.162 | 0.001 | 0.837 | USDC - Ethereum  86,563,537.15 USDC ($86,563,537)  DEX → Unknown  https://t.co/Pa5DWTIOQV |
| 48 | 🟡 | +0.1491 | 0.150 | 0.001 | 0.850 | solana:7HgfXftRBBqsYtAEYcqjGLQrNJLL6Tww9ek4rE3Apump only getting bigger https://t.co/wac7yaarFB |
| 49 | 🟡 | +0.1284 | 0.132 | 0.004 | 0.865 | #Bitcoin continues to move sideways, and nothing has changed yet. The price has not yet approached the resistance are… |
| 50 | 🟡 | +0.1263 | 0.127 | 0.000 | 0.873 | CONCISE AND UPDATED LIST OF SOLANA AGREEMENTS, DIVIDED BY CATEGORY    #Solana https://t.co/8pQaZUVWLg |
| 51 | 🟡 | +0.0788 | 0.079 | 0.000 | 0.921 | 💰$BTC/USDT \| 15m Timeframe 🕯  $BTC is currently trading within a triangle pattern on the shorter timeframe. We will … |
| 52 | 🟡 | +0.0623 | 0.063 | 0.000 | 0.937 | Requesting $BNB funds from the #Stakely Faucet on the BNB Chain blockchain. Request ID: IZ8URKBK https://t.co/OjmkOHT3tz |
| 53 | 🟡 | +0.0418 | 0.042 | 0.001 | 0.957 | Requesting $SOL funds from the #Stakely Faucet on the Solana blockchain. Request ID: IG0XT86L #privacy https://t.co/y… |
| 54 | 🟡 | +0.0345 | 0.035 | 0.000 | 0.965 | BNB perp liquidation pressure elevated at 44/100. 3 cascade events and 2 warning precursors in the last 2h. ADA and X… |
| 55 | 🟡 | -0.0722 | 0.199 | 0.271 | 0.530 | $BTC rejected from the key resistance level exactly as expected and retraced to test the ascending trendline support.… |
| 56 | 🔴 | -0.9698 | 0.004 | 0.974 | 0.022 | I can’t imagine buying the SpaceX IPO when you can buy Bitcoin at $62K |

## 📜 Raw Execution Logs

### Unit Tests
```
test_align_labels_to_sequences (test_data_processing.TestDataProcessing.test_align_labels_to_sequences) ... ok
test_slice_continuous_windows (test_data_processing.TestDataProcessing.test_slice_continuous_windows) ... ok
test_slice_continuous_windows_with_gaps (test_data_processing.TestDataProcessing.test_slice_continuous_windows_with_gaps) ... ok
test_dimensions_and_forward_pass (test_lstm_autoencoder.TestLSTMAutoencoder.test_dimensions_and_forward_pass) ... ok
test_parameters_gradients (test_lstm_autoencoder.TestLSTMAutoencoder.test_parameters_gradients) ... ok
test_retrain_job (test_retraining_scheduler.TestRetrainingScheduler.test_retrain_job) ... ok
test_start_and_shutdown_scheduler (test_retraining_scheduler.TestRetrainingScheduler.test_start_and_shutdown_scheduler) ... ok
test_backward_compat_alias (test_sentiment_scorer.TestSentimentScorer.test_backward_compat_alias) ... ok
test_compound_score (test_sentiment_scorer.TestSentimentScorer.test_compound_score) ... ok
test_cryptobert_fallback_if_unavailable (test_sentiment_scorer.TestSentimentScorer.test_cryptobert_fallback_if_unavailable) ... ok
test_cryptobert_model_f1_score (test_sentiment_scorer.TestSentimentScorer.test_cryptobert_model_f1_score) ... ok
test_finbert_fallback_if_unavailable (test_sentiment_scorer.TestSentimentScorer.test_finbert_fallback_if_unavailable) ... ok
test_finbert_model_f1_score (test_sentiment_scorer.TestSentimentScorer.test_finbert_model_f1_score) ... ok
test_is_english (test_sentiment_scorer.TestSentimentScorer.test_is_english) ... ok
test_setup_signals_sets_event (test_signals.TestSignals.test_setup_signals_sets_event) ... ok
test_write_routing (test_timescale_sink.TestTimescaleSink.test_write_routing) ... ok
test_drops_news_about_another_coin_with_our_tag_appended (test_xquik_filtering.TestXquikOfftopicFilter.test_drops_news_about_another_coin_with_our_tag_appended) ... ok
test_drops_pure_tag_and_link_spam (test_xquik_filtering.TestXquikOfftopicFilter.test_drops_pure_tag_and_link_spam) ... ok
test_drops_tag_blast_news_tweets (test_xquik_filtering.TestXquikOfftopicFilter.test_drops_tag_blast_news_tweets) ... ok
test_generic_tags_do_not_count_as_coin_tags (test_xquik_filtering.TestXquikOfftopicFilter.test_generic_tags_do_not_count_as_coin_tags) ... ok
test_keeps_small_multi_coin_comparisons (test_xquik_filtering.TestXquikOfftopicFilter.test_keeps_small_multi_coin_comparisons) ... ok
test_keeps_tweets_whose_prose_mentions_the_coin (test_xquik_filtering.TestXquikOfftopicFilter.test_keeps_tweets_whose_prose_mentions_the_coin) ... ok
test_keeps_tweets_with_coin_only_in_tag_but_real_prose (test_xquik_filtering.TestXquikOfftopicFilter.test_keeps_tweets_with_coin_only_in_tag_but_real_prose) ... ok
test_unknown_symbol_is_never_filtered (test_xquik_filtering.TestXquikOfftopicFilter.test_unknown_symbol_is_never_filtered) ... ok
```

### Database Integration (Transaction-Isolated)
```
🧪 Testing Trade Aggregator...
  Rows in DB: 2
  BTCUSDT: O=71000.0 H=71100.0 L=70900.0 C=70900.0 V=1.100 trades=4 buy=0.700 sell=0.400 net=0.300 vwap=71022.73
  BTCUSDT: O=71200.0 H=71200.0 L=71200.0 C=71200.0 V=0.010 trades=1 buy=0.010 sell=0.000 net=0.010 vwap=71200.00
  ✅ Trade Aggregator OK

🧪 Testing Orderbook Aggregator...
  Rows in DB: 2
  ETHUSDT: spread=1.0000 mid=2200.75 bid_depth=165.00 ask_depth=120.00 imbalance=0.1556 snapshots=2
  ETHUSDT: spread=1.0000 mid=2203.50 bid_depth=100.00 ask_depth=80.00 imbalance=0.1111 snapshots=1
  ✅ Orderbook Aggregator OK
```

### XQuik Live API
```

```

---
*Report generated automatically by `scripts/run_all_tests.py`.*