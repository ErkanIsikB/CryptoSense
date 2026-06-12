# CryptoSense Automated Test Suite Execution Report

**Execution Status**: 🟢 PASS
**Date/Time**: 2026-06-12 18:41:11
**Total Duration**: 4m 15.2s
**Python Version**: 3.12.10

## Executive Summary

> **27/27** tests passed across **3** test suites

| Test Suite | Total Tests | Passed | Failed | Skipped | Duration |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Unit Tests | 24 | 24 | 0 | 0 | 1m 51.5s |
| Database Integration | 2 | 2 | 0 | 0 | 3.87s |
| XQuik Live API | 1 | 1 | 0 | 0 | 2m 9.4s |
| **TOTAL** | **27** | **27** | **0** | **0** | **4m 15.2s** |

## 📋 Detailed Per-Test Results

### 📊 Data Processing
> **3/3** tests passed — total duration: **9ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_align_labels_to_sequences` | 9ms | Verifies that ground-truth labels are correctly aligned to windowed feature sequences. |
| ✅ | `test_slice_continuous_windows` | <1ms | Validates sliding-window slicing on continuous time-series data without gaps. |
| ✅ | `test_slice_continuous_windows_with_gaps` | <1ms | Ensures the window slicer correctly handles temporal gaps in the input data. |

### 🧠 LSTM Autoencoder
> **2/2** tests passed — total duration: **720ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_dimensions_and_forward_pass` | 74ms | Checks that the LSTM Autoencoder output tensor dimensions match the input. |
| ✅ | `test_parameters_gradients` | 646ms | Confirms that gradients flow through all trainable parameters during backprop. |

### 🔄 Retraining Scheduler
> **2/2** tests passed — total duration: **6ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_retrain_job` | 5ms | Verifies that the periodic retrain job invokes the training pipeline correctly. |
| ✅ | `test_start_and_shutdown_scheduler` | 2ms | Tests that the APScheduler-based retraining loop starts and shuts down cleanly. |

### 💬 Sentiment Scorer (FinBERT)
> **7/7** tests passed — total duration: **1m 50.7s**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_backward_compat_alias` | <1ms | Verifies score_texts_batched is a backward-compatible alias for score_news_batched. |
| ✅ | `test_compound_score` | <1ms | Validates the compound score formula: score = P(positive) − P(negative). |
| ✅ | `test_cryptobert_fallback_if_unavailable` | 1m 47.5s | Ensures CryptoBERT fallback returns neutral scores when the pipeline is unavailable. |
| ✅ | `test_cryptobert_model_f1_score` | 532ms | Validates CryptoBERT classification performance (Macro F1 ≥ 0.70) on a 18-sample crypto tweet dataset. |
| ✅ | `test_finbert_fallback_if_unavailable` | 1.91s | Ensures FinBERT fallback returns neutral scores when the pipeline is unavailable. |
| ✅ | `test_finbert_model_f1_score` | 78ms | Validates FinBERT classification performance (Macro F1 ≥ 0.75) on a 24-sample news dataset. |
| ✅ | `test_is_english` | 670ms | Validates English language detection for non-English tweet filtering. |

### 🛑 Signal Handling
> **1/1** tests passed — total duration: **3ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_setup_signals_sets_event` | 3ms | Confirms that SIGTERM/SIGINT handlers are wired to set the shutdown event. |

### 🗄️ TimescaleDB Sink
> **1/1** tests passed — total duration: **5ms**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_write_routing` | 5ms | Verifies that incoming data is routed to the correct DB table (trades vs orderbook). |

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
> **2/2** tests passed — total duration: **3.87s**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_trade_aggregator` | 1.94s | Pushes mock trade data through the aggregator and verifies correct 5-min candle insertion. |
| ✅ | `test_orderbook_aggregator` | 1.94s | Pushes mock orderbook snapshots through the aggregator and verifies 5-min summary insertion. |

### 🌐 XQuik Live API
> **1/1** tests passed — total duration: **2m 9.4s**

| Status | Test | Duration | Description |
| :---: | :--- | :--- | :--- |
| ✅ | `test_live_tweets` | 2m 9.4s | End-to-end live API call to XQuik to verify connectivity and response parsing. |

## 🐦 Live Tweet Sentiment Analysis

> **55** tweets collected from XQuik were scored by FinBERT

### Summary Statistics

| Metric | Value |
| :--- | :--- |
| **Overall Mood** | 🟢 Bullish |
| **Average Compound Score** | +0.3865 |
| **Positive Tweets** | 45 (82%) |
| **Negative Tweets** | 3 (5%) |
| **Neutral Tweets** | 7 (13%) |

### Individual Tweet Scores

| # | Sentiment | Compound | Pos | Neg | Neu | Tweet |
| :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | 🟢 | +0.7855 | 0.786 | 0.001 | 0.213 | Perfect time to remind everyone that SpaceX is now the 8th largest public company holder of Bitcoin.  It has 18,712 b… |
| 2 | 🟢 | +0.7823 | 0.789 | 0.006 | 0.205 | My thesis is the best way to invest into A.I, data centers, robotics, space? Etc is to buy and hold Bitcoin |
| 3 | 🟢 | +0.7614 | 0.762 | 0.000 | 0.238 | Almost a year ago, my @solanamobile arrived.  Between moving across the world, changing cities, dealing with visas, f… |
| 4 | 🟢 | +0.7501 | 0.750 | 0.000 | 0.250 | Bitcoin could drop to around the $60k zone again, and then bounce back up to the $70,000 - $100,000 zone.  $BTC is ju… |
| 5 | 🟢 | +0.7054 | 0.706 | 0.000 | 0.294 | 🚨BREAKING:  Bitcoin surges above $64,000, triggering $30 million in shorts liquidation in just one hour. https://t.co… |
| 6 | 🟢 | +0.6931 | 0.693 | 0.000 | 0.307 | solana:7YMkZZwdcwUbXKjYpr5gFAVoB6aF4f9iLWK6pUcppump to $10m mcap. https://t.co/EwjWkm6n2L |
| 7 | 🟢 | +0.6916 | 0.693 | 0.002 | 0.305 | Been up for like 30 hours finally going back to my air bnb to rest up 😭 |
| 8 | 🟢 | +0.6822 | 0.682 | 0.000 | 0.317 | Buying $GODL is now officially as easy as ordering Uber and here's why you should get positioned  GODL is a digital s… |
| 9 | 🟢 | +0.6670 | 0.667 | 0.000 | 0.333 | The largest initial public offering The world's first trillionaire  One asset behind both: $SPCX   Accessible to EVER… |
| 10 | 🟢 | +0.6579 | 0.683 | 0.025 | 0.293 | 🐉 Just rolled into ROLLING UP ($ROLL) 🥢  Sick of rugpulls leaving a bad taste? Time for something fresh. ROLLING UP b… |
| 11 | 🟢 | +0.6423 | 0.643 | 0.000 | 0.357 | New presale alert! 📢  @Forbidden_Oasis is a hidden paradise in the crypto desert. 🏝️ ➡️ BNB rewards for holders ➡️ Tr… |
| 12 | 🟢 | +0.6350 | 0.635 | 0.000 | 0.364 | 🟣 $GREM is on fire. 🔥  The @GREMTOKEN liquidity pool has become one of the most active environments in the $ASTY netw… |
| 13 | 🟢 | +0.6193 | 0.620 | 0.000 | 0.380 | solana:4nV5gNwwP68zUDat26ySChREqVaQaLudfJBkSgEzpump and $WOJAK 2 iconic memes with delusional holders who want to see… |
| 14 | 🟢 | +0.6103 | 0.610 | 0.000 | 0.390 | 1/ The first Venture Token on @solana is @SP3NDdotshop.  SP3ND is already live, generating revenue, and helping users… |
| 15 | 🟢 | +0.5686 | 0.569 | 0.001 | 0.430 | ethereum:0x68749665ff8d2d112fa859aa293f07a622782f38 send it higher https://t.co/z27vcXWiwa |
| 16 | 🟢 | +0.5623 | 0.564 | 0.001 | 0.435 | Last 60m - #Coinbase Spot (USD Trades)  📈 Top 3 Gainers: $GIGA (Gigachad) : ↑ 9.06% $FIGHT (FIGHT) : ↑ 7.04% $ALLO (A… |
| 17 | 🟢 | +0.5623 | 0.564 | 0.001 | 0.435 | Last 60m - #Coinbase Spot (USD Trades)  📈 Top 3 Gainers: $GIGA (Gigachad) : ↑ 9.06% $FIGHT (FIGHT) : ↑ 7.04% $ALLO (A… |
| 18 | 🟢 | +0.5553 | 0.555 | 0.000 | 0.444 | Bitcoin is back above $64,000  $20,000,000,000 has been added to crypto market in just 30 MINUTES https://t.co/tQ7qjE… |
| 19 | 🟢 | +0.5475 | 0.548 | 0.000 | 0.452 | Looks like a good day to launch my memecoin too on solana. Have been delaying it for months due to transfer of funds … |
| 20 | 🟢 | +0.5462 | 0.553 | 0.007 | 0.441 | THE WORLD COMPUTER THESIS ∞  For seventeen years, cryptocurrencies have demonstrated one thing:  They can tokenize sp… |
| 21 | 🟢 | +0.5348 | 0.540 | 0.005 | 0.456 | As expected, $BTC and $ETH rebounded after finding their local bottoms. June's recovery was largely driven by the rel… |
| 22 | 🟢 | +0.5348 | 0.540 | 0.005 | 0.456 | As expected, $BTC and $ETH rebounded after finding their local bottoms. June's recovery was largely driven by the rel… |
| 23 | 🟢 | +0.5212 | 0.522 | 0.000 | 0.478 | all I want is a deviation from this $btc range so we can understand the next move is that too hard to ask? |
| 24 | 🟢 | +0.5075 | 0.508 | 0.000 | 0.492 | I will be participating in the Hot Emin Trials and $HOTEMIN launch on $AVAX 🔺 |
| 25 | 🟢 | +0.5075 | 0.508 | 0.000 | 0.492 | I will be participating in the Hot Emin Trials and $HOTEMIN launch on $AVAX 🔺 |
| 26 | 🟢 | +0.4998 | 0.502 | 0.002 | 0.496 | IA says the truth, there is no second best 🤣  Ethereum |
| 27 | 🟢 | +0.4851 | 0.485 | 0.000 | 0.514 | ok yeah ggs ethereum:0x893643f9e232e4e857f278d61641c955589a7a37 is going so high, this teams network is so goated. I’… |
| 28 | 🟢 | +0.4615 | 0.465 | 0.003 | 0.532 | Bitcoin is on sale👊🏻 The opportunity won't be Buy your Bitcoin P2P on  @hodlhodl   Withdraw it Take custody 🧡 Trezor … |
| 29 | 🟢 | +0.4225 | 0.423 | 0.001 | 0.576 | 🇯🇵 Japan Just Made Crypto Official  A 55% tax on crypto gains. That's what Japanese investors have been dealing with.… |
| 30 | 🟢 | +0.4157 | 0.416 | 0.000 | 0.584 | We officially have the animal runner for June   solana:7YMkZZwdcwUbXKjYpr5gFAVoB6aF4f9iLWK6pUcppump |
| 31 | 🟢 | +0.4151 | 0.416 | 0.001 | 0.583 | What Your D0 Bot Can Do: Trading    Trade crypto directly in Telegram with D0 — safe, fast, and simple.  You can: ▪️S… |
| 32 | 🟢 | +0.4011 | 0.401 | 0.000 | 0.599 | 🎉If someone had advised you to hoard Bitcoin when it was worthless, you might have thought it was a scam. But today, … |
| 33 | 🟢 | +0.3757 | 0.377 | 0.001 | 0.623 | Glad to announce that i secured @dedmundos GTD spots for my communities.  Mint Details •Supply: 2,500 •Price: FREE MI… |
| 34 | 🟢 | +0.3548 | 0.355 | 0.000 | 0.645 | Closing in on 500,000 $BAT locked in the cave!  1179 stakers earning $guano simply by locking $BAT on Solana. https:/… |
| 35 | 🟢 | +0.3451 | 0.347 | 0.002 | 0.651 | $BTC in Midterm Election Years https://t.co/4viY0qbwah |
| 36 | 🟢 | +0.3445 | 0.345 | 0.000 | 0.655 | gn Jupiter fam, @JupiterExchange joining the Solana RPC Working Group matters, better read layer means cleaner pricin… |
| 37 | 🟢 | +0.3435 | 0.345 | 0.002 | 0.653 | Went To Dallas got an air bnb but was all alone 😩 so I had to take care of myself 😭 #nsfwtwt #freak #solo #single htt… |
| 38 | 🟢 | +0.3360 | 0.338 | 0.002 | 0.660 | 📢 MAJOR ANNOUNCEMENT: WinWave unveils one of its biggest welcome offers yet 🎰 ￳ 📌 More Info in Article Below 👇👇👇 http… |
| 39 | 🟢 | +0.3360 | 0.338 | 0.002 | 0.660 | 📢 MAJOR ANNOUNCEMENT: WinWave unveils one of its biggest welcome offers yet 🎰 ￳ 📌 More Info in Article Below 👇👇👇 http… |
| 40 | 🟢 | +0.3302 | 0.400 | 0.070 | 0.530 | Arthur Hayes says a major SpaceX IPO could pull even more capital away from Bitcoin and into AI.  Is AI taking all th… |
| 41 | 🟢 | +0.3253 | 0.325 | 0.000 | 0.675 | Crypto markets see mixed news today Polish president vetoes crypto bill, KuCoin dispute   Regulatory debates and FTX … |
| 42 | 🟢 | +0.3040 | 0.305 | 0.001 | 0.695 | Bears are fighting back hard $BTC https://t.co/MuXpmjkb8F |
| 43 | 🟢 | +0.2990 | 0.300 | 0.001 | 0.699 | took some profit here   $BTC https://t.co/GM1DP1bywr |
| 44 | 🟢 | +0.2419 | 0.242 | 0.000 | 0.757 | $BTC broke up! Will it hold! https://t.co/dC7D4Xf8lv |
| 45 | 🟢 | +0.2195 | 0.223 | 0.003 | 0.774 | CRV - DOGE - BNB - SOL (XRP is the same as SOL)  Why does one of the four have nothing to do with the others?  I am r… |
| 46 | 🟡 | +0.1235 | 0.254 | 0.131 | 0.615 | The first metric that Hyperliquid has flipped Binance over  Wait until hyperliquid:native flips BNB |
| 47 | 🟡 | +0.1162 | 0.117 | 0.000 | 0.883 | USDC - Ethereum  95,587,074.93 USDC ($95,587,075)  Unknown → Unknown  https://t.co/RKIj8ltB6B |
| 48 | 🟡 | +0.0846 | 0.085 | 0.000 | 0.915 | Shoutout   @Booksey #Community  solana:3TYgKwkE2Y3rxdw9osLRSpxpXmSC1C1oo19W9KHspump   3TYgKwkE2Y3rxdw9osLRSpxpXmSC1C1… |
| 49 | 🟡 | +0.0326 | 0.052 | 0.019 | 0.929 | $BTC gave multiple warnings.  Two failed breakouts.  A channel breakdown.  And now a sharp move lower.  The bearish s… |
| 50 | 🟡 | +0.0247 | 0.220 | 0.195 | 0.585 | $ETH could be a +1 for our planed short if we hit this supply at the same time with $BTC supply and USDT.D demand htt… |
| 51 | 🟡 | -0.0029 | 0.395 | 0.398 | 0.207 | JUST IN: More Bitcoin Holders Are Underwater Than Profitable, Says Pompliano |
| 52 | 🟡 | -0.0418 | 0.328 | 0.370 | 0.302 | My biggest takeaway after reading this:  While many people are still debating whether Bitcoin is an asset, others hav… |
| 53 | 🔴 | -0.1586 | 0.326 | 0.485 | 0.189 | Wild that we’ve been here multiple times in the past 5 years and now that we’re nearing legislation that finally lets… |
| 54 | 🔴 | -0.8541 | 0.057 | 0.912 | 0.031 | Bitcoin 🙂 |
| 55 | 🔴 | -0.9537 | 0.005 | 0.959 | 0.036 | Shorted $BTC here at 64k. https://t.co/ANVx9bbQik |

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