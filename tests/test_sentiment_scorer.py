import unittest
from unittest.mock import patch
from src.models.sentiment_models import (
    compound_score,
    score_news_batched,
    score_tweets_batched,
    score_texts_batched,
    is_english,
)

class TestSentimentScorer(unittest.TestCase):
    def test_compound_score(self):
        # Balanced neutral case
        self.assertAlmostEqual(compound_score({"positive": 0.1, "negative": 0.1, "neutral": 0.8}), 0.0)
        # Fully positive
        self.assertAlmostEqual(compound_score({"positive": 1.0, "negative": 0.0, "neutral": 0.0}), 1.0)
        # Fully negative
        self.assertAlmostEqual(compound_score({"positive": 0.0, "negative": 1.0, "neutral": 0.0}), -1.0)
        # Missing values (should default to 0.0)
        self.assertAlmostEqual(compound_score({}), 0.0)

    def test_finbert_fallback_if_unavailable(self):
        """When FinBERT pipeline is None, fallback should return neutral scores."""
        # Empty texts should return empty list
        res = score_news_batched([])
        self.assertEqual(res, [])
        
        # When FinBERT is unavailable, fallback returns neutral
        with patch("src.models.sentiment_models._get_finbert_pipeline", return_value=None):
            res_list = score_news_batched(["hello world"])
            self.assertEqual(len(res_list), 1)
            self.assertAlmostEqual(res_list[0]["neutral"], 1.0)
            self.assertAlmostEqual(res_list[0]["positive"], 0.0)
            self.assertAlmostEqual(res_list[0]["negative"], 0.0)

    def test_cryptobert_fallback_if_unavailable(self):
        """When CryptoBERT pipeline is None, fallback should return neutral scores."""
        res = score_tweets_batched([])
        self.assertEqual(res, [])

        with patch("src.models.sentiment_models._get_cryptobert_pipeline", return_value=None):
            res_list = score_tweets_batched(["BTC to the moon"])
            self.assertEqual(len(res_list), 1)
            self.assertAlmostEqual(res_list[0]["neutral"], 1.0)
            self.assertAlmostEqual(res_list[0]["positive"], 0.0)
            self.assertAlmostEqual(res_list[0]["negative"], 0.0)

    def test_backward_compat_alias(self):
        """score_texts_batched should be an alias for score_news_batched."""
        self.assertIs(score_texts_batched, score_news_batched)

    def test_is_english(self):
        """Validate English language detection."""
        # Clear English text
        self.assertTrue(is_english("Bitcoin rallied past 100k as institutional adoption accelerated."))
        # Clear non-English text
        self.assertFalse(is_english("Bitcoin yükselişe geçti, kurumsal yatırımcılar arttı."))
        self.assertFalse(is_english("Le Bitcoin a franchi les 100 000 dollars grâce à l'adoption institutionnelle."))
        # Noise stripping check (slang/emojis/links/tags should be cleaned, leaving English prose)
        self.assertTrue(is_english("LFG! $BTC to the moon 🚀🚀 #Bitcoin https://t.co/abc"))
        # Empty/whitespace text should return False
        self.assertFalse(is_english(""))
        self.assertFalse(is_english("   "))

    def test_finbert_model_f1_score(self):
        """Validate FinBERT classification performance (Macro F1 >= 0.75) on a validation dataset."""
        validation_data = [
            # ── Positive (8) ──────────────────────────────────────────
            ("Company profits rose by 50% this quarter, exceeding expectations.", "positive"),
            ("Revenue surged to record highs as demand increased.", "positive"),
            ("Shares jumped following the positive earnings report.", "positive"),
            ("Bitcoin rallied past $100k as institutional adoption accelerated.", "positive"),
            ("The firm announced a massive share buyback program, boosting investor confidence.", "positive"),
            ("Quarterly dividends increased by 20%, signaling strong cash flow.", "positive"),
            ("The acquisition was completed successfully, expanding market share significantly.", "positive"),
            ("Analysts upgraded the stock to 'strong buy' after impressive guidance.", "positive"),

            # ── Negative (8) ──────────────────────────────────────────
            ("Sales plummeted and the company is facing bankruptcy.", "negative"),
            ("The stock plunged after the fraud allegations were published.", "negative"),
            ("The company announced major layoffs and a decline in profits.", "negative"),
            ("Ethereum crashed 30% amid a broader crypto market sell-off.", "negative"),
            ("The SEC filed a lawsuit against the exchange for securities violations.", "negative"),
            ("Revenue missed estimates by a wide margin, leading to a sharp decline in share price.", "negative"),
            ("The CEO resigned abruptly following an internal investigation into misconduct.", "negative"),
            ("Credit rating was downgraded to junk status as debt concerns mounted.", "negative"),

            # ── Neutral (8) ───────────────────────────────────────────
            ("The company's stock price remained unchanged today.", "neutral"),
            ("The market is waiting for the federal reserve's decision.", "neutral"),
            ("They announced the scheduled date for their annual meeting.", "neutral"),
            ("Trading volume was in line with the 30-day average.", "neutral"),
            ("The central bank kept interest rates unchanged as expected.", "neutral"),
            ("The company released its standard quarterly filing with the SEC.", "neutral"),
            ("Bitcoin's hash rate remained stable over the past week.", "neutral"),
            ("The board appointed a new independent director to the audit committee.", "neutral"),
        ]
        
        texts = [item[0] for item in validation_data]
        true_labels = [item[1] for item in validation_data]
        
        # Score with FinBERT (news model)
        predictions = score_news_batched(texts)
        
        # Check if the fallback was triggered
        is_fallback = all(
            probs.get("neutral") == 1.0 and 
            probs.get("positive") == 0.0 and 
            probs.get("negative") == 0.0 
            for probs in predictions
        )
        if is_fallback:
            self.skipTest("FinBERT model is not loaded (transformers or device not available; fell back to neutral).")
            
        pred_labels = []
        for probs in predictions:
            pred_label = max(probs, key=probs.get)
            pred_labels.append(pred_label)
            
        classes = ["positive", "negative", "neutral"]
        f1_scores = []
        
        for cls in classes:
            tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
            
        macro_f1 = sum(f1_scores) / len(classes)
        print(f"\n🧪 FinBERT Macro F1 Score: {macro_f1:.4f}")
        
        self.assertGreaterEqual(macro_f1, 0.75, f"FinBERT F1 score {macro_f1:.4f} is below the 0.75 target threshold")

    def test_cryptobert_model_f1_score(self):
        """Validate CryptoBERT classification performance (Macro F1 >= 0.70) on crypto tweet data."""
        validation_data = [
            # ── Bullish / Positive ────────────────────────────────────
            ("$BTC to the moon! This rally is unstoppable 🚀🚀", "positive"),
            ("Just went all in on $ETH, this dip is a gift", "positive"),
            ("Massive bullish breakout on $SOL, next stop $300", "positive"),
            ("Bitcoin ETF approved! Institutions are flooding in", "positive"),
            ("$BTC breaking ATH again, bears are absolutely rekt", "positive"),
            ("Huge green candle on $ETH, the bull run is here", "positive"),

            # ── Bearish / Negative ────────────────────────────────────
            ("$BTC dumping hard, this is going to zero", "negative"),
            ("Another rug pull, crypto is a scam honestly", "negative"),
            ("Massive sell-off on $ETH, panic everywhere", "negative"),
            ("Lost everything on that $SOL trade, worst decision ever", "negative"),
            ("Exchange hacked, millions stolen. Crypto is dead", "negative"),
            ("$BTC crash incoming, the bubble is finally popping", "negative"),

            # ── Neutral ───────────────────────────────────────────────
            ("$BTC trading sideways around $95k today", "neutral"),
            ("Waiting for the FOMC decision before making any moves", "neutral"),
            ("Transferred my $ETH from Coinbase to cold storage", "neutral"),
            ("Bitcoin halving is scheduled for next month", "neutral"),
            ("New token listing announced on Binance for tomorrow", "neutral"),
            ("Daily trading volume for $SOL was about average today", "neutral"),
        ]

        texts = [item[0] for item in validation_data]
        true_labels = [item[1] for item in validation_data]

        predictions = score_tweets_batched(texts)

        is_fallback = all(
            probs.get("neutral") == 1.0 and
            probs.get("positive") == 0.0 and
            probs.get("negative") == 0.0
            for probs in predictions
        )
        if is_fallback:
            self.skipTest("CryptoBERT model is not loaded (fell back to neutral).")

        pred_labels = []
        for probs in predictions:
            pred_label = max(probs, key=probs.get)
            pred_labels.append(pred_label)

        classes = ["positive", "negative", "neutral"]
        f1_scores = []

        for cls in classes:
            tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(classes)
        print(f"\n🧪 CryptoBERT Macro F1 Score: {macro_f1:.4f}")

        self.assertGreaterEqual(macro_f1, 0.70, f"CryptoBERT F1 score {macro_f1:.4f} is below the 0.70 target threshold")

if __name__ == "__main__":
    unittest.main()
