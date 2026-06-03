import unittest
from src.feature_engineering.sentiment_scorer import compound_score, score_texts_batched

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

    def test_score_texts_batched_fallback_if_empty(self):
        # Empty texts should return neutral scores
        res = score_texts_batched([])
        self.assertEqual(res, [])
        
        # In case texts are provided but FinBERT model fails/fallback triggers, it should return neutral scores
        res_list = score_texts_batched(["hello world"])
        self.assertEqual(len(res_list), 1)
        self.assertAlmostEqual(res_list[0]["neutral"], 1.0)
        self.assertAlmostEqual(res_list[0]["positive"], 0.0)
        self.assertAlmostEqual(res_list[0]["negative"], 0.0)

    def test_sentiment_model_f1_score(self):
        """Validate FinBERT classification performance (Macro F1 >= 0.75) on a validation dataset."""
        # 1. Define a small validation dataset of clear financial texts and their expected labels.
        validation_data = [
            ("Company profits rose by 50% this quarter, exceeding expectations.", "positive"),
            ("Revenue surged to record highs as demand increased.", "positive"),
            ("Shares jumped following the positive earnings report.", "positive"),
            
            ("Sales plummeted and the company is facing bankruptcy.", "negative"),
            ("The stock plunged after the fraud allegations were published.", "negative"),
            ("The company announced major layoffs and a decline in profits.", "negative"),
            
            ("The company's stock price remained unchanged today.", "neutral"),
            ("The market is waiting for the federal reserve's decision.", "neutral"),
            ("They announced the scheduled date for their annual meeting.", "neutral"),
        ]
        
        texts = [item[0] for item in validation_data]
        true_labels = [item[1] for item in validation_data]
        
        # 2. Run batched scoring
        predictions = score_texts_batched(texts)
        
        # Check if the fallback was triggered (which returns exactly 1.0 neutral and 0.0 positive/negative for all texts)
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
            # Predict the label with the highest score
            pred_label = max(probs, key=probs.get)
            pred_labels.append(pred_label)
            
        # Calculate Precision, Recall, and F1 score for each class, then take macro average
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
        
        self.assertGreaterEqual(macro_f1, 0.75, f"F1 score {macro_f1:.4f} is below the 0.75 target threshold")

if __name__ == "__main__":
    unittest.main()
