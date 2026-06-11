"""Centralized test runner.

Discovers and executes all unit tests in the tests/ directory and runs
the transaction-isolated database integration test. Compiles and saves
a detailed Markdown test report to test_report.md.
"""
import sys
import os
import time
import unittest
import io
import traceback

# Adjust path to find src and scripts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from tests.test_integration import run_integration_tests
from src.db.db import close_pool


# ── Custom TestResult to capture per-test data ───────────────────────

class DetailedTestResult(unittest.TestResult):
    """Captures per-test timing, status, and failure details."""

    def __init__(self, stream=None, descriptions=True, verbosity=2):
        super().__init__(stream, descriptions, verbosity)
        self.test_details = []  # list of dicts
        self._test_start_time = None
        self._stream = stream

    def startTest(self, test):
        super().startTest(test)
        self._test_start_time = time.time()

    def _record(self, test, status, message=""):
        duration = time.time() - self._test_start_time if self._test_start_time else 0.0
        module = test.__class__.__module__ or ""
        class_name = test.__class__.__name__
        method = test._testMethodName
        docstring = test.shortDescription() or ""
        self.test_details.append({
            "module": module,
            "class": class_name,
            "method": method,
            "full_id": f"{module}.{class_name}.{method}",
            "status": status,
            "duration": duration,
            "message": message,
            "docstring": docstring,
        })

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "PASS")
        if self._stream:
            self._stream.write(f"{test} ... ok\n")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        tb = self._exc_info_to_string(err, test)
        self._record(test, "FAIL", tb)
        if self._stream:
            self._stream.write(f"{test} ... FAIL\n")

    def addError(self, test, err):
        super().addError(test, err)
        tb = self._exc_info_to_string(err, test)
        self._record(test, "ERROR", tb)
        if self._stream:
            self._stream.write(f"{test} ... ERROR\n")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "SKIP", reason)
        if self._stream:
            self._stream.write(f"{test} ... skipped: {reason}\n")


# ── Test descriptions (keyed by method name) ─────────────────────────

TEST_DESCRIPTIONS = {
    # Data Processing
    "test_align_labels_to_sequences":
        "Verifies that ground-truth labels are correctly aligned to windowed feature sequences.",
    "test_slice_continuous_windows":
        "Validates sliding-window slicing on continuous time-series data without gaps.",
    "test_slice_continuous_windows_with_gaps":
        "Ensures the window slicer correctly handles temporal gaps in the input data.",
    # LSTM Autoencoder
    "test_dimensions_and_forward_pass":
        "Checks that the LSTM Autoencoder output tensor dimensions match the input.",
    "test_parameters_gradients":
        "Confirms that gradients flow through all trainable parameters during backprop.",
    # Retraining Scheduler
    "test_start_and_shutdown_scheduler":
        "Tests that the APScheduler-based retraining loop starts and shuts down cleanly.",
    "test_retrain_job":
        "Verifies that the periodic retrain job invokes the training pipeline correctly.",
    # Sentiment Scorer
    "test_compound_score":
        "Validates the compound score formula: score = P(positive) − P(negative).",
    "test_finbert_fallback_if_unavailable":
        "Ensures FinBERT fallback returns neutral scores when the pipeline is unavailable.",
    "test_cryptobert_fallback_if_unavailable":
        "Ensures CryptoBERT fallback returns neutral scores when the pipeline is unavailable.",
    "test_backward_compat_alias":
        "Verifies score_texts_batched is a backward-compatible alias for score_news_batched.",
    "test_is_english":
        "Validates English language detection for non-English tweet filtering.",
    "test_finbert_model_f1_score":
        "Validates FinBERT classification performance (Macro F1 ≥ 0.75) on a 24-sample news dataset.",
    "test_cryptobert_model_f1_score":
        "Validates CryptoBERT classification performance (Macro F1 ≥ 0.70) on a 18-sample crypto tweet dataset.",
    # Signals
    "test_setup_signals_sets_event":
        "Confirms that SIGTERM/SIGINT handlers are wired to set the shutdown event.",
    # TimescaleDB Sink
    "test_write_routing":
        "Verifies that incoming data is routed to the correct DB table (trades vs orderbook).",
    # XQuik Filtering
    "test_keeps_tweets_whose_prose_mentions_the_coin":
        "Keeps tweets where the coin name appears in the prose body text.",
    "test_keeps_tweets_with_coin_only_in_tag_but_real_prose":
        "Keeps tweets that have the coin only in a hashtag but contain real prose content.",
    "test_keeps_small_multi_coin_comparisons":
        "Keeps tweets comparing a small number of coins (legitimate discussion).",
    "test_drops_tag_blast_news_tweets":
        "Drops mass-tagged news-blast tweets that mention many unrelated coins.",
    "test_drops_news_about_another_coin_with_our_tag_appended":
        "Drops tweets primarily about another coin with our coin's tag appended as spam.",
    "test_drops_pure_tag_and_link_spam":
        "Drops tweets that consist solely of hashtags and links with no real content.",
    "test_generic_tags_do_not_count_as_coin_tags":
        "Ensures generic tags like #crypto or #blockchain are not counted as coin-specific.",
    "test_unknown_symbol_is_never_filtered":
        "Ensures tweets for unknown/untracked symbols bypass the off-topic filter.",
    # Integration
    "test_trade_aggregator":
        "Pushes mock trade data through the aggregator and verifies correct 5-min candle insertion.",
    "test_orderbook_aggregator":
        "Pushes mock orderbook snapshots through the aggregator and verifies 5-min summary insertion.",
    # XQuik Live API
    "test_live_tweets":
        "End-to-end live API call to XQuik to verify connectivity and response parsing.",
}

# Map module names to friendly component names
COMPONENT_MAP = {
    "test_data_processing": "📊 Data Processing",
    "test_lstm_autoencoder": "🧠 LSTM Autoencoder",
    "test_retraining_scheduler": "🔄 Retraining Scheduler",
    "test_sentiment_scorer": "💬 Sentiment Scorer (FinBERT)",
    "test_signals": "🛑 Signal Handling",
    "test_timescale_sink": "🗄️ TimescaleDB Sink",
    "test_xquik_filtering": "🔍 XQuik Off-Topic Filter",
}

STATUS_ICONS = {
    "PASS": "✅",
    "FAIL": "❌",
    "ERROR": "💥",
    "SKIP": "⏭️",
}


def run_suite():
    """Discover and run all unit tests, returning detailed per-test results."""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(ROOT_DIR, "tests"), pattern="test_*.py")

    stream = io.StringIO()
    result = DetailedTestResult(stream=stream)

    start_time = time.time()
    suite(result)
    duration = time.time() - start_time

    return result, stream.getvalue(), duration


def _format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 0.001:
        return "<1ms"
    elif seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def _build_detailed_per_test_section(all_details):
    """Build per-component tables with individual test rows."""
    lines = []
    lines.append("## 📋 Detailed Per-Test Results")
    lines.append("")

    # Group by component (module)
    from collections import OrderedDict
    groups = OrderedDict()
    for d in all_details:
        mod = d["module"]
        groups.setdefault(mod, []).append(d)

    for module, tests in groups.items():
        component_name = COMPONENT_MAP.get(module, f"🧩 {module}")
        passed = sum(1 for t in tests if t["status"] == "PASS")
        total = len(tests)
        component_duration = sum(t["duration"] for t in tests)

        lines.append(f"### {component_name}")
        lines.append(f"> **{passed}/{total}** tests passed — total duration: **{_format_duration(component_duration)}**")
        lines.append("")
        lines.append("| Status | Test | Duration | Description |")
        lines.append("| :---: | :--- | :--- | :--- |")

        for t in tests:
            icon = STATUS_ICONS.get(t["status"], "❓")
            method = t["method"]
            dur = _format_duration(t["duration"])
            desc = TEST_DESCRIPTIONS.get(method, t["docstring"] or "—")
            # Truncate long descriptions
            if len(desc) > 120:
                desc = desc[:117] + "…"
            lines.append(f"| {icon} | `{method}` | {dur} | {desc} |")

        lines.append("")

        # Show failure/error details inline
        for t in tests:
            if t["status"] in ("FAIL", "ERROR") and t["message"]:
                lines.append(f"<details>")
                lines.append(f"<summary>{STATUS_ICONS[t['status']]} <code>{t['method']}</code> — failure details</summary>")
                lines.append("")
                lines.append("```")
                lines.append(t["message"].rstrip())
                lines.append("```")
                lines.append("</details>")
                lines.append("")

    return lines


def main():
    print("=========================================")
    print("🚀 Running CryptoSense Test Suite")
    print("=========================================")

    overall_start = time.time()

    # ── 1. Run unit tests ─────────────────────────────────────────
    unit_result, unit_log, unit_duration = run_suite()

    # ── 2. Run integration tests ──────────────────────────────────
    integration_passed = True
    integration_error = ""
    integration_start = time.time()

    integration_stream = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = integration_stream

    try:
        run_integration_tests()
    except Exception as e:
        integration_passed = False
        traceback.print_exc()
        integration_error = str(e)
    finally:
        sys.stdout = original_stdout
        close_pool()

    integration_duration = time.time() - integration_start
    integration_log = integration_stream.getvalue()

    # ── 3. Run XQuik Live API tests ───────────────────────────────
    xquick_passed = True
    xquick_error = ""
    xquick_start = time.time()
    collected_tweets = []

    xquick_stream = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = xquick_stream

    try:
        import asyncio
        from tests.test_xquick import test_live_tweets
        collected_tweets = asyncio.run(test_live_tweets()) or []
    except Exception as e:
        xquick_passed = False
        traceback.print_exc()
        xquick_error = str(e)
    finally:
        sys.stdout = original_stdout

    xquick_duration = time.time() - xquick_start
    xquick_log = xquick_stream.getvalue()

    # ── 4. Score collected tweets with CryptoBERT ─────────────────
    tweet_scores = []
    if collected_tweets:
        print(f"🔬 Scoring {len(collected_tweets)} collected tweets with CryptoBERT...")
        try:
            from src.models.sentiment_models import score_tweets_batched, compound_score
            batch_probs = score_tweets_batched(collected_tweets)
            for text, probs in zip(collected_tweets, batch_probs):
                tweet_scores.append({
                    "text": text,
                    "positive": probs.get("positive", 0.0),
                    "negative": probs.get("negative", 0.0),
                    "neutral": probs.get("neutral", 0.0),
                    "compound": compound_score(probs),
                })
            print(f"✅ Sentiment scoring complete for {len(tweet_scores)} tweets.")
        except Exception as e:
            print(f"⚠️ Sentiment scoring failed: {e}")
            traceback.print_exc()

    overall_duration = time.time() - overall_start

    # ── 4. Compile all test details ───────────────────────────────
    # Add integration test entries to the detail list
    integration_details = [
        {
            "module": "test_integration",
            "class": "TransactionIsolated",
            "method": "test_trade_aggregator",
            "full_id": "test_integration.TransactionIsolated.test_trade_aggregator",
            "status": "PASS" if integration_passed else "FAIL",
            "duration": integration_duration / 2,  # approximate split
            "message": integration_error if not integration_passed else "",
            "docstring": "",
        },
        {
            "module": "test_integration",
            "class": "TransactionIsolated",
            "method": "test_orderbook_aggregator",
            "full_id": "test_integration.TransactionIsolated.test_orderbook_aggregator",
            "status": "PASS" if integration_passed else "FAIL",
            "duration": integration_duration / 2,
            "message": integration_error if not integration_passed else "",
            "docstring": "",
        },
    ]

    xquick_details = [
        {
            "module": "test_xquick",
            "class": "XQuikLiveAPI",
            "method": "test_live_tweets",
            "full_id": "test_xquick.XQuikLiveAPI.test_live_tweets",
            "status": "PASS" if xquick_passed else "FAIL",
            "duration": xquick_duration,
            "message": xquick_error if not xquick_passed else "",
            "docstring": "",
        },
    ]

    COMPONENT_MAP["test_integration"] = "🗃️ Database Integration (Transaction-Isolated)"
    COMPONENT_MAP["test_xquick"] = "🌐 XQuik Live API"

    all_details = unit_result.test_details + integration_details + xquick_details

    # ── 5. Compute summary stats ──────────────────────────────────
    total_tests = unit_result.testsRun
    failed_tests = len(unit_result.failures) + len(unit_result.errors)
    passed_tests = total_tests - failed_tests
    skipped_tests = len(unit_result.skipped)

    all_passed = failed_tests == 0 and integration_passed and xquick_passed
    status = "SUCCESS" if all_passed else "FAILED"

    grand_total = total_tests + 2 + 1  # unit + 2 integration + 1 xquik
    grand_passed = passed_tests + (2 if integration_passed else 0) + (1 if xquick_passed else 0)
    grand_failed = grand_total - grand_passed

    # ── 6. Build report ───────────────────────────────────────────
    report = []
    report.append("# CryptoSense Automated Test Suite Execution Report")
    report.append("")
    report.append(f"**Execution Status**: {'🟢 PASS' if all_passed else '🔴 FAIL'}")
    report.append(f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Duration**: {_format_duration(overall_duration)}")
    report.append(f"**Python Version**: {sys.version.split()[0]}")
    report.append("")

    # ── Executive Summary ─────────────────────────────────────────
    report.append("## Executive Summary")
    report.append("")
    report.append(f"> **{grand_passed}/{grand_total}** tests passed across **3** test suites")
    report.append("")
    report.append("| Test Suite | Total Tests | Passed | Failed | Skipped | Duration |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    report.append(
        f"| Unit Tests | {total_tests} | {passed_tests} | {failed_tests} | {skipped_tests} | {_format_duration(unit_duration)} |"
    )
    report.append(
        f"| Database Integration | 2 | {2 if integration_passed else 0} | {0 if integration_passed else 2} | 0 | {_format_duration(integration_duration)} |"
    )
    report.append(
        f"| XQuik Live API | 1 | {1 if xquick_passed else 0} | {0 if xquick_passed else 1} | 0 | {_format_duration(xquick_duration)} |"
    )
    report.append(
        f"| **TOTAL** | **{grand_total}** | **{grand_passed}** | **{grand_failed}** | **{skipped_tests}** | **{_format_duration(overall_duration)}** |"
    )
    report.append("")

    # ── Failures Section (if any) ─────────────────────────────────
    if not all_passed:
        report.append("## ❌ Failures Details")
        for failure in unit_result.failures:
            report.append(f"### Unit Test Failure: `{failure[0]}`")
            report.append("```")
            report.append(failure[1])
            report.append("```")
            report.append("")
        for error in unit_result.errors:
            report.append(f"### Unit Test Error: `{error[0]}`")
            report.append("```")
            report.append(error[1])
            report.append("```")
            report.append("")
        if not integration_passed:
            report.append("### Database Integration Test Failure")
            report.append("```")
            report.append(integration_log)
            if integration_error:
                report.append(f"Error details: {integration_error}")
            report.append("```")
            report.append("")
        if not xquick_passed:
            report.append("### XQuik Live API Test Failure")
            report.append("```")
            report.append(xquick_log)
            if xquick_error:
                report.append(f"Error details: {xquick_error}")
            report.append("```")
            report.append("")

    # ── Detailed Per-Test Results ─────────────────────────────────
    report.extend(_build_detailed_per_test_section(all_details))

    # ── Live Tweet Sentiment Analysis ─────────────────────────────
    if tweet_scores:
        report.append("## 🐦 Live Tweet Sentiment Analysis")
        report.append("")
        report.append(f"> **{len(tweet_scores)}** tweets collected from XQuik were scored by FinBERT")
        report.append("")

        # Compute summary stats
        avg_compound = sum(t["compound"] for t in tweet_scores) / len(tweet_scores)
        pos_count = sum(1 for t in tweet_scores if t["compound"] > 0.15)
        neg_count = sum(1 for t in tweet_scores if t["compound"] < -0.15)
        neu_count = len(tweet_scores) - pos_count - neg_count

        if avg_compound > 0.15:
            overall_mood = "🟢 Bullish"
        elif avg_compound < -0.15:
            overall_mood = "🔴 Bearish"
        else:
            overall_mood = "🟡 Neutral"

        report.append("### Summary Statistics")
        report.append("")
        report.append(f"| Metric | Value |")
        report.append(f"| :--- | :--- |")
        report.append(f"| **Overall Mood** | {overall_mood} |")
        report.append(f"| **Average Compound Score** | {avg_compound:+.4f} |")
        report.append(f"| **Positive Tweets** | {pos_count} ({pos_count / len(tweet_scores) * 100:.0f}%) |")
        report.append(f"| **Negative Tweets** | {neg_count} ({neg_count / len(tweet_scores) * 100:.0f}%) |")
        report.append(f"| **Neutral Tweets** | {neu_count} ({neu_count / len(tweet_scores) * 100:.0f}%) |")
        report.append("")

        # Sort by compound score descending (most positive first)
        sorted_tweets = sorted(tweet_scores, key=lambda t: t["compound"], reverse=True)

        report.append("### Individual Tweet Scores")
        report.append("")
        report.append("| # | Sentiment | Compound | Pos | Neg | Neu | Tweet |")
        report.append("| :---: | :---: | :--- | :--- | :--- | :--- | :--- |")

        for i, t in enumerate(sorted_tweets, 1):
            # Determine sentiment emoji
            if t["compound"] > 0.15:
                emoji = "🟢"
            elif t["compound"] < -0.15:
                emoji = "🔴"
            else:
                emoji = "🟡"

            # Clean and truncate tweet text for table display
            clean_text = t["text"].replace("\n", " ").replace("|", "\\|")
            if len(clean_text) > 120:
                clean_text = clean_text[:117] + "…"

            report.append(
                f"| {i} | {emoji} | {t['compound']:+.4f} | {t['positive']:.3f} | "
                f"{t['negative']:.3f} | {t['neutral']:.3f} | {clean_text} |"
            )

        report.append("")
    elif collected_tweets is not None:
        report.append("## 🐦 Live Tweet Sentiment Analysis")
        report.append("")
        report.append("> No tweets were collected during the XQuik monitoring window.")
        report.append("")

    report.append("## 📜 Raw Execution Logs")
    report.append("")
    report.append("### Unit Tests")
    report.append("```")
    report.append(unit_log.rstrip())
    report.append("```")
    report.append("")
    report.append("### Database Integration (Transaction-Isolated)")
    report.append("```")
    report.append(integration_log.rstrip())
    report.append("```")
    report.append("")
    report.append("### XQuik Live API")
    report.append("```")
    report.append(xquick_log.rstrip())
    report.append("```")
    report.append("")

    report.append("---")
    report.append("*Report generated automatically by `scripts/run_all_tests.py`.*")

    report_content = "\n".join(report)

    # ── Save report ───────────────────────────────────────────────
    report_path = os.path.join(ROOT_DIR, "test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Report saved to: {report_path}")
    print(f"Overall Status: {status}")
    print(f"Total: {grand_passed}/{grand_total} tests passed in {_format_duration(overall_duration)}")
    print("=========================================")

    if status == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

