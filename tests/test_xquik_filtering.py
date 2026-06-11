import unittest

from src.data_sources.xquik.xquik_ingestion import _is_offtopic_news_tweet


class TestXquikOfftopicFilter(unittest.TestCase):
    """The keyword monitors match hashtag-stuffed generic crypto-news tweets;
    _is_offtopic_news_tweet must drop those while keeping genuine coin tweets."""

    def test_keeps_tweets_whose_prose_mentions_the_coin(self):
        self.assertFalse(_is_offtopic_news_tweet("BTC", "Bitcoin smashes through $100k as ETFs pile in"))
        self.assertFalse(_is_offtopic_news_tweet("AVAX", "Avalanche subnet activity hits all-time high #AVAX #crypto"))
        self.assertFalse(_is_offtopic_news_tweet("BNB", "BNB burn scheduled for next week looks bullish"))

    def test_keeps_tweets_with_coin_only_in_tag_but_real_prose(self):
        # The coin appears only as a tag, but the tweet has substance and no
        # multi-coin tag blast — typical genuine retail tweet
        self.assertFalse(_is_offtopic_news_tweet("BTC", "Just bought the dip! #Bitcoin"))
        self.assertFalse(_is_offtopic_news_tweet("BTC", "$BTC breaking out of the falling wedge, target 120k"))

    def test_keeps_small_multi_coin_comparisons(self):
        # Two coin tags is still plausibly about both coins
        self.assertFalse(_is_offtopic_news_tweet("SOL", "$SOL $BTC which one pumps harder this cycle?"))

    def test_drops_tag_blast_news_tweets(self):
        text = "Daily crypto market update: top movers today https://t.co/x #BTC #ETH #SOL #AVAX #BNB"
        self.assertTrue(_is_offtopic_news_tweet("SOL", text))
        self.assertTrue(_is_offtopic_news_tweet("BNB", text))

    def test_drops_news_about_another_coin_with_our_tag_appended(self):
        text = "Bitcoin smashes through $100k as ETFs pile in #BTC #ETH #SOL #DOGE #crypto"
        self.assertTrue(_is_offtopic_news_tweet("ETH", text))
        # …but the coin the story is actually about is kept
        self.assertFalse(_is_offtopic_news_tweet("BTC", text))

    def test_drops_pure_tag_and_link_spam(self):
        self.assertTrue(_is_offtopic_news_tweet("BTC", "#BTC #ETH #SOL #XRP #DOGE #ADA https://t.co/abc"))
        self.assertTrue(_is_offtopic_news_tweet("BTC", "gm 🚀🚀 #BTC"))

    def test_generic_tags_do_not_count_as_coin_tags(self):
        # #crypto/#trading/#news must not push a tweet over the coin-tag limit
        self.assertFalse(
            _is_offtopic_news_tweet("ETH", "Gas fees dropping fast on L2s right now #ETH #crypto #trading #news")
        )

    def test_unknown_symbol_is_never_filtered(self):
        self.assertFalse(_is_offtopic_news_tweet("DOGE", "#BTC #ETH #SOL #XRP wall of tags"))


if __name__ == "__main__":
    unittest.main()
