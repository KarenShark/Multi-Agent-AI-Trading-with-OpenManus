from app.agent.base import BaseAgent
from app.schema_trading import AnalystOutput, SentimentOutput
from app.tool.enhanced_news_fetcher import EnhancedNewsFetcher
from app.tool.news_sentiment_fetcher import NewsSentimentFetcher


class SentimentAnalyzerAgent(BaseAgent):
    def __init__(self, name="sentiment_analyzer"):
        super().__init__(name=name, system_prompt="")
        self.news_tool = NewsSentimentFetcher()  # Legacy tool as fallback
        self.enhanced_news_tool = EnhancedNewsFetcher()  # Enhanced tool as primary

    def step(self, inputs):
        ana = AnalystOutput(**inputs["analyst"])
        symbols = [t.symbol for t in ana.tickers]

        try:
            # Try enhanced news API first
            result = self.enhanced_news_tool.execute(
                symbols=symbols, max_articles=8, days_back=5
            )

            if result.get("success"):
                sentiment_data = result["data"]

                # Extract enhanced sentiment scores
                scores = {}
                sources = {}

                for symbol in symbols:
                    symbol_data = sentiment_data.get(symbol, {})

                    # Use time-weighted sentiment as primary score
                    scores[symbol] = symbol_data.get("time_weighted_sentiment", 0.0)

                    # Prepare enhanced sources with sentiment context
                    articles = symbol_data.get("articles", [])
                    article_summaries = []

                    for i, article in enumerate(articles[:3]):
                        title = article.get("title", "")
                        source = article.get("source", "")
                        # Add sentiment indicator to title
                        sentiment_score = symbol_data.get("sentiment_score", 0.0)
                        if sentiment_score > 0.1:
                            indicator = "📈"
                        elif sentiment_score < -0.1:
                            indicator = "📉"
                        else:
                            indicator = "📊"

                        article_summaries.append(f"{indicator} {title} ({source})")

                    sources[symbol] = article_summaries

                # Add sentiment confidence and breakdown info
                sentiment_metadata = {}
                for symbol in symbols:
                    symbol_data = sentiment_data.get(symbol, {})
                    sentiment_metadata[symbol] = {
                        "confidence": symbol_data.get("relevance_score", 0.0),
                        "article_count": symbol_data.get("article_count", 0),
                        "breakdown": symbol_data.get("sentiment_breakdown", {}),
                        "keywords": list(
                            symbol_data.get("keyword_sentiment", {}).keys()
                        ),
                    }

                out = SentimentOutput(
                    scores=scores, sources=sources, metadata=sentiment_metadata
                )

            else:
                # Fallback to legacy tool
                print("Enhanced news API failed, falling back to legacy tool...")
                fallback_result = self.news_tool.execute(
                    symbols=symbols, max_articles=5, days_back=7
                )

                if fallback_result.get("success"):
                    fallback_data = fallback_result["data"]
                    scores = fallback_data["sentiment_scores"]

                    sources = {}
                    for symbol in symbols:
                        articles = fallback_data["news_articles"].get(symbol, [])
                        sources[symbol] = [article["title"] for article in articles[:3]]

                    out = SentimentOutput(scores=scores, sources=sources)
                else:
                    # Final fallback to neutral
                    scores = {symbol: 0.0 for symbol in symbols}
                    out = SentimentOutput(scores=scores, sources={})

        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            # Fallback to neutral scores if error occurs
            scores = {symbol: 0.0 for symbol in symbols}
            sources = {symbol: ["No news data available"] for symbol in symbols}
            out = SentimentOutput(scores=scores, sources=sources)

        return {"sentiment": out.model_dump()}
