from app.agent.base import BaseAgent
from app.schema_trading import AnalystOutput, SentimentOutput
from app.tool.news_sentiment_fetcher import NewsSentimentFetcher


class SentimentAnalyzerAgent(BaseAgent):
    def __init__(self, name="sentiment_analyzer"):
        super().__init__(name=name, system_prompt="")
        self.news_tool = NewsSentimentFetcher()

    def step(self, inputs):
        ana = AnalystOutput(**inputs["analyst"])
        symbols = [t.symbol for t in ana.tickers]

        try:
            # Fetch real news sentiment
            result = self.news_tool.execute(
                symbols=symbols, max_articles=5, days_back=7
            )

            if result.get("success"):
                sentiment_data = result["data"]
                scores = sentiment_data["sentiment_scores"]

                # Prepare sources information
                sources = {}
                for symbol in symbols:
                    articles = sentiment_data["news_articles"].get(symbol, [])
                    sources[symbol] = [
                        article["title"] for article in articles[:3]
                    ]  # Top 3 headlines

                out = SentimentOutput(scores=scores, sources=sources)
            else:
                # Fallback to neutral scores
                scores = {symbol: 0.0 for symbol in symbols}
                out = SentimentOutput(scores=scores, sources={})

        except Exception as e:
            # Fallback to neutral scores if error occurs
            scores = {symbol: 0.0 for symbol in symbols}
            out = SentimentOutput(scores=scores, sources={})

        return {"sentiment": out.model_dump()}
