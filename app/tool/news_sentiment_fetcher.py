from typing import Dict, List, Optional

import requests
import yfinance as yf
from textblob import TextBlob

from app.tool.base import BaseTool


class NewsSentimentFetcher(BaseTool):
    name: str = "news_sentiment_fetcher"
    description: str = "Fetch news and analyze sentiment for stock symbols"

    parameters: dict = {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock symbols to fetch news for",
            },
            "max_articles": {
                "type": "integer",
                "description": "Maximum number of articles to analyze per symbol",
                "default": 10,
            },
            "days_back": {
                "type": "integer",
                "description": "Number of days back to fetch news",
                "default": 7,
            },
        },
        "required": ["symbols"],
    }

    def execute(
        self, symbols: List[str], max_articles: int = 10, days_back: int = 7
    ) -> Dict:
        """
        Fetch news and analyze sentiment for given symbols

        Args:
            symbols: List of stock symbols
            max_articles: Maximum articles per symbol
            days_back: Days back to fetch news

        Returns:
            Dictionary containing news sentiment analysis
        """
        result = {"sentiment_scores": {}, "news_articles": {}, "summary": {}}

        try:
            for symbol in symbols:
                # Get news using yfinance (fallback method)
                ticker = yf.Ticker(symbol)
                news = ticker.news

                if not news:
                    result["sentiment_scores"][symbol] = 0.0
                    result["news_articles"][symbol] = []
                    continue

                articles = []
                sentiments = []

                for article in news[:max_articles]:
                    title = article.get("title", "")
                    summary = article.get("summary", "")

                    # Combine title and summary for sentiment analysis
                    text = f"{title}. {summary}"

                    # Simple sentiment analysis using TextBlob
                    blob = TextBlob(text)
                    sentiment = (
                        blob.sentiment.polarity
                    )  # Range: -1 (negative) to 1 (positive)

                    articles.append(
                        {
                            "title": title,
                            "summary": summary,
                            "sentiment": sentiment,
                            "url": article.get("link", ""),
                            "publish_time": article.get("providerPublishTime", 0),
                        }
                    )

                    sentiments.append(sentiment)

                # Calculate average sentiment
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

                result["sentiment_scores"][symbol] = avg_sentiment
                result["news_articles"][symbol] = articles
                result["summary"][symbol] = {
                    "avg_sentiment": avg_sentiment,
                    "total_articles": len(articles),
                    "positive_articles": len([s for s in sentiments if s > 0.1]),
                    "negative_articles": len([s for s in sentiments if s < -0.1]),
                    "neutral_articles": len(
                        [s for s in sentiments if -0.1 <= s <= 0.1]
                    ),
                }

        except Exception as e:
            return {
                "error": f"Failed to fetch news sentiment: {str(e)}",
                "success": False,
            }

        return {"data": result, "success": True}

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.1:
            return "Positive"
        elif score < -0.1:
            return "Negative"
        else:
            return "Neutral"
