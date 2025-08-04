from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from app.tool.base import BaseTool


class YFinanceFetcher(BaseTool):
    name: str = "yfinance_fetcher"
    description: str = "Fetch financial data using yfinance library for stocks analysis"

    parameters: dict = {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock symbols to fetch data for",
            },
            "period": {
                "type": "string",
                "description": "Time period for historical data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
                "default": "1y",
            },
            "interval": {
                "type": "string",
                "description": "Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)",
                "default": "1d",
            },
            "info_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Company info fields to retrieve (marketCap, sector, industry, etc.)",
                "default": [
                    "marketCap",
                    "sector",
                    "industry",
                    "trailingPE",
                    "forwardPE",
                ],
            },
        },
        "required": ["symbols"],
    }

    def execute(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        info_fields: List[str] = None,
    ) -> Dict:
        """
        Fetch financial data for given symbols

        Args:
            symbols: List of stock symbols
            period: Time period for historical data
            interval: Data interval
            info_fields: Company info fields to retrieve

        Returns:
            Dictionary containing historical data and company info
        """
        if info_fields is None:
            info_fields = ["marketCap", "sector", "industry", "trailingPE", "forwardPE"]

        result = {"historical_data": {}, "company_info": {}, "current_prices": {}}

        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)

                # Get historical data
                hist = ticker.history(period=period, interval=interval)
                if not hist.empty:
                    result["historical_data"][symbol] = {
                        "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                        "open": hist["Open"].tolist(),
                        "high": hist["High"].tolist(),
                        "low": hist["Low"].tolist(),
                        "close": hist["Close"].tolist(),
                        "volume": hist["Volume"].tolist(),
                    }

                    # Current price (latest close)
                    result["current_prices"][symbol] = float(hist["Close"].iloc[-1])

                # Get company info
                info = ticker.info
                company_data = {}
                for field in info_fields:
                    company_data[field] = info.get(field, None)
                result["company_info"][symbol] = company_data

        except Exception as e:
            return {"error": f"Failed to fetch data: {str(e)}", "success": False}

        return {"data": result, "success": True}
