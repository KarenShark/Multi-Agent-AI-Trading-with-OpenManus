from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.tool.base import BaseTool


class TechnicalIndicators(BaseTool):
    name: str = "technical_indicators"
    description: str = (
        "Calculate technical indicators like MA, RSI, MACD for stock analysis"
    )

    parameters: dict = {
        "type": "object",
        "properties": {
            "prices": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of price values (typically closing prices)",
            },
            "indicators": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of indicators to calculate: sma, ema, rsi, macd, bollinger",
                "default": ["sma", "rsi", "macd"],
            },
            "sma_period": {
                "type": "integer",
                "description": "Period for Simple Moving Average",
                "default": 20,
            },
            "ema_period": {
                "type": "integer",
                "description": "Period for Exponential Moving Average",
                "default": 20,
            },
            "rsi_period": {
                "type": "integer",
                "description": "Period for RSI calculation",
                "default": 14,
            },
            "macd_fast": {
                "type": "integer",
                "description": "Fast period for MACD",
                "default": 12,
            },
            "macd_slow": {
                "type": "integer",
                "description": "Slow period for MACD",
                "default": 26,
            },
            "macd_signal": {
                "type": "integer",
                "description": "Signal period for MACD",
                "default": 9,
            },
        },
        "required": ["prices"],
    }

    def execute(
        self,
        prices: List[float],
        indicators: List[str] = None,
        sma_period: int = 20,
        ema_period: int = 20,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ) -> Dict:
        """
        Calculate technical indicators for given price data

        Args:
            prices: List of price values
            indicators: List of indicators to calculate
            Various period parameters for different indicators

        Returns:
            Dictionary containing calculated indicators
        """
        if indicators is None:
            indicators = ["sma", "rsi", "macd"]

        if len(prices) < max(sma_period, ema_period, rsi_period, macd_slow):
            return {
                "error": "Insufficient price data for calculations",
                "success": False,
            }

        result = {}

        try:
            df = pd.DataFrame({"close": prices})

            # Simple Moving Average
            if "sma" in indicators:
                result["sma"] = df["close"].rolling(window=sma_period).mean().tolist()

            # Exponential Moving Average
            if "ema" in indicators:
                result["ema"] = df["close"].ewm(span=ema_period).mean().tolist()

            # RSI
            if "rsi" in indicators:
                result["rsi"] = self._calculate_rsi(df["close"], rsi_period).tolist()

            # MACD
            if "macd" in indicators:
                macd_data = self._calculate_macd(
                    df["close"], macd_fast, macd_slow, macd_signal
                )
                result["macd"] = {
                    "macd": macd_data["macd"].tolist(),
                    "signal": macd_data["signal"].tolist(),
                    "histogram": macd_data["histogram"].tolist(),
                }

            # Bollinger Bands
            if "bollinger" in indicators:
                bb_data = self._calculate_bollinger_bands(df["close"], sma_period)
                result["bollinger"] = {
                    "upper": bb_data["upper"].tolist(),
                    "middle": bb_data["middle"].tolist(),
                    "lower": bb_data["lower"].tolist(),
                }

        except Exception as e:
            return {
                "error": f"Failed to calculate indicators: {str(e)}",
                "success": False,
            }

        return {"indicators": result, "success": True}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line

        return {"macd": macd, "signal": signal_line, "histogram": histogram}

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        return {
            "upper": sma + (std * std_dev),
            "middle": sma,
            "lower": sma - (std * std_dev),
        }
