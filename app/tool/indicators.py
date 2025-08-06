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
            "bb_period": {
                "type": "integer",
                "description": "Period for Bollinger Bands",
                "default": 20,
            },
            "bb_std": {
                "type": "number",
                "description": "Standard deviation multiplier for Bollinger Bands",
                "default": 2.0,
            },
            "stoch_k": {
                "type": "integer",
                "description": "K period for Stochastic Oscillator",
                "default": 14,
            },
            "stoch_d": {
                "type": "integer",
                "description": "D period for Stochastic Oscillator",
                "default": 3,
            },
            "williams_r": {
                "type": "integer",
                "description": "Period for Williams %R",
                "default": 14,
            },
        },
        "required": ["prices"],
    }

    def execute(
        self,
        prices: List[float],
        high_prices: List[float] = None,
        low_prices: List[float] = None,
        indicators: List[str] = None,
        sma_period: int = 20,
        ema_period: int = 20,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        stoch_k: int = 14,
        stoch_d: int = 3,
        williams_r: int = 14,
    ) -> Dict:
        """
        Calculate technical indicators for given price data

        Args:
            prices: List of close price values
            high_prices: List of high price values (for stochastic, Williams %R)
            low_prices: List of low price values (for stochastic, Williams %R)
            indicators: List of indicators to calculate
            Various period parameters for different indicators

        Returns:
            Dictionary containing calculated indicators
        """
        if indicators is None:
            indicators = ["sma", "rsi", "macd"]

        # Check if we have enough data for the most demanding indicator
        min_required = max(
            sma_period,
            ema_period,
            rsi_period,
            macd_slow,
            bb_period,
            stoch_k,
            williams_r,
        )
        if len(prices) < min_required:
            return {
                "error": f"Insufficient price data for calculations (need {min_required}, got {len(prices)})",
                "success": False,
            }

        result = {}

        try:
            df = pd.DataFrame({"close": prices})
            if high_prices:
                df["high"] = high_prices
            if low_prices:
                df["low"] = low_prices

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
                bb_data = self._calculate_bollinger_bands(
                    df["close"], bb_period, bb_std
                )
                result["bollinger"] = {
                    "upper": bb_data["upper"].tolist(),
                    "middle": bb_data["middle"].tolist(),
                    "lower": bb_data["lower"].tolist(),
                }

            # Stochastic Oscillator
            if (
                "stochastic" in indicators
                and "high" in df.columns
                and "low" in df.columns
            ):
                stoch_data = self._calculate_stochastic(
                    df["high"], df["low"], df["close"], stoch_k, stoch_d
                )
                result["stochastic"] = {
                    "k": stoch_data["k"].tolist(),
                    "d": stoch_data["d"].tolist(),
                }

            # Williams %R
            if (
                "williams_r" in indicators
                and "high" in df.columns
                and "low" in df.columns
            ):
                williams_data = self._calculate_williams_r(
                    df["high"], df["low"], df["close"], williams_r
                )
                result["williams_r"] = williams_data.tolist()

            # MACD Signal Analysis (enhanced)
            if "macd_signals" in indicators:
                macd_data = self._calculate_macd(
                    df["close"], macd_fast, macd_slow, macd_signal
                )
                signals = self._analyze_macd_signals(macd_data)
                result["macd_signals"] = signals

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
        self, prices: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        return {
            "upper": sma + (std * std_dev),
            "middle": sma,
            "lower": sma - (std * std_dev),
        }

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Dict:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {
            "k": k_percent,
            "d": d_percent,
        }

    def _calculate_williams_r(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r

    def _analyze_macd_signals(self, macd_data: Dict) -> Dict:
        """Analyze MACD for trading signals"""
        macd = macd_data["macd"]
        signal = macd_data["signal"]
        histogram = macd_data["histogram"]

        # Generate trading signals
        signals = []
        for i in range(1, len(macd)):
            if pd.isna(macd.iloc[i]) or pd.isna(signal.iloc[i]):
                signals.append("neutral")
                continue

            # Bullish signal: MACD crosses above signal line
            if macd.iloc[i] > signal.iloc[i] and macd.iloc[i - 1] <= signal.iloc[i - 1]:
                signals.append("bullish_crossover")
            # Bearish signal: MACD crosses below signal line
            elif (
                macd.iloc[i] < signal.iloc[i] and macd.iloc[i - 1] >= signal.iloc[i - 1]
            ):
                signals.append("bearish_crossover")
            # Bullish divergence: histogram increasing
            elif histogram.iloc[i] > histogram.iloc[i - 1]:
                signals.append("bullish_momentum")
            # Bearish divergence: histogram decreasing
            elif histogram.iloc[i] < histogram.iloc[i - 1]:
                signals.append("bearish_momentum")
            else:
                signals.append("neutral")

        # Add first signal as neutral
        signals.insert(0, "neutral")

        return {
            "signals": signals,
            "latest_signal": signals[-1] if signals else "neutral",
            "signal_strength": (
                abs(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0
            ),
        }
