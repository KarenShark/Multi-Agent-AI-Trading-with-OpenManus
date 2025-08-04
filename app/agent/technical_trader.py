from app.agent.base import BaseAgent
from app.schema_trading import AnalystOutput, SignalItem, TechnicalOutput
from app.tool.indicators import TechnicalIndicators
from app.tool.yfinance_fetcher import YFinanceFetcher


class TechnicalTraderAgent(BaseAgent):
    def __init__(self, name="technical_trader"):
        super().__init__(name=name, system_prompt="")
        self.yfinance_tool = YFinanceFetcher()
        self.indicators_tool = TechnicalIndicators()

    def step(self, inputs):
        ana = AnalystOutput(**inputs["analyst"])
        symbols = [t.symbol for t in ana.tickers]

        signals = {}

        try:
            # Fetch historical price data
            result = self.yfinance_tool.execute(
                symbols=symbols,
                period="3mo",  # 3 months of data for technical analysis
                interval="1d",
            )

            if result.get("success"):
                historical_data = result["data"]["historical_data"]

                for symbol in symbols:
                    if symbol in historical_data:
                        prices = historical_data[symbol]["close"]

                        # Calculate technical indicators
                        indicators_result = self.indicators_tool.execute(
                            prices=prices,
                            indicators=["sma", "rsi", "macd"],
                            sma_period=20,
                            rsi_period=14,
                        )

                        if indicators_result.get("success"):
                            indicators = indicators_result["indicators"]

                            # Simple trading logic based on indicators
                            signal = self._generate_signal(indicators, prices)
                            signals[symbol] = signal
                        else:
                            signals[symbol] = SignalItem(action="flat", confidence=0.5)
                    else:
                        signals[symbol] = SignalItem(action="flat", confidence=0.5)
            else:
                # Fallback to flat signals
                signals = {
                    symbol: SignalItem(action="flat", confidence=0.5)
                    for symbol in symbols
                }

        except Exception as e:
            # Fallback to flat signals if error occurs
            signals = {
                symbol: SignalItem(action="flat", confidence=0.5) for symbol in symbols
            }

        out = TechnicalOutput(signals=signals)
        return {"technical": out.model_dump()}

    def _generate_signal(self, indicators, prices):
        """Generate trading signal based on technical indicators"""
        try:
            current_price = prices[-1]
            sma = indicators.get("sma", [])[-1] if indicators.get("sma") else None
            rsi = indicators.get("rsi", [])[-1] if indicators.get("rsi") else None
            macd_data = indicators.get("macd", {})

            # Simple signal logic
            bullish_signals = 0
            bearish_signals = 0

            # SMA signal
            if sma and current_price > sma:
                bullish_signals += 1
            elif sma and current_price < sma:
                bearish_signals += 1

            # RSI signal
            if rsi:
                if rsi < 30:  # Oversold
                    bullish_signals += 1
                elif rsi > 70:  # Overbought
                    bearish_signals += 1

            # MACD signal
            if macd_data:
                macd_line = macd_data.get("macd", [])
                signal_line = macd_data.get("signal", [])

                if len(macd_line) >= 2 and len(signal_line) >= 2:
                    if (
                        macd_line[-1] > signal_line[-1]
                        and macd_line[-2] <= signal_line[-2]
                    ):
                        bullish_signals += 1
                    elif (
                        macd_line[-1] < signal_line[-1]
                        and macd_line[-2] >= signal_line[-2]
                    ):
                        bearish_signals += 1

            # Generate final signal
            if bullish_signals > bearish_signals:
                confidence = min(0.8, bullish_signals / 3.0)
                return SignalItem(action="long", confidence=confidence)
            elif bearish_signals > bullish_signals:
                confidence = min(0.8, bearish_signals / 3.0)
                return SignalItem(action="short", confidence=confidence)
            else:
                return SignalItem(action="flat", confidence=0.5)

        except Exception:
            return SignalItem(action="flat", confidence=0.5)
