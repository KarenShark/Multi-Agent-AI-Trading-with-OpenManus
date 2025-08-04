from app.agent.base import BaseAgent
from app.schema_trading import AnalystOutput, TickerItem, UniverseConfig
from app.tool.yfinance_fetcher import YFinanceFetcher


class MarketAnalystAgent(BaseAgent):
    def __init__(self, name="market_analyst"):
        super().__init__(name=name, system_prompt="")  # prompt 后续填
        self.yfinance_tool = YFinanceFetcher()

    def step(self, inputs):
        cfg = UniverseConfig(**inputs.get("universe", {}))

        # Use different stock pools based on objective
        if cfg.objective == "growth":
            candidate_tickers = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "NVDA",
                "META",
                "NFLX",
            ]
        elif cfg.objective == "value":
            candidate_tickers = ["BRK-B", "JPM", "V", "JNJ", "PG", "WMT", "HD", "UNH"]
        else:  # balanced
            candidate_tickers = [
                "AAPL",
                "MSFT",
                "AMZN",
                "GOOGL",
                "TSLA",
                "JPM",
                "JNJ",
                "V",
            ]

        # Limit to max_candidates
        selected_tickers = candidate_tickers[: cfg.max_candidates]

        # Fetch real company data
        try:
            result = self.yfinance_tool.execute(
                symbols=selected_tickers,
                period="1mo",
                info_fields=["marketCap", "sector", "industry", "trailingPE"],
            )

            if result.get("success"):
                company_info = result["data"]["company_info"]
                items = []
                rationale = {}

                for ticker in selected_tickers:
                    info = company_info.get(ticker, {})
                    items.append(
                        TickerItem(
                            symbol=ticker,
                            industry=info.get("industry"),
                            market_cap=info.get("marketCap"),
                        )
                    )

                    # Generate rationale based on objective
                    pe_ratio = info.get("trailingPE", "N/A")
                    market_cap = info.get("marketCap", 0)

                    if cfg.objective == "growth":
                        rationale[ticker] = (
                            f"Selected for growth potential. Industry: {info.get('industry', 'N/A')}, Market Cap: ${market_cap/1e9:.1f}B"
                        )
                    elif cfg.objective == "value":
                        rationale[ticker] = (
                            f"Selected for value opportunity. P/E: {pe_ratio}, Industry: {info.get('industry', 'N/A')}"
                        )
                    else:
                        rationale[ticker] = (
                            f"Balanced pick. P/E: {pe_ratio}, Market Cap: ${market_cap/1e9:.1f}B"
                        )

            else:
                # Fallback to simple list if data fetch fails
                items = [TickerItem(symbol=s) for s in selected_tickers]
                rationale = {
                    s: f"Selected based on {cfg.objective} objective"
                    for s in selected_tickers
                }

        except Exception as e:
            # Fallback to simple list if error occurs
            items = [TickerItem(symbol=s) for s in selected_tickers]
            rationale = {
                s: f"Selected based on {cfg.objective} objective (data fetch failed)"
                for s in selected_tickers
            }

        out = AnalystOutput(tickers=items, rationale=rationale)
        return {"analyst": out.model_dump()}
