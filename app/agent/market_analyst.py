from app.agent.base import BaseAgent
from app.schema_trading import AnalystOutput, TickerItem, UniverseConfig
from app.tool.fundamental_fetcher import FundamentalFetcher
from app.tool.yfinance_fetcher import YFinanceFetcher


class MarketAnalystAgent(BaseAgent):
    def __init__(self, name="market_analyst"):
        super().__init__(name=name, system_prompt="")  # prompt 后续填
        self.yfinance_tool = YFinanceFetcher()
        self.fundamental_tool = FundamentalFetcher()

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

        # Perform fundamental analysis-based selection
        try:
            print(f"Performing fundamental analysis for {cfg.objective} strategy...")

            # Fetch fundamental data for all candidates
            fundamental_result = self.fundamental_tool.execute(
                symbols=candidate_tickers,
                metrics=[
                    "valuation",
                    "profitability",
                    "liquidity",
                    "leverage",
                    "growth",
                ],
            )

            if fundamental_result.get("success"):
                # Score and rank candidates based on objective
                scored_candidates = self._score_candidates(
                    fundamental_result["data"], cfg.objective
                )

                # Select top candidates
                selected_tickers = [
                    item[0] for item in scored_candidates[: cfg.max_candidates]
                ]

                # Get detailed info for selected tickers
                items = []
                rationale = {}

                for ticker in selected_tickers:
                    ticker_data = fundamental_result["data"].get(ticker, {})
                    company_info = ticker_data.get("raw_data", {}).get(
                        "company_info", {}
                    )
                    analysis = ticker_data.get("analysis", {})
                    calculated_metrics = ticker_data.get("calculated_metrics", {})

                    items.append(
                        TickerItem(
                            symbol=ticker,
                            industry=company_info.get("industry"),
                            market_cap=company_info.get("market_cap"),
                        )
                    )

                    # Generate detailed rationale based on fundamental analysis
                    rationale[ticker] = self._generate_detailed_rationale(
                        ticker, ticker_data, cfg.objective
                    )
            else:
                # Fallback to basic selection if fundamental analysis fails
                print("Fundamental analysis failed, falling back to basic selection...")
                selected_tickers = candidate_tickers[: cfg.max_candidates]
                items, rationale = self._basic_selection_fallback(
                    selected_tickers, cfg.objective
                )

        except Exception as e:
            print(f"Error in fundamental analysis: {e}")
            # Fallback to basic selection
            selected_tickers = candidate_tickers[: cfg.max_candidates]
            items, rationale = self._basic_selection_fallback(
                selected_tickers, cfg.objective
            )

        out = AnalystOutput(tickers=items, rationale=rationale)
        return {"analyst": out.model_dump()}

    def _score_candidates(self, fundamental_data: dict, objective: str) -> list:
        """Score and rank candidates based on investment objective"""
        scored_items = []

        for symbol, data in fundamental_data.items():
            try:
                # Extract key metrics
                raw_data = data.get("raw_data", {})
                calculated_metrics = data.get("calculated_metrics", {})
                analysis = data.get("analysis", {})

                # Base score from overall fundamental score
                base_score = calculated_metrics.get("financial_health", {}).get(
                    "overall_score", 50
                )

                # Adjust score based on objective
                if objective == "growth":
                    score = self._calculate_growth_score(
                        raw_data, calculated_metrics, base_score
                    )
                elif objective == "value":
                    score = self._calculate_value_score(
                        raw_data, calculated_metrics, base_score
                    )
                else:  # balanced
                    score = self._calculate_balanced_score(
                        raw_data, calculated_metrics, base_score
                    )

                scored_items.append((symbol, score, data))

            except Exception as e:
                print(f"Error scoring {symbol}: {e}")
                # Default neutral score if error
                scored_items.append((symbol, 50, data))

        # Sort by score (highest first)
        return sorted(scored_items, key=lambda x: x[1], reverse=True)

    def _calculate_growth_score(
        self, raw_data: dict, calculated_metrics: dict, base_score: float
    ) -> float:
        """Calculate score for growth strategy"""
        score = base_score

        # Growth metrics bonus
        growth_data = raw_data.get("growth", {})
        if growth_data:
            revenue_growth = growth_data.get("revenue_growth", 0) or 0
            earnings_growth = growth_data.get("earnings_growth", 0) or 0

            if revenue_growth > 0.1:  # 10% revenue growth
                score += 15
            if earnings_growth > 0.15:  # 15% earnings growth
                score += 15

        # Profitability bonus for growth
        profitability = raw_data.get("profitability", {})
        if profitability:
            roe = profitability.get("return_on_equity", 0) or 0
            profit_margin = profitability.get("profit_margin", 0) or 0

            if roe > 0.15:  # ROE > 15%
                score += 10
            if profit_margin > 0.15:  # Profit margin > 15%
                score += 10

        # Penalize high debt for growth stocks
        leverage = raw_data.get("leverage", {})
        if leverage:
            debt_to_equity = leverage.get("debt_to_equity", 0) or 0
            if debt_to_equity > 0.6:
                score -= 10

        return min(100, max(0, score))

    def _calculate_value_score(
        self, raw_data: dict, calculated_metrics: dict, base_score: float
    ) -> float:
        """Calculate score for value strategy"""
        score = base_score

        # Valuation metrics bonus
        valuation = raw_data.get("valuation", {})
        if valuation:
            pe_ratio = valuation.get("trailing_pe")
            pb_ratio = valuation.get("price_to_book")

            if pe_ratio and pe_ratio < 15:  # Low P/E
                score += 15
            if pb_ratio and pb_ratio < 2.0:  # Low P/B
                score += 15

        # Dividend and value metrics
        profitability = raw_data.get("profitability", {})
        if profitability:
            profit_margin = profitability.get("profit_margin", 0) or 0
            if profit_margin > 0.1:  # Consistent profitability
                score += 10

        # Financial strength for value
        leverage = raw_data.get("leverage", {})
        liquidity = raw_data.get("liquidity", {})

        if leverage:
            debt_to_equity = leverage.get("debt_to_equity", 0) or 0
            if debt_to_equity < 0.4:  # Low debt
                score += 10

        if liquidity:
            current_ratio = liquidity.get("current_ratio", 0) or 0
            if current_ratio > 1.5:  # Good liquidity
                score += 10

        return min(100, max(0, score))

    def _calculate_balanced_score(
        self, raw_data: dict, calculated_metrics: dict, base_score: float
    ) -> float:
        """Calculate score for balanced strategy"""
        # Balanced approach: combine growth and value with equal weights
        growth_score = self._calculate_growth_score(
            raw_data, calculated_metrics, base_score
        )
        value_score = self._calculate_value_score(
            raw_data, calculated_metrics, base_score
        )

        # 60% base score, 20% growth factors, 20% value factors
        balanced_score = (
            base_score * 0.6
            + (growth_score - base_score) * 0.2
            + (value_score - base_score) * 0.2
        )

        return min(100, max(0, balanced_score))

    def _generate_detailed_rationale(
        self, symbol: str, ticker_data: dict, objective: str
    ) -> str:
        """Generate detailed rationale based on fundamental analysis"""
        try:
            raw_data = ticker_data.get("raw_data", {})
            analysis = ticker_data.get("analysis", {})
            calculated_metrics = ticker_data.get("calculated_metrics", {})

            company_info = raw_data.get("company_info", {})
            valuation = raw_data.get("valuation", {})
            profitability = raw_data.get("profitability", {})
            financial_health = calculated_metrics.get("financial_health", {})

            # Base information
            industry = company_info.get("industry", "N/A")
            market_cap = company_info.get("market_cap", 0)
            overall_score = financial_health.get("overall_score", 50)
            recommendation = analysis.get("recommendation", "Hold")

            # Key metrics
            pe_ratio = valuation.get("trailing_pe")
            roe = profitability.get("return_on_equity")
            profit_margin = profitability.get("profit_margin")

            # Build rationale based on objective
            if objective == "growth":
                growth_data = raw_data.get("growth", {})
                revenue_growth = growth_data.get("revenue_growth", 0) or 0

                rationale = f"Growth selection: {industry} sector, ${market_cap/1e9:.1f}B market cap. "
                rationale += f"Fundamental score: {overall_score:.0f}/100. "

                if revenue_growth > 0:
                    rationale += f"Revenue growth: {revenue_growth*100:.1f}%. "
                if roe:
                    rationale += f"ROE: {roe*100:.1f}%. "

                rationale += f"Recommendation: {recommendation}."

            elif objective == "value":
                rationale = f"Value selection: {industry} sector, attractive valuation metrics. "
                rationale += f"Fundamental score: {overall_score:.0f}/100. "

                if pe_ratio:
                    rationale += f"P/E: {pe_ratio:.1f}. "
                if profit_margin:
                    rationale += f"Profit margin: {profit_margin*100:.1f}%. "

                rationale += f"Recommendation: {recommendation}."

            else:  # balanced
                rationale = f"Balanced selection: {industry}, ${market_cap/1e9:.1f}B market cap. "
                rationale += f"Strong fundamentals (score: {overall_score:.0f}/100). "

                if pe_ratio and roe:
                    rationale += f"P/E: {pe_ratio:.1f}, ROE: {roe*100:.1f}%. "

                rationale += f"Recommendation: {recommendation}."

            return rationale

        except Exception as e:
            return f"Selected for {objective} strategy based on fundamental analysis."

    def _basic_selection_fallback(
        self, selected_tickers: list, objective: str
    ) -> tuple:
        """Fallback selection method using basic yfinance data"""
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

                    # Generate basic rationale
                    pe_ratio = info.get("trailingPE", "N/A")
                    market_cap = info.get("marketCap", 0)

                    if objective == "growth":
                        rationale[ticker] = (
                            f"Growth potential. Industry: {info.get('industry', 'N/A')}, "
                            f"Market Cap: ${market_cap/1e9:.1f}B"
                        )
                    elif objective == "value":
                        rationale[ticker] = (
                            f"Value opportunity. P/E: {pe_ratio}, "
                            f"Industry: {info.get('industry', 'N/A')}"
                        )
                    else:
                        rationale[ticker] = (
                            f"Balanced pick. P/E: {pe_ratio}, "
                            f"Market Cap: ${market_cap/1e9:.1f}B"
                        )

                return items, rationale
            else:
                # Final fallback
                items = [TickerItem(symbol=s) for s in selected_tickers]
                rationale = {
                    s: f"Selected based on {objective} objective"
                    for s in selected_tickers
                }
                return items, rationale

        except Exception as e:
            # Final fallback
            items = [TickerItem(symbol=s) for s in selected_tickers]
            rationale = {
                s: f"Selected based on {objective} objective (basic data unavailable)"
                for s in selected_tickers
            }
            return items, rationale
