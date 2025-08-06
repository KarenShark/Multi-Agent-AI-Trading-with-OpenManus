import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
import yfinance as yf

from app.tool.base import BaseTool


class FundamentalFetcher(BaseTool):
    """
    Enhanced fundamental analysis data fetcher
    Combines multiple data sources for comprehensive financial metrics
    """

    name: str = "fundamental_fetcher"
    description: str = (
        "Fetch fundamental financial data including P/E, ROE, debt ratios, and financial statements"
    )

    parameters: dict = {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock symbols to analyze",
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of fundamental metrics to fetch",
                "default": [
                    "valuation",
                    "profitability",
                    "liquidity",
                    "leverage",
                    "efficiency",
                    "growth",
                    "financial_statements",
                ],
            },
            "period": {
                "type": "string",
                "description": "Time period for historical data (quarterly, annual)",
                "default": "annual",
            },
        },
        "required": ["symbols"],
    }

    def __init__(self):
        super().__init__()
        # Alpha Vantage API key (optional, fallback to yfinance if not available)
        self._alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", None)
        self._alpha_vantage_base = "https://www.alphavantage.co/query"

    def execute(
        self,
        symbols: List[str],
        metrics: List[str] = None,
        period: str = "annual",
    ) -> Dict:
        """
        Fetch fundamental data for given symbols

        Args:
            symbols: List of stock symbols
            metrics: List of metric categories to fetch
            period: Data period (annual/quarterly)

        Returns:
            Dictionary containing fundamental analysis data
        """
        if metrics is None:
            metrics = [
                "valuation",
                "profitability",
                "liquidity",
                "leverage",
                "efficiency",
                "growth",
            ]

        result = {}

        try:
            for symbol in symbols:
                print(f"Fetching fundamental data for {symbol}...")

                # Get comprehensive fundamental data
                fundamental_data = self._get_comprehensive_fundamentals(
                    symbol, metrics, period
                )

                # Calculate derived metrics and ratios
                enhanced_metrics = self._calculate_enhanced_metrics(
                    fundamental_data, symbol
                )

                # Perform fundamental analysis
                analysis = self._perform_fundamental_analysis(enhanced_metrics, symbol)

                result[symbol] = {
                    "raw_data": fundamental_data,
                    "calculated_metrics": enhanced_metrics,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat(),
                    "data_quality": self._assess_data_quality(fundamental_data),
                }

        except Exception as e:
            return {
                "error": f"Failed to fetch fundamental data: {str(e)}",
                "success": False,
            }

        return {
            "data": result,
            "success": True,
            "metadata": {
                "metrics_included": metrics,
                "period": period,
                "data_sources": self._get_data_sources_info(),
            },
        }

    def _get_comprehensive_fundamentals(
        self, symbol: str, metrics: List[str], period: str
    ) -> Dict:
        """Get comprehensive fundamental data from multiple sources"""
        data = {}

        try:
            # Primary source: yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Basic company information
            data["company_info"] = {
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "country": info.get("country", ""),
                "currency": info.get("currency", "USD"),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "float_shares": info.get("floatShares", 0),
            }

            # Valuation metrics
            if "valuation" in metrics:
                data["valuation"] = {
                    "current_price": info.get("currentPrice", 0),
                    "trailing_pe": info.get("trailingPE", None),
                    "forward_pe": info.get("forwardPE", None),
                    "price_to_book": info.get("priceToBook", None),
                    "price_to_sales": info.get("priceToSalesTrailing12Months", None),
                    "peg_ratio": info.get("pegRatio", None),
                    "price_to_earnings_growth": info.get(
                        "priceEarningsToGrowthRatio", None
                    ),
                    "enterprise_to_revenue": info.get("enterpriseToRevenue", None),
                    "enterprise_to_ebitda": info.get("enterpriseToEbitda", None),
                }

            # Profitability metrics
            if "profitability" in metrics:
                data["profitability"] = {
                    "return_on_equity": info.get("returnOnEquity", None),
                    "return_on_assets": info.get("returnOnAssets", None),
                    "profit_margin": info.get("profitMargins", None),
                    "operating_margin": info.get("operatingMargins", None),
                    "gross_margin": info.get("grossMargins", None),
                    "ebitda_margin": info.get("ebitdaMargins", None),
                    "net_income": info.get("netIncomeToCommon", None),
                    "revenue": info.get("totalRevenue", None),
                    "gross_profit": info.get("grossProfits", None),
                    "operating_income": info.get("operatingIncome", None),
                    "ebitda": info.get("ebitda", None),
                }

            # Liquidity metrics
            if "liquidity" in metrics:
                data["liquidity"] = {
                    "current_ratio": info.get("currentRatio", None),
                    "quick_ratio": info.get("quickRatio", None),
                    "cash_ratio": self._calculate_cash_ratio(info),
                    "total_cash": info.get("totalCash", None),
                    "cash_per_share": info.get("totalCashPerShare", None),
                    "working_capital": self._calculate_working_capital(info),
                    "free_cash_flow": info.get("freeCashflow", None),
                    "operating_cash_flow": info.get("operatingCashflow", None),
                }

            # Leverage metrics
            if "leverage" in metrics:
                data["leverage"] = {
                    "debt_to_equity": info.get("debtToEquity", None),
                    "total_debt": info.get("totalDebt", None),
                    "long_term_debt": info.get("longTermDebt", None),
                    "short_term_debt": self._calculate_short_term_debt(info),
                    "debt_to_total_capital": self._calculate_debt_to_capital(info),
                    "interest_coverage": self._calculate_interest_coverage(info),
                    "debt_to_ebitda": self._calculate_debt_to_ebitda(info),
                }

            # Efficiency metrics
            if "efficiency" in metrics:
                data["efficiency"] = {
                    "asset_turnover": self._calculate_asset_turnover(info),
                    "inventory_turnover": self._calculate_inventory_turnover(info),
                    "receivables_turnover": self._calculate_receivables_turnover(info),
                    "total_asset_turnover": info.get("totalAssetTurnover", None),
                    "revenue_per_employee": self._calculate_revenue_per_employee(info),
                    "working_capital_turnover": self._calculate_working_capital_turnover(
                        info
                    ),
                }

            # Growth metrics
            if "growth" in metrics:
                data["growth"] = {
                    "revenue_growth": info.get("revenueGrowth", None),
                    "earnings_growth": info.get("earningsGrowth", None),
                    "earnings_quarterly_growth": info.get(
                        "earningsQuarterlyGrowth", None
                    ),
                    "revenue_quarterly_growth": info.get(
                        "revenueQuarterlyGrowth", None
                    ),
                    "book_value_per_share": info.get("bookValue", None),
                    "tangible_book_value": self._calculate_tangible_book_value(info),
                    "retained_earnings": info.get("retainedEarnings", None),
                }

            # Financial statements summary
            if "financial_statements" in metrics:
                try:
                    # Get financial statements
                    income_stmt = ticker.income_stmt
                    balance_sheet = ticker.balance_sheet
                    cash_flow = ticker.cashflow

                    data["financial_statements"] = {
                        "income_statement": self._summarize_income_statement(
                            income_stmt
                        ),
                        "balance_sheet": self._summarize_balance_sheet(balance_sheet),
                        "cash_flow": self._summarize_cash_flow(cash_flow),
                    }
                except:
                    data["financial_statements"] = {
                        "error": "Unable to fetch detailed financial statements"
                    }

            # Try to enhance with Alpha Vantage data if available
            if self._alpha_vantage_key:
                alpha_data = self._fetch_alpha_vantage_data(symbol)
                if alpha_data:
                    data["alpha_vantage"] = alpha_data

        except Exception as e:
            print(f"Error fetching fundamental data for {symbol}: {e}")
            data["error"] = str(e)

        return data

    def _calculate_enhanced_metrics(self, fundamental_data: Dict, symbol: str) -> Dict:
        """Calculate additional derived metrics"""
        enhanced = {}

        try:
            valuation = fundamental_data.get("valuation", {})
            profitability = fundamental_data.get("profitability", {})
            company_info = fundamental_data.get("company_info", {})

            # Calculate additional valuation metrics
            enhanced["valuation_scores"] = {
                "pe_relative_to_industry": self._get_industry_pe_comparison(
                    symbol, valuation.get("trailing_pe")
                ),
                "pb_relative_to_industry": self._get_industry_pb_comparison(
                    symbol, valuation.get("price_to_book")
                ),
                "valuation_grade": self._calculate_valuation_grade(valuation),
                "fair_value_estimate": self._estimate_fair_value(fundamental_data),
            }

            # Calculate financial health score
            enhanced["financial_health"] = {
                "overall_score": self._calculate_financial_health_score(
                    fundamental_data
                ),
                "liquidity_score": self._calculate_liquidity_score(
                    fundamental_data.get("liquidity", {})
                ),
                "leverage_score": self._calculate_leverage_score(
                    fundamental_data.get("leverage", {})
                ),
                "profitability_score": self._calculate_profitability_score(
                    profitability
                ),
                "efficiency_score": self._calculate_efficiency_score(
                    fundamental_data.get("efficiency", {})
                ),
            }

            # Growth quality assessment
            enhanced["growth_quality"] = {
                "sustainable_growth_rate": self._calculate_sustainable_growth_rate(
                    fundamental_data
                ),
                "growth_consistency": self._assess_growth_consistency(
                    fundamental_data.get("growth", {})
                ),
                "quality_of_earnings": self._assess_earnings_quality(fundamental_data),
            }

            # Risk metrics
            enhanced["risk_metrics"] = {
                "financial_leverage_risk": self._assess_leverage_risk(
                    fundamental_data.get("leverage", {})
                ),
                "liquidity_risk": self._assess_liquidity_risk(
                    fundamental_data.get("liquidity", {})
                ),
                "operational_risk": self._assess_operational_risk(fundamental_data),
                "market_risk_indicators": self._get_market_risk_indicators(
                    fundamental_data
                ),
            }

        except Exception as e:
            enhanced["calculation_error"] = str(e)

        return enhanced

    def _perform_fundamental_analysis(
        self, enhanced_metrics: Dict, symbol: str
    ) -> Dict:
        """Perform comprehensive fundamental analysis"""
        analysis = {}

        try:
            financial_health = enhanced_metrics.get("financial_health", {})
            valuation_scores = enhanced_metrics.get("valuation_scores", {})
            growth_quality = enhanced_metrics.get("growth_quality", {})
            risk_metrics = enhanced_metrics.get("risk_metrics", {})

            # Overall fundamental score (0-100)
            overall_score = self._calculate_overall_fundamental_score(enhanced_metrics)

            # Investment recommendation
            recommendation = self._generate_investment_recommendation(
                overall_score, enhanced_metrics
            )

            # Key strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(
                enhanced_metrics
            )

            # Peer comparison (simplified)
            peer_comparison = self._get_simplified_peer_comparison(
                symbol, enhanced_metrics
            )

            analysis = {
                "overall_score": overall_score,
                "recommendation": recommendation,
                "key_strengths": strengths,
                "key_weaknesses": weaknesses,
                "peer_comparison": peer_comparison,
                "investment_thesis": self._generate_investment_thesis(
                    enhanced_metrics, symbol
                ),
                "risk_assessment": self._generate_risk_assessment(risk_metrics),
                "price_target_range": self._estimate_price_target_range(
                    enhanced_metrics
                ),
            }

        except Exception as e:
            analysis["analysis_error"] = str(e)

        return analysis

    # Helper calculation methods
    def _calculate_cash_ratio(self, info: Dict) -> Optional[float]:
        """Calculate cash ratio"""
        cash = info.get("totalCash", 0)
        current_liabilities = info.get("totalCurrentLiabilities", 0)
        if current_liabilities and current_liabilities > 0:
            return cash / current_liabilities
        return None

    def _calculate_working_capital(self, info: Dict) -> Optional[float]:
        """Calculate working capital"""
        current_assets = info.get("totalCurrentAssets", 0)
        current_liabilities = info.get("totalCurrentLiabilities", 0)
        if current_assets is not None and current_liabilities is not None:
            return current_assets - current_liabilities
        return None

    def _calculate_short_term_debt(self, info: Dict) -> Optional[float]:
        """Calculate short-term debt"""
        total_debt = info.get("totalDebt", 0)
        long_term_debt = info.get("longTermDebt", 0)
        if total_debt is not None and long_term_debt is not None:
            return total_debt - long_term_debt
        return None

    def _calculate_debt_to_capital(self, info: Dict) -> Optional[float]:
        """Calculate debt to total capital ratio"""
        total_debt = info.get("totalDebt", 0)
        total_equity = info.get("totalStockholderEquity", 0)
        if (
            total_debt is not None
            and total_equity is not None
            and (total_debt + total_equity) > 0
        ):
            return total_debt / (total_debt + total_equity)
        return None

    def _calculate_interest_coverage(self, info: Dict) -> Optional[float]:
        """Calculate interest coverage ratio"""
        ebit = info.get("ebitda", 0)  # Approximation
        interest_expense = info.get("interestExpense", 0)
        if ebit and interest_expense and interest_expense > 0:
            return ebit / interest_expense
        return None

    def _calculate_debt_to_ebitda(self, info: Dict) -> Optional[float]:
        """Calculate debt to EBITDA ratio"""
        total_debt = info.get("totalDebt", 0)
        ebitda = info.get("ebitda", 0)
        if total_debt is not None and ebitda and ebitda > 0:
            return total_debt / ebitda
        return None

    def _calculate_asset_turnover(self, info: Dict) -> Optional[float]:
        """Calculate asset turnover ratio"""
        revenue = info.get("totalRevenue", 0)
        total_assets = info.get("totalAssets", 0)
        if revenue and total_assets and total_assets > 0:
            return revenue / total_assets
        return None

    def _calculate_inventory_turnover(self, info: Dict) -> Optional[float]:
        """Calculate inventory turnover ratio"""
        cogs = info.get("costOfRevenue", 0)
        inventory = info.get("inventory", 0)
        if cogs and inventory and inventory > 0:
            return cogs / inventory
        return None

    def _calculate_receivables_turnover(self, info: Dict) -> Optional[float]:
        """Calculate receivables turnover ratio"""
        revenue = info.get("totalRevenue", 0)
        receivables = info.get("accountsReceivable", 0)
        if revenue and receivables and receivables > 0:
            return revenue / receivables
        return None

    def _calculate_revenue_per_employee(self, info: Dict) -> Optional[float]:
        """Calculate revenue per employee"""
        revenue = info.get("totalRevenue", 0)
        employees = info.get("fullTimeEmployees", 0)
        if revenue and employees and employees > 0:
            return revenue / employees
        return None

    def _calculate_working_capital_turnover(self, info: Dict) -> Optional[float]:
        """Calculate working capital turnover"""
        revenue = info.get("totalRevenue", 0)
        working_capital = self._calculate_working_capital(info)
        if revenue and working_capital and working_capital > 0:
            return revenue / working_capital
        return None

    def _calculate_tangible_book_value(self, info: Dict) -> Optional[float]:
        """Calculate tangible book value per share"""
        book_value = info.get("bookValue", 0)
        intangible_assets = info.get("intangibleAssets", 0) or 0
        shares = info.get("sharesOutstanding", 0)
        if book_value is not None and shares and shares > 0:
            return (book_value * shares - intangible_assets) / shares
        return None

    # Financial statement summarization methods
    def _summarize_income_statement(self, income_stmt) -> Dict:
        """Summarize income statement"""
        try:
            if income_stmt.empty:
                return {"error": "No income statement data available"}

            latest = income_stmt.iloc[:, 0]  # Most recent year
            return {
                "total_revenue": latest.get("Total Revenue", 0),
                "gross_profit": latest.get("Gross Profit", 0),
                "operating_income": latest.get("Operating Income", 0),
                "net_income": latest.get("Net Income", 0),
                "basic_eps": latest.get("Basic EPS", 0),
                "diluted_eps": latest.get("Diluted EPS", 0),
            }
        except:
            return {"error": "Unable to process income statement"}

    def _summarize_balance_sheet(self, balance_sheet) -> Dict:
        """Summarize balance sheet"""
        try:
            if balance_sheet.empty:
                return {"error": "No balance sheet data available"}

            latest = balance_sheet.iloc[:, 0]  # Most recent year
            return {
                "total_assets": latest.get("Total Assets", 0),
                "total_liabilities": latest.get(
                    "Total Liabilities Net Minority Interest", 0
                ),
                "total_equity": latest.get("Total Equity Gross Minority Interest", 0),
                "cash_and_equivalents": latest.get("Cash And Cash Equivalents", 0),
                "total_debt": latest.get("Total Debt", 0),
                "working_capital": latest.get("Working Capital", 0),
            }
        except:
            return {"error": "Unable to process balance sheet"}

    def _summarize_cash_flow(self, cash_flow) -> Dict:
        """Summarize cash flow statement"""
        try:
            if cash_flow.empty:
                return {"error": "No cash flow data available"}

            latest = cash_flow.iloc[:, 0]  # Most recent year
            return {
                "operating_cash_flow": latest.get("Operating Cash Flow", 0),
                "investing_cash_flow": latest.get("Investing Cash Flow", 0),
                "financing_cash_flow": latest.get("Financing Cash Flow", 0),
                "free_cash_flow": latest.get("Free Cash Flow", 0),
                "capital_expenditure": latest.get("Capital Expenditure", 0),
            }
        except:
            return {"error": "Unable to process cash flow statement"}

    # Scoring and analysis methods
    def _calculate_financial_health_score(self, fundamental_data: Dict) -> float:
        """Calculate overall financial health score (0-100)"""
        scores = []
        weights = []

        # Liquidity score (25% weight)
        liquidity_score = self._calculate_liquidity_score(
            fundamental_data.get("liquidity", {})
        )
        if liquidity_score is not None:
            scores.append(liquidity_score)
            weights.append(0.25)

        # Leverage score (25% weight)
        leverage_score = self._calculate_leverage_score(
            fundamental_data.get("leverage", {})
        )
        if leverage_score is not None:
            scores.append(leverage_score)
            weights.append(0.25)

        # Profitability score (30% weight)
        profitability_score = self._calculate_profitability_score(
            fundamental_data.get("profitability", {})
        )
        if profitability_score is not None:
            scores.append(profitability_score)
            weights.append(0.30)

        # Efficiency score (20% weight)
        efficiency_score = self._calculate_efficiency_score(
            fundamental_data.get("efficiency", {})
        )
        if efficiency_score is not None:
            scores.append(efficiency_score)
            weights.append(0.20)

        if scores:
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return min(100, max(0, weighted_score))

        return 50  # Neutral score if no data

    def _calculate_liquidity_score(self, liquidity_data: Dict) -> Optional[float]:
        """Calculate liquidity score (0-100)"""
        score = 50  # Start with neutral

        current_ratio = liquidity_data.get("current_ratio")
        if current_ratio:
            if current_ratio >= 2.0:
                score += 20
            elif current_ratio >= 1.5:
                score += 10
            elif current_ratio < 1.0:
                score -= 20

        quick_ratio = liquidity_data.get("quick_ratio")
        if quick_ratio:
            if quick_ratio >= 1.0:
                score += 15
            elif quick_ratio < 0.5:
                score -= 15

        free_cash_flow = liquidity_data.get("free_cash_flow")
        if free_cash_flow:
            if free_cash_flow > 0:
                score += 15
            else:
                score -= 15

        return min(100, max(0, score))

    def _calculate_leverage_score(self, leverage_data: Dict) -> Optional[float]:
        """Calculate leverage score (0-100)"""
        score = 50  # Start with neutral

        debt_to_equity = leverage_data.get("debt_to_equity")
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                score += 20
            elif debt_to_equity < 0.6:
                score += 10
            elif debt_to_equity > 1.0:
                score -= 20

        interest_coverage = leverage_data.get("interest_coverage")
        if interest_coverage:
            if interest_coverage > 5.0:
                score += 15
            elif interest_coverage > 2.5:
                score += 10
            elif interest_coverage < 1.5:
                score -= 20

        debt_to_ebitda = leverage_data.get("debt_to_ebitda")
        if debt_to_ebitda:
            if debt_to_ebitda < 2.0:
                score += 15
            elif debt_to_ebitda > 4.0:
                score -= 15

        return min(100, max(0, score))

    def _calculate_profitability_score(
        self, profitability_data: Dict
    ) -> Optional[float]:
        """Calculate profitability score (0-100)"""
        score = 50  # Start with neutral

        roe = profitability_data.get("return_on_equity")
        if roe:
            if roe > 0.15:
                score += 20
            elif roe > 0.10:
                score += 10
            elif roe < 0.05:
                score -= 15

        profit_margin = profitability_data.get("profit_margin")
        if profit_margin:
            if profit_margin > 0.15:
                score += 15
            elif profit_margin > 0.10:
                score += 10
            elif profit_margin < 0.05:
                score -= 10

        operating_margin = profitability_data.get("operating_margin")
        if operating_margin:
            if operating_margin > 0.20:
                score += 15
            elif operating_margin > 0.10:
                score += 10
            elif operating_margin < 0.05:
                score -= 10

        return min(100, max(0, score))

    def _calculate_efficiency_score(self, efficiency_data: Dict) -> Optional[float]:
        """Calculate efficiency score (0-100)"""
        score = 50  # Start with neutral

        asset_turnover = efficiency_data.get("asset_turnover")
        if asset_turnover:
            if asset_turnover > 1.0:
                score += 20
            elif asset_turnover > 0.5:
                score += 10
            elif asset_turnover < 0.3:
                score -= 10

        # Additional efficiency metrics can be added here

        return min(100, max(0, score))

    # Simplified analysis methods
    def _get_industry_pe_comparison(
        self, symbol: str, pe_ratio: Optional[float]
    ) -> str:
        """Get simplified industry PE comparison"""
        if not pe_ratio:
            return "Insufficient data"

        # Simplified industry averages (in reality, would use external data)
        industry_pe_map = {
            "Technology": 25,
            "Healthcare": 20,
            "Financial": 12,
            "Energy": 15,
            "Consumer": 18,
        }

        # Default industry PE
        avg_industry_pe = 18

        if pe_ratio < avg_industry_pe * 0.8:
            return "Below industry average (potentially undervalued)"
        elif pe_ratio > avg_industry_pe * 1.2:
            return "Above industry average (potentially overvalued)"
        else:
            return "Near industry average"

    def _get_industry_pb_comparison(
        self, symbol: str, pb_ratio: Optional[float]
    ) -> str:
        """Get simplified industry PB comparison"""
        if not pb_ratio:
            return "Insufficient data"

        if pb_ratio < 1.0:
            return "Trading below book value"
        elif pb_ratio < 2.0:
            return "Reasonable book value multiple"
        else:
            return "High book value multiple"

    def _calculate_valuation_grade(self, valuation_data: Dict) -> str:
        """Calculate simplified valuation grade"""
        score = 0
        factors = 0

        pe = valuation_data.get("trailing_pe")
        if pe and 10 <= pe <= 20:
            score += 1
        factors += 1

        pb = valuation_data.get("price_to_book")
        if pb and 1.0 <= pb <= 3.0:
            score += 1
        factors += 1

        ps = valuation_data.get("price_to_sales")
        if ps and 1.0 <= ps <= 5.0:
            score += 1
        factors += 1

        if factors == 0:
            return "Insufficient data"

        grade_pct = score / factors
        if grade_pct >= 0.8:
            return "A (Attractive)"
        elif grade_pct >= 0.6:
            return "B (Fair)"
        elif grade_pct >= 0.4:
            return "C (Neutral)"
        else:
            return "D (Expensive)"

    def _estimate_fair_value(self, fundamental_data: Dict) -> Dict:
        """Estimate fair value range"""
        try:
            profitability = fundamental_data.get("profitability", {})
            valuation = fundamental_data.get("valuation", {})

            current_price = valuation.get("current_price", 0)
            if not current_price:
                return {"error": "No current price data"}

            # Simple DCF-based estimate
            roe = profitability.get("return_on_equity", 0.1)
            growth_rate = min(
                0.15, max(0.02, roe * 0.6)
            )  # Conservative growth estimate

            # Fair value range based on P/E multiples
            earnings_per_share = (
                current_price / valuation.get("trailing_pe", 15)
                if valuation.get("trailing_pe")
                else current_price / 15
            )

            conservative_pe = 12
            optimistic_pe = 20

            return {
                "fair_value_low": earnings_per_share * conservative_pe,
                "fair_value_high": earnings_per_share * optimistic_pe,
                "current_price": current_price,
                "implied_upside_low": (
                    earnings_per_share * conservative_pe / current_price - 1
                )
                * 100,
                "implied_upside_high": (
                    earnings_per_share * optimistic_pe / current_price - 1
                )
                * 100,
            }
        except:
            return {"error": "Unable to calculate fair value"}

    def _calculate_overall_fundamental_score(self, enhanced_metrics: Dict) -> float:
        """Calculate overall fundamental score"""
        financial_health = enhanced_metrics.get("financial_health", {})
        return financial_health.get("overall_score", 50)

    def _generate_investment_recommendation(
        self, overall_score: float, enhanced_metrics: Dict
    ) -> str:
        """Generate investment recommendation"""
        if overall_score >= 80:
            return "Strong Buy"
        elif overall_score >= 65:
            return "Buy"
        elif overall_score >= 50:
            return "Hold"
        elif overall_score >= 35:
            return "Weak Hold"
        else:
            return "Sell"

    def _identify_strengths_weaknesses(self, enhanced_metrics: Dict) -> tuple:
        """Identify key strengths and weaknesses"""
        strengths = []
        weaknesses = []

        financial_health = enhanced_metrics.get("financial_health", {})

        if financial_health.get("liquidity_score", 0) >= 70:
            strengths.append("Strong liquidity position")
        elif financial_health.get("liquidity_score", 0) <= 30:
            weaknesses.append("Weak liquidity position")

        if financial_health.get("leverage_score", 0) >= 70:
            strengths.append("Conservative debt levels")
        elif financial_health.get("leverage_score", 0) <= 30:
            weaknesses.append("High debt burden")

        if financial_health.get("profitability_score", 0) >= 70:
            strengths.append("Strong profitability metrics")
        elif financial_health.get("profitability_score", 0) <= 30:
            weaknesses.append("Weak profitability")

        return strengths or ["No significant strengths identified"], weaknesses or [
            "No significant weaknesses identified"
        ]

    def _get_simplified_peer_comparison(
        self, symbol: str, enhanced_metrics: Dict
    ) -> Dict:
        """Get simplified peer comparison"""
        return {
            "note": "Peer comparison requires external data source",
            "recommendation": "Compare with industry ETF performance",
        }

    def _generate_investment_thesis(self, enhanced_metrics: Dict, symbol: str) -> str:
        """Generate investment thesis"""
        financial_health = enhanced_metrics.get("financial_health", {})
        overall_score = financial_health.get("overall_score", 50)

        if overall_score >= 70:
            return f"{symbol} demonstrates strong fundamentals with solid financial health metrics, making it an attractive investment opportunity."
        elif overall_score >= 50:
            return f"{symbol} shows mixed fundamentals with some areas of strength and concern, suitable for balanced portfolios."
        else:
            return f"{symbol} exhibits weaker fundamentals with several areas of concern, requiring careful consideration before investment."

    def _generate_risk_assessment(self, risk_metrics: Dict) -> str:
        """Generate risk assessment"""
        return "Moderate risk profile based on fundamental analysis. Monitor debt levels and cash flow trends."

    def _estimate_price_target_range(self, enhanced_metrics: Dict) -> Dict:
        """Estimate price target range"""
        valuation_scores = enhanced_metrics.get("valuation_scores", {})
        fair_value = valuation_scores.get("fair_value_estimate", {})

        if "fair_value_low" in fair_value and "fair_value_high" in fair_value:
            return {
                "target_low": fair_value["fair_value_low"],
                "target_high": fair_value["fair_value_high"],
                "time_horizon": "12 months",
            }

        return {"error": "Unable to estimate price targets"}

    # Additional helper methods
    def _calculate_sustainable_growth_rate(
        self, fundamental_data: Dict
    ) -> Optional[float]:
        """Calculate sustainable growth rate"""
        profitability = fundamental_data.get("profitability", {})
        roe = profitability.get("return_on_equity")

        if roe:
            # Assuming 60% retention ratio
            return roe * 0.6
        return None

    def _assess_growth_consistency(self, growth_data: Dict) -> str:
        """Assess growth consistency"""
        revenue_growth = growth_data.get("revenue_growth")
        earnings_growth = growth_data.get("earnings_growth")

        if revenue_growth and earnings_growth:
            if revenue_growth > 0 and earnings_growth > 0:
                return "Positive growth trend"
            else:
                return "Declining growth"
        return "Insufficient data"

    def _assess_earnings_quality(self, fundamental_data: Dict) -> str:
        """Assess earnings quality"""
        liquidity = fundamental_data.get("liquidity", {})
        free_cash_flow = liquidity.get("free_cash_flow", 0)

        profitability = fundamental_data.get("profitability", {})
        net_income = profitability.get("net_income", 0)

        if free_cash_flow and net_income and net_income > 0:
            if free_cash_flow / net_income > 0.8:
                return "High quality earnings"
            else:
                return "Lower quality earnings"
        return "Insufficient data"

    def _assess_leverage_risk(self, leverage_data: Dict) -> str:
        """Assess leverage risk"""
        debt_to_equity = leverage_data.get("debt_to_equity", 0)

        if debt_to_equity is None:
            return "Insufficient data"
        elif debt_to_equity < 0.3:
            return "Low leverage risk"
        elif debt_to_equity < 0.6:
            return "Moderate leverage risk"
        else:
            return "High leverage risk"

    def _assess_liquidity_risk(self, liquidity_data: Dict) -> str:
        """Assess liquidity risk"""
        current_ratio = liquidity_data.get("current_ratio", 0)

        if current_ratio is None:
            return "Insufficient data"
        elif current_ratio >= 2.0:
            return "Low liquidity risk"
        elif current_ratio >= 1.0:
            return "Moderate liquidity risk"
        else:
            return "High liquidity risk"

    def _assess_operational_risk(self, fundamental_data: Dict) -> str:
        """Assess operational risk"""
        profitability = fundamental_data.get("profitability", {})
        operating_margin = profitability.get("operating_margin", 0)

        if operating_margin is None:
            return "Insufficient data"
        elif operating_margin >= 0.15:
            return "Low operational risk"
        elif operating_margin >= 0.05:
            return "Moderate operational risk"
        else:
            return "High operational risk"

    def _get_market_risk_indicators(self, fundamental_data: Dict) -> Dict:
        """Get market risk indicators"""
        valuation = fundamental_data.get("valuation", {})
        return {
            "pe_ratio": valuation.get("trailing_pe"),
            "pb_ratio": valuation.get("price_to_book"),
            "market_cap": fundamental_data.get("company_info", {}).get("market_cap"),
            "beta_proxy": "Market correlation analysis required",
        }

    def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[Dict]:
        """Fetch additional data from Alpha Vantage if API key is available"""
        if not self._alpha_vantage_key:
            return None

        try:
            # Example: Fetch company overview
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self._alpha_vantage_key,
            }

            response = requests.get(self._alpha_vantage_base, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass

        return None

    def _assess_data_quality(self, fundamental_data: Dict) -> Dict:
        """Assess the quality of fundamental data"""
        quality_score = 0
        total_metrics = 0

        # Check completeness of key metrics
        key_sections = ["valuation", "profitability", "liquidity", "leverage"]

        for section in key_sections:
            section_data = fundamental_data.get(section, {})
            if section_data and not section_data.get("error"):
                non_null_metrics = sum(
                    1 for v in section_data.values() if v is not None and v != 0
                )
                total_metrics_in_section = len(section_data)
                if total_metrics_in_section > 0:
                    quality_score += non_null_metrics / total_metrics_in_section
                    total_metrics += 1

        overall_quality = (
            (quality_score / max(1, total_metrics)) if total_metrics > 0 else 0
        )

        return {
            "quality_score": overall_quality,
            "completeness": overall_quality * 100,
            "assessment": (
                "High"
                if overall_quality > 0.8
                else "Medium" if overall_quality > 0.5 else "Low"
            ),
        }

    def _get_data_sources_info(self) -> Dict:
        """Get information about data sources used"""
        sources = ["Yahoo Finance (yfinance)"]
        if self._alpha_vantage_key:
            sources.append("Alpha Vantage")

        return {
            "primary_source": "Yahoo Finance",
            "additional_sources": sources[1:] if len(sources) > 1 else [],
            "data_reliability": "High for major stocks, may vary for smaller companies",
        }
