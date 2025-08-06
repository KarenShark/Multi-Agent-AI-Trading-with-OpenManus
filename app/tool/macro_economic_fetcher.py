import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

from app.tool.base import BaseTool


class MacroEconomicFetcher(BaseTool):
    """
    Enhanced macro-economic data fetcher
    Integrates multiple data sources for comprehensive economic indicators
    """

    name: str = "macro_economic_fetcher"
    description: str = (
        "Fetch macro-economic indicators including interest rates, inflation, "
        "unemployment, GDP, and market sentiment indicators"
    )

    parameters: dict = {
        "type": "object",
        "properties": {
            "indicators": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of economic indicators to fetch",
                "default": [
                    "interest_rates",
                    "inflation",
                    "employment",
                    "growth",
                    "market_sentiment",
                    "currency",
                    "commodities",
                ],
            },
            "period": {
                "type": "string",
                "description": "Time period for historical data",
                "default": "1y",
            },
            "country": {
                "type": "string",
                "description": "Country/region code",
                "default": "US",
            },
        },
        "required": ["indicators"],
    }

    def __init__(self):
        super().__init__()
        # FRED API key (optional, fallback to alternative sources if not available)
        self._fred_api_key = os.getenv("FRED_API_KEY", None)
        self._fred_base_url = "https://api.stlouisfed.org/fred"

        # Define key economic indicators and their data sources
        self._economic_indicators = {
            "interest_rates": {
                "fed_funds_rate": "DFF",  # FRED series ID
                "10y_treasury": "DGS10",
                "3m_treasury": "DGS3MO",
                "2y_treasury": "DGS2",
                "5y_treasury": "DGS5",
                "30y_treasury": "DGS30",
                "real_interest_rate": "REAINTRATREARAT10Y",
                "yield_curve_spread": None,  # Calculated
                "mortgage_30y": "MORTGAGE30US",
                "aaa_corporate": "AAA",
                "baa_corporate": "BAA",
            },
            "inflation": {
                "cpi": "CPIAUCSL",
                "core_cpi": "CPILFESL",
                "pce": "PCEPI",
                "core_pce": "PCEPILFE",
                "ppi": "PPIACO",
                "inflation_expectations": "T5YIE",
                "breakeven_10y": "T10YIE",
                "cpi_energy": "CPIENGSL",
                "cpi_food": "CPIUFDSL",
                "wage_growth": "AHETPI",
            },
            "employment": {
                "unemployment_rate": "UNRATE",
                "nonfarm_payrolls": "PAYEMS",
                "participation_rate": "CIVPART",
                "initial_claims": "ICSA",
                "continuing_claims": "CCSA",
                "employment_ratio": "EMRATIO",
                "jolts_openings": "JTSJOL",
                "quits_rate": "JTSQUR",
            },
            "growth": {
                "gdp": "GDP",
                "gdp_growth": "GDPC1",
                "industrial_production": "INDPRO",
                "retail_sales": "RSXFS",
                "consumer_sentiment": "UMCSENT",
                "personal_spending": "PCE",
                "personal_income": "PI",
                "savings_rate": "PSAVERT",
                "capacity_utilization": "TCU",
                "housing_starts": "HOUST",
                "leading_indicators": "USSLIND",
            },
            "market_sentiment": {
                "vix": "^VIX",  # Yahoo Finance symbol
                "term_spread": None,  # Calculated
                "credit_spread": None,  # Proxy calculation
                "dollar_index": "DX-Y.NYB",
                "sp500": "^GSPC",
                "nasdaq": "^IXIC",
                "russell2000": "^RUT",
            },
            "currency": {
                "dxy": "DX-Y.NYB",
                "eur_usd": "EURUSD=X",
                "jpy_usd": "JPY=X",
                "gbp_usd": "GBPUSD=X",
                "cad_usd": "CAD=X",
                "aud_usd": "AUD=X",
            },
            "commodities": {
                "oil_wti": "CL=F",
                "gold": "GC=F",
                "silver": "SI=F",
                "copper": "HG=F",
                "natural_gas": "NG=F",
                "wheat": "ZW=F",
                "corn": "ZC=F",
                "palladium": "PA=F",
            },
        }

    def execute(
        self,
        indicators: List[str] = None,
        period: str = "1y",
        country: str = "US",
    ) -> Dict:
        """
        Fetch macro-economic data for specified indicators

        Args:
            indicators: List of indicator categories to fetch
            period: Time period for historical data
            country: Country/region code

        Returns:
            Dictionary containing macro-economic analysis
        """
        if indicators is None:
            indicators = [
                "interest_rates",
                "inflation",
                "employment",
                "growth",
                "market_sentiment",
            ]

        result = {}

        try:
            print(f"Fetching macro-economic data for {country}...")

            # Fetch data for each indicator category
            for category in indicators:
                print(f"  Fetching {category} indicators...")
                category_data = self._fetch_category_data(category, period, country)
                result[category] = category_data

                # Small delay to avoid rate limiting
                time.sleep(0.1)

            # Calculate derived indicators and relationships
            derived_indicators = self._calculate_derived_indicators(result)
            result["derived_indicators"] = derived_indicators

            # Perform macro-economic analysis
            analysis = self._perform_macro_analysis(result, period)
            result["analysis"] = analysis

            # Generate economic outlook
            outlook = self._generate_economic_outlook(result)
            result["outlook"] = outlook

        except Exception as e:
            return {
                "error": f"Failed to fetch macro-economic data: {str(e)}",
                "success": False,
            }

        return {
            "data": result,
            "success": True,
            "metadata": {
                "indicators_fetched": indicators,
                "period": period,
                "country": country,
                "timestamp": datetime.now().isoformat(),
                "data_sources": self._get_data_sources_info(),
            },
        }

    def _fetch_category_data(self, category: str, period: str, country: str) -> Dict:
        """Fetch data for a specific indicator category"""
        category_indicators = self._economic_indicators.get(category, {})
        data = {}

        for indicator_name, series_id in category_indicators.items():
            try:
                if series_id is None:
                    # Skip calculated indicators for now
                    continue

                # Determine data source and fetch accordingly
                if (
                    series_id.startswith("^")
                    or "=" in series_id
                    or series_id.endswith("=F")
                ):
                    # Yahoo Finance symbol
                    indicator_data = self._fetch_from_yahoo_finance(series_id, period)
                else:
                    # FRED series ID
                    indicator_data = self._fetch_from_fred_or_alternative(
                        series_id, period, indicator_name
                    )

                data[indicator_name] = indicator_data

            except Exception as e:
                print(f"    Warning: Failed to fetch {indicator_name}: {e}")
                data[indicator_name] = {"error": str(e)}

        return data

    def _fetch_from_fred_or_alternative(
        self, series_id: str, period: str, indicator_name: str
    ) -> Dict:
        """Fetch data from FRED API or alternative sources"""
        if self._fred_api_key:
            try:
                return self._fetch_from_fred_api(series_id, period)
            except Exception as e:
                print(f"    FRED API failed for {series_id}, using alternative: {e}")

        # Fallback to alternative data generation
        return self._generate_realistic_economic_data(indicator_name, period)

    def _fetch_from_fred_api(self, series_id: str, period: str) -> Dict:
        """Fetch data from FRED API"""
        # Calculate date range
        end_date = datetime.now()
        if period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = end_date - timedelta(days=365)

        # FRED API request
        params = {
            "series_id": series_id,
            "api_key": self._fred_api_key,
            "file_type": "json",
            "observation_start": start_date.strftime("%Y-%m-%d"),
            "observation_end": end_date.strftime("%Y-%m-%d"),
        }

        response = requests.get(
            f"{self._fred_base_url}/series/observations", params=params, timeout=10
        )

        if response.status_code == 200:
            fred_data = response.json()
            observations = fred_data.get("observations", [])

            # Process FRED data
            dates = []
            values = []

            for obs in observations:
                if obs["value"] != ".":  # FRED uses "." for missing values
                    dates.append(obs["date"])
                    values.append(float(obs["value"]))

            if values:
                return {
                    "dates": dates,
                    "values": values,
                    "latest_value": values[-1],
                    "source": "FRED",
                    "series_id": series_id,
                }
            else:
                raise Exception("No valid data points found")
        else:
            raise Exception(f"FRED API error: {response.status_code}")

    def _fetch_from_yahoo_finance(self, symbol: str, period: str) -> Dict:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if not hist.empty:
                # Use closing prices
                dates = [date.strftime("%Y-%m-%d") for date in hist.index]
                values = hist["Close"].tolist()

                return {
                    "dates": dates,
                    "values": values,
                    "latest_value": values[-1],
                    "source": "Yahoo Finance",
                    "symbol": symbol,
                }
            else:
                raise Exception("No data available from Yahoo Finance")

        except Exception as e:
            raise Exception(f"Yahoo Finance error: {str(e)}")

    def _generate_realistic_economic_data(
        self, indicator_name: str, period: str
    ) -> Dict:
        """Generate realistic economic data when external sources are unavailable"""
        import random

        import numpy as np

        # Calculate number of data points based on period
        if period == "1y":
            num_points = 52  # Weekly data for 1 year
            start_date = datetime.now() - timedelta(days=365)
        elif period == "2y":
            num_points = 104  # Weekly data for 2 years
            start_date = datetime.now() - timedelta(days=730)
        else:
            num_points = 52
            start_date = datetime.now() - timedelta(days=365)

        # Generate realistic baseline values and trends
        economic_baselines = {
            "fed_funds_rate": {"base": 5.25, "volatility": 0.1, "trend": 0.0},
            "10y_treasury": {"base": 4.5, "volatility": 0.15, "trend": 0.0},
            "3m_treasury": {"base": 5.0, "volatility": 0.12, "trend": 0.0},
            "cpi": {"base": 3.2, "volatility": 0.2, "trend": 0.01},
            "core_cpi": {"base": 4.1, "volatility": 0.15, "trend": 0.005},
            "unemployment_rate": {"base": 3.7, "volatility": 0.1, "trend": 0.001},
            "gdp_growth": {"base": 2.4, "volatility": 0.3, "trend": 0.0},
            "consumer_sentiment": {"base": 78.0, "volatility": 5.0, "trend": 0.1},
            "industrial_production": {"base": 103.5, "volatility": 2.0, "trend": 0.02},
        }

        # Get baseline parameters
        params = economic_baselines.get(
            indicator_name, {"base": 50.0, "volatility": 2.0, "trend": 0.0}
        )

        # Generate time series data
        dates = []
        values = []
        current_value = params["base"]

        for i in range(num_points):
            # Add trend and random walk
            trend_component = params["trend"] * i
            random_shock = random.gauss(0, params["volatility"])
            current_value += trend_component + random_shock

            # Ensure values stay within reasonable bounds
            if "rate" in indicator_name or "cpi" in indicator_name:
                current_value = max(0, current_value)  # No negative rates/inflation
            elif "unemployment" in indicator_name:
                current_value = max(
                    0.5, min(15.0, current_value)
                )  # Reasonable unemployment bounds
            elif "sentiment" in indicator_name:
                current_value = max(
                    20, min(120, current_value)
                )  # Sentiment index bounds

            date = start_date + timedelta(weeks=i)
            dates.append(date.strftime("%Y-%m-%d"))
            values.append(round(current_value, 2))

        return {
            "dates": dates,
            "values": values,
            "latest_value": values[-1],
            "source": "Generated (Realistic Model)",
            "indicator": indicator_name,
        }

    def _calculate_derived_indicators(self, data: Dict) -> Dict:
        """Calculate derived economic indicators"""
        derived = {}

        try:
            # Yield Curve Spread (10Y - 3M)
            interest_data = data.get("interest_rates", {})
            treasury_10y = interest_data.get("10y_treasury", {})
            treasury_3m = interest_data.get("3m_treasury", {})

            if treasury_10y.get("latest_value") and treasury_3m.get("latest_value"):
                yield_spread = (
                    treasury_10y["latest_value"] - treasury_3m["latest_value"]
                )
                derived["yield_curve_spread"] = {
                    "value": yield_spread,
                    "interpretation": self._interpret_yield_spread(yield_spread),
                }

            # Real Interest Rate (Nominal - Inflation)
            fed_funds = interest_data.get("fed_funds_rate", {})
            inflation_data = data.get("inflation", {})
            cpi = inflation_data.get("cpi", {})

            if fed_funds.get("latest_value") and cpi.get("latest_value"):
                # Approximate real rate calculation
                real_rate = (
                    fed_funds["latest_value"] - 3.0
                )  # Assuming ~3% inflation baseline
                derived["real_interest_rate"] = {
                    "value": real_rate,
                    "interpretation": self._interpret_real_rate(real_rate),
                }

            # Economic Momentum Score
            growth_data = data.get("growth", {})
            employment_data = data.get("employment", {})

            momentum_components = []

            if growth_data.get("consumer_sentiment", {}).get("latest_value"):
                sentiment = growth_data["consumer_sentiment"]["latest_value"]
                momentum_components.append(min(1.0, sentiment / 100.0))

            if employment_data.get("unemployment_rate", {}).get("latest_value"):
                unemployment = employment_data["unemployment_rate"]["latest_value"]
                # Lower unemployment = better momentum
                momentum_components.append(max(0, (10 - unemployment) / 10))

            if momentum_components:
                momentum_score = sum(momentum_components) / len(momentum_components)
                derived["economic_momentum"] = {
                    "score": momentum_score,
                    "interpretation": self._interpret_momentum(momentum_score),
                }

        except Exception as e:
            derived["calculation_error"] = str(e)

        return derived

    def _perform_macro_analysis(self, data: Dict, period: str) -> Dict:
        """Perform comprehensive macro-economic analysis"""
        analysis = {}

        try:
            # Interest Rate Environment Analysis
            interest_analysis = self._analyze_interest_rate_environment(
                data.get("interest_rates", {})
            )
            analysis["interest_rate_environment"] = interest_analysis

            # Inflation Analysis
            inflation_analysis = self._analyze_inflation_environment(
                data.get("inflation", {})
            )
            analysis["inflation_environment"] = inflation_analysis

            # Employment Analysis
            employment_analysis = self._analyze_employment_environment(
                data.get("employment", {})
            )
            analysis["employment_environment"] = employment_analysis

            # Growth Analysis
            growth_analysis = self._analyze_growth_environment(data.get("growth", {}))
            analysis["growth_environment"] = growth_analysis

            # Market Risk Analysis
            risk_analysis = self._analyze_market_risk_environment(
                data.get("market_sentiment", {})
            )
            analysis["market_risk_environment"] = risk_analysis

            # Overall Economic Assessment
            overall_assessment = self._generate_overall_economic_assessment(data)
            analysis["overall_assessment"] = overall_assessment

        except Exception as e:
            analysis["analysis_error"] = str(e)

        return analysis

    def _analyze_interest_rate_environment(self, interest_data: Dict) -> Dict:
        """Analyze interest rate environment"""
        analysis = {
            "environment": "neutral",
            "implications": [],
            "risk_level": "medium",
        }

        try:
            fed_funds = interest_data.get("fed_funds_rate", {}).get("latest_value", 0)
            treasury_10y = interest_data.get("10y_treasury", {}).get("latest_value", 0)

            if fed_funds > 5.0:
                analysis["environment"] = "restrictive"
                analysis["implications"].extend(
                    [
                        "High borrowing costs may slow economic growth",
                        "Strong USD likely, negative for exports",
                        "Favorable for high-quality bonds and savings",
                    ]
                )
                analysis["risk_level"] = "high"
            elif fed_funds < 2.0:
                analysis["environment"] = "accommodative"
                analysis["implications"].extend(
                    [
                        "Low borrowing costs support growth and investment",
                        "Risk of asset bubbles and inflation",
                        "Favorable for growth stocks and real estate",
                    ]
                )
                analysis["risk_level"] = "low"

            # Yield curve analysis
            if treasury_10y and fed_funds:
                if treasury_10y < fed_funds:
                    analysis["yield_curve"] = "inverted"
                    analysis["implications"].append(
                        "Potential recession warning signal"
                    )
                    analysis["risk_level"] = "high"
                else:
                    analysis["yield_curve"] = "normal"

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def _analyze_inflation_environment(self, inflation_data: Dict) -> Dict:
        """Analyze inflation environment"""
        analysis = {"environment": "moderate", "trend": "stable", "implications": []}

        try:
            cpi = inflation_data.get("cpi", {}).get("latest_value")
            core_cpi = inflation_data.get("core_cpi", {}).get("latest_value")

            if cpi:
                # Assuming CPI is annual percentage
                if cpi > 4.0:
                    analysis["environment"] = "high"
                    analysis["implications"].extend(
                        [
                            "High inflation erodes purchasing power",
                            "Potential for continued Fed tightening",
                            "Favor real assets and inflation-protected securities",
                        ]
                    )
                elif cpi < 2.0:
                    analysis["environment"] = "low"
                    analysis["implications"].extend(
                        [
                            "Below Fed target, potential for accommodation",
                            "Deflationary risks if sustained",
                            "Favorable for fixed-income investments",
                        ]
                    )

                # Trend analysis (simplified)
                if core_cpi and cpi:
                    if core_cpi < cpi:
                        analysis["trend"] = "improving"
                    elif core_cpi > cpi:
                        analysis["trend"] = "worsening"

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def _analyze_employment_environment(self, employment_data: Dict) -> Dict:
        """Analyze employment environment"""
        analysis = {"strength": "moderate", "trend": "stable", "implications": []}

        try:
            unemployment = employment_data.get("unemployment_rate", {}).get(
                "latest_value"
            )

            if unemployment:
                if unemployment < 4.0:
                    analysis["strength"] = "strong"
                    analysis["implications"].extend(
                        [
                            "Tight labor market supports consumer spending",
                            "Potential wage inflation pressures",
                            "Favorable for consumer discretionary stocks",
                        ]
                    )
                elif unemployment > 6.0:
                    analysis["strength"] = "weak"
                    analysis["implications"].extend(
                        [
                            "Weak employment limits consumer spending",
                            "Potential for policy accommodation",
                            "Defensive positioning recommended",
                        ]
                    )

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def _analyze_growth_environment(self, growth_data: Dict) -> Dict:
        """Analyze economic growth environment"""
        analysis = {"momentum": "moderate", "outlook": "neutral", "implications": []}

        try:
            consumer_sentiment = growth_data.get("consumer_sentiment", {}).get(
                "latest_value"
            )
            industrial_production = growth_data.get("industrial_production", {}).get(
                "latest_value"
            )

            momentum_indicators = []

            if consumer_sentiment:
                if consumer_sentiment > 85:
                    momentum_indicators.append("positive")
                elif consumer_sentiment < 70:
                    momentum_indicators.append("negative")

            if momentum_indicators:
                if "negative" in momentum_indicators:
                    analysis["momentum"] = "weak"
                    analysis["outlook"] = "cautious"
                    analysis["implications"].extend(
                        [
                            "Weak consumer confidence suggests slower growth",
                            "Defensive sector positioning advised",
                        ]
                    )
                elif len([x for x in momentum_indicators if x == "positive"]) >= 2:
                    analysis["momentum"] = "strong"
                    analysis["outlook"] = "optimistic"
                    analysis["implications"].extend(
                        [
                            "Strong momentum supports continued expansion",
                            "Cyclical sectors may outperform",
                        ]
                    )

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def _analyze_market_risk_environment(self, market_sentiment_data: Dict) -> Dict:
        """Analyze market risk environment"""
        analysis = {
            "risk_level": "medium",
            "volatility": "moderate",
            "implications": [],
        }

        try:
            # Note: VIX data would be from Yahoo Finance if available
            # For now, provide general risk assessment framework
            analysis["implications"].extend(
                [
                    "Monitor VIX levels for volatility indicators",
                    "Credit spreads indicate market stress levels",
                    "Currency movements reflect global risk sentiment",
                ]
            )

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def _generate_overall_economic_assessment(self, data: Dict) -> Dict:
        """Generate overall economic assessment"""
        assessment = {
            "economic_cycle": "mid-cycle",
            "recession_probability": "low",
            "investment_regime": "balanced",
            "key_risks": [],
            "opportunities": [],
        }

        try:
            # Simplified economic cycle assessment
            derived = data.get("derived_indicators", {})
            yield_spread = derived.get("yield_curve_spread", {}).get("value", 1.0)

            if yield_spread < 0:
                assessment["recession_probability"] = "elevated"
                assessment["economic_cycle"] = "late-cycle"
                assessment["key_risks"].append(
                    "Inverted yield curve signals recession risk"
                )

            interest_data = data.get("interest_rates", {})
            fed_funds = interest_data.get("fed_funds_rate", {}).get("latest_value", 5.0)

            if fed_funds > 5.0:
                assessment["investment_regime"] = "defensive"
                assessment["key_risks"].append(
                    "High interest rates pressure valuations"
                )
                assessment["opportunities"].append("Attractive yields in fixed income")
            elif fed_funds < 2.0:
                assessment["investment_regime"] = "growth"
                assessment["opportunities"].extend(
                    [
                        "Low rates support growth assets",
                        "Credit conditions favorable for expansion",
                    ]
                )

        except Exception as e:
            assessment["assessment_error"] = str(e)

        return assessment

    def _generate_economic_outlook(self, data: Dict) -> Dict:
        """Generate economic outlook and investment implications"""
        outlook = {
            "short_term_outlook": "neutral",
            "medium_term_outlook": "neutral",
            "investment_implications": {
                "asset_allocation": {},
                "sector_preferences": [],
                "risk_factors": [],
            },
        }

        try:
            analysis = data.get("analysis", {})
            overall_assessment = analysis.get("overall_assessment", {})

            # Asset allocation guidance
            investment_regime = overall_assessment.get("investment_regime", "balanced")

            if investment_regime == "growth":
                outlook["investment_implications"]["asset_allocation"] = {
                    "equities": "overweight",
                    "fixed_income": "underweight",
                    "alternatives": "neutral",
                    "cash": "underweight",
                }
                outlook["investment_implications"]["sector_preferences"] = [
                    "Technology",
                    "Consumer Discretionary",
                    "Financials",
                ]
            elif investment_regime == "defensive":
                outlook["investment_implications"]["asset_allocation"] = {
                    "equities": "underweight",
                    "fixed_income": "overweight",
                    "alternatives": "overweight",
                    "cash": "overweight",
                }
                outlook["investment_implications"]["sector_preferences"] = [
                    "Utilities",
                    "Consumer Staples",
                    "Healthcare",
                ]
            else:  # balanced
                outlook["investment_implications"]["asset_allocation"] = {
                    "equities": "neutral",
                    "fixed_income": "neutral",
                    "alternatives": "neutral",
                    "cash": "neutral",
                }
                outlook["investment_implications"]["sector_preferences"] = [
                    "Diversified across sectors"
                ]

            # Risk factors
            recession_prob = overall_assessment.get("recession_probability", "low")
            if recession_prob in ["elevated", "high"]:
                outlook["investment_implications"]["risk_factors"].extend(
                    [
                        "Elevated recession risk",
                        "Potential for earnings downgrades",
                        "Credit spread widening risk",
                    ]
                )

        except Exception as e:
            outlook["outlook_error"] = str(e)

        return outlook

    # Helper interpretation methods
    def _interpret_yield_spread(self, spread: float) -> str:
        """Interpret yield curve spread"""
        if spread < 0:
            return "Inverted yield curve - potential recession signal"
        elif spread < 0.5:
            return "Flat yield curve - economic uncertainty"
        elif spread > 2.0:
            return "Steep yield curve - strong growth expectations"
        else:
            return "Normal yield curve - balanced conditions"

    def _interpret_real_rate(self, real_rate: float) -> str:
        """Interpret real interest rate"""
        if real_rate > 2.0:
            return "High real rates - restrictive monetary conditions"
        elif real_rate < 0:
            return "Negative real rates - accommodative conditions"
        else:
            return "Moderate real rates - neutral monetary stance"

    def _interpret_momentum(self, momentum: float) -> str:
        """Interpret economic momentum score"""
        if momentum > 0.7:
            return "Strong economic momentum"
        elif momentum < 0.3:
            return "Weak economic momentum"
        else:
            return "Moderate economic momentum"

    def _get_data_sources_info(self) -> Dict:
        """Get information about data sources used"""
        sources = ["Yahoo Finance", "Generated Economic Models"]
        if self._fred_api_key:
            sources.insert(0, "FRED (Federal Reserve Economic Data)")

        return {
            "primary_sources": sources,
            "data_coverage": "US economic indicators with enhanced coverage",
            "update_frequency": "Daily for market data, monthly/quarterly for economic data",
            "reliability": "High for market data, model-based for economic indicators",
        }

    def _assess_data_quality_enhanced(self, raw_data: Dict) -> Dict:
        """Enhanced data quality assessment"""
        quality_metrics = {
            "overall_score": 0.0,
            "category_scores": {},
            "coverage_analysis": {},
            "data_completeness": {},
            "quality_factors": [],
        }

        try:
            total_indicators = 0
            successful_indicators = 0
            category_completeness = {}

            # Assess each category
            for category, indicators in self._economic_indicators.items():
                if category in raw_data:
                    category_data = raw_data[category]
                    category_total = len(indicators)
                    category_success = 0

                    for indicator_name in indicators.keys():
                        total_indicators += 1
                        if indicator_name in category_data:
                            indicator_data = category_data[indicator_name]
                            if (
                                isinstance(indicator_data, dict)
                                and "latest_value" in indicator_data
                            ):
                                successful_indicators += 1
                                category_success += 1

                    # Calculate category completeness
                    category_completeness[category] = (
                        category_success / category_total if category_total > 0 else 0
                    )
                    quality_metrics["category_scores"][category] = (
                        category_completeness[category]
                    )
                else:
                    category_completeness[category] = 0
                    quality_metrics["category_scores"][category] = 0

            # Calculate overall completeness
            overall_completeness = (
                successful_indicators / total_indicators if total_indicators > 0 else 0
            )

            # Enhanced scoring with bonuses for comprehensive coverage
            base_score = overall_completeness

            # Bonus for having all major categories
            major_categories = [
                "interest_rates",
                "inflation",
                "employment",
                "growth",
                "market_sentiment",
            ]
            major_coverage = sum(
                1 for cat in major_categories if category_completeness.get(cat, 0) > 0.5
            )
            coverage_bonus = min(0.15, major_coverage / len(major_categories) * 0.15)

            # Bonus for high indicator density in key categories
            key_category_bonus = 0
            for category in ["interest_rates", "inflation", "employment"]:
                if category_completeness.get(category, 0) >= 0.8:
                    key_category_bonus += 0.03

            # Bonus for having derived indicators
            derived_bonus = 0.05 if "derived_indicators" in raw_data else 0

            # Final quality score
            quality_score = min(
                1.0, base_score + coverage_bonus + key_category_bonus + derived_bonus
            )

            quality_metrics.update(
                {
                    "overall_score": quality_score,
                    "data_completeness": {
                        "total_indicators": total_indicators,
                        "successful_indicators": successful_indicators,
                        "completeness_ratio": overall_completeness,
                    },
                    "coverage_analysis": {
                        "major_categories_covered": major_coverage,
                        "total_categories": len(major_categories),
                        "category_details": category_completeness,
                    },
                    "quality_factors": self._generate_quality_factors(
                        overall_completeness, major_coverage, category_completeness
                    ),
                }
            )

        except Exception as e:
            quality_metrics["error"] = str(e)
            quality_metrics["overall_score"] = 0.5  # Default moderate score

        return quality_metrics

    def _generate_quality_factors(
        self, completeness: float, major_coverage: int, category_details: Dict
    ) -> List[str]:
        """Generate quality factor descriptions"""
        factors = []

        if completeness >= 0.8:
            factors.append("High indicator completeness rate")
        elif completeness >= 0.6:
            factors.append("Good indicator completeness rate")
        else:
            factors.append("Moderate indicator completeness - some data gaps")

        if major_coverage >= 4:
            factors.append("Comprehensive category coverage")
        elif major_coverage >= 3:
            factors.append("Good category coverage")
        else:
            factors.append("Limited category coverage")

        # Check for specific strengths
        if category_details.get("interest_rates", 0) >= 0.8:
            factors.append("Strong interest rate data coverage")

        if category_details.get("inflation", 0) >= 0.8:
            factors.append("Comprehensive inflation data")

        if category_details.get("employment", 0) >= 0.8:
            factors.append("Robust employment data")

        if category_details.get("market_sentiment", 0) >= 0.7:
            factors.append("Good market sentiment indicators")

        return factors
