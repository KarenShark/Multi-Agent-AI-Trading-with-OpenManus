from typing import Any, Dict

from app.agent.base import BaseAgent
from app.tool.macro_economic_fetcher import MacroEconomicFetcher


class MacroAnalystAgent(BaseAgent):
    """
    Macro Economic Analyst Agent
    Analyzes macro-economic conditions and their impact on investment strategy
    """

    def __init__(self, name="macro_analyst"):
        super().__init__(name=name, system_prompt="")
        self.macro_tool = MacroEconomicFetcher()

    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze macro-economic environment and generate investment guidance

        Args:
            inputs: Dictionary containing analysis context and parameters

        Returns:
            Dictionary containing macro analysis results
        """
        try:
            # Extract parameters from inputs
            context = inputs.get("macro_context", {})
            indicators = context.get(
                "indicators",
                [
                    "interest_rates",
                    "inflation",
                    "employment",
                    "growth",
                    "market_sentiment",
                ],
            )
            period = context.get("period", "1y")
            country = context.get("country", "US")

            print(f"Performing macro-economic analysis for {country}...")

            # Fetch comprehensive macro-economic data
            macro_result = self.macro_tool.execute(
                indicators=indicators, period=period, country=country
            )

            if macro_result.get("success"):
                macro_data = macro_result["data"]

                # Generate investment strategy adjustments
                strategy_adjustments = self._generate_strategy_adjustments(macro_data)

                # Calculate macro risk factors
                risk_factors = self._calculate_macro_risk_factors(macro_data)

                # Generate sector rotation guidance
                sector_guidance = self._generate_sector_guidance(macro_data)

                # Assess market timing factors
                timing_factors = self._assess_market_timing_factors(macro_data)

                # Create comprehensive macro analysis output
                macro_analysis = {
                    "raw_data": macro_data,
                    "strategy_adjustments": strategy_adjustments,
                    "risk_factors": risk_factors,
                    "sector_guidance": sector_guidance,
                    "timing_factors": timing_factors,
                    "investment_regime": self._determine_investment_regime(macro_data),
                    "confidence_level": self._assess_analysis_confidence(macro_data),
                }

                return {"macro": macro_analysis}

            else:
                print(f"Macro data fetch failed: {macro_result.get('error')}")
                # Return neutral macro analysis
                return {"macro": self._get_neutral_macro_analysis()}

        except Exception as e:
            print(f"Error in macro analysis: {e}")
            return {"macro": self._get_neutral_macro_analysis()}

    def _generate_strategy_adjustments(self, macro_data: Dict) -> Dict:
        """Generate strategy adjustments based on macro conditions"""
        adjustments = {
            "risk_adjustment": 0.0,  # -1.0 (defensive) to +1.0 (aggressive)
            "duration_bias": 0.0,  # -1.0 (short duration) to +1.0 (long duration)
            "growth_value_tilt": 0.0,  # -1.0 (value) to +1.0 (growth)
            "size_bias": 0.0,  # -1.0 (large cap) to +1.0 (small cap)
            "rationale": [],
        }

        try:
            analysis = macro_data.get("analysis", {})
            outlook = macro_data.get("outlook", {})

            # Interest rate environment impact
            interest_env = analysis.get("interest_rate_environment", {})
            if interest_env.get("environment") == "restrictive":
                adjustments["risk_adjustment"] -= 0.3
                adjustments["duration_bias"] -= 0.4
                adjustments["growth_value_tilt"] -= 0.2
                adjustments["rationale"].append(
                    "Restrictive rates favor defensive positioning"
                )
            elif interest_env.get("environment") == "accommodative":
                adjustments["risk_adjustment"] += 0.3
                adjustments["growth_value_tilt"] += 0.3
                adjustments["size_bias"] += 0.2
                adjustments["rationale"].append(
                    "Accommodative rates support growth assets"
                )

            # Inflation environment impact
            inflation_env = analysis.get("inflation_environment", {})
            if inflation_env.get("environment") == "high":
                adjustments["duration_bias"] -= 0.3
                adjustments["growth_value_tilt"] -= 0.2
                adjustments["rationale"].append(
                    "High inflation pressures fixed income and growth stocks"
                )
            elif inflation_env.get("environment") == "low":
                adjustments["duration_bias"] += 0.2
                adjustments["growth_value_tilt"] += 0.1
                adjustments["rationale"].append(
                    "Low inflation supports duration and growth"
                )

            # Employment strength impact
            employment_env = analysis.get("employment_environment", {})
            if employment_env.get("strength") == "strong":
                adjustments["risk_adjustment"] += 0.2
                adjustments["rationale"].append(
                    "Strong employment supports risk assets"
                )
            elif employment_env.get("strength") == "weak":
                adjustments["risk_adjustment"] -= 0.2
                adjustments["rationale"].append(
                    "Weak employment warrants defensive positioning"
                )

            # Growth momentum impact
            growth_env = analysis.get("growth_environment", {})
            if growth_env.get("momentum") == "strong":
                adjustments["risk_adjustment"] += 0.2
                adjustments["size_bias"] += 0.1
                adjustments["rationale"].append(
                    "Strong growth momentum favors cyclical positioning"
                )
            elif growth_env.get("momentum") == "weak":
                adjustments["risk_adjustment"] -= 0.2
                adjustments["rationale"].append(
                    "Weak growth momentum suggests defensive stance"
                )

            # Overall assessment impact
            overall = analysis.get("overall_assessment", {})
            recession_prob = overall.get("recession_probability", "low")
            if recession_prob in ["elevated", "high"]:
                adjustments["risk_adjustment"] -= 0.4
                adjustments["growth_value_tilt"] -= 0.3
                adjustments["size_bias"] -= 0.2
                adjustments["rationale"].append(
                    "Elevated recession risk requires defensive positioning"
                )

            # Cap adjustments to reasonable ranges
            for key in [
                "risk_adjustment",
                "duration_bias",
                "growth_value_tilt",
                "size_bias",
            ]:
                adjustments[key] = max(-1.0, min(1.0, adjustments[key]))

        except Exception as e:
            adjustments["error"] = str(e)

        return adjustments

    def _calculate_macro_risk_factors(self, macro_data: Dict) -> Dict:
        """Calculate macro-economic risk factors"""
        risk_factors = {
            "recession_risk": 0.2,  # 0.0 (low) to 1.0 (high)
            "inflation_risk": 0.3,  # 0.0 (low) to 1.0 (high)
            "interest_rate_risk": 0.3,  # 0.0 (low) to 1.0 (high)
            "currency_risk": 0.2,  # 0.0 (low) to 1.0 (high)
            "geopolitical_risk": 0.3,  # 0.0 (low) to 1.0 (high)
            "overall_risk": 0.3,  # Composite risk score
            "risk_description": [],
        }

        try:
            analysis = macro_data.get("analysis", {})
            derived = macro_data.get("derived_indicators", {})

            # Recession risk assessment
            overall_assessment = analysis.get("overall_assessment", {})
            recession_prob = overall_assessment.get("recession_probability", "low")

            if recession_prob == "high":
                risk_factors["recession_risk"] = 0.8
                risk_factors["risk_description"].append("High recession probability")
            elif recession_prob == "elevated":
                risk_factors["recession_risk"] = 0.6
                risk_factors["risk_description"].append("Elevated recession risk")
            elif recession_prob == "medium":
                risk_factors["recession_risk"] = 0.4
            else:
                risk_factors["recession_risk"] = 0.2

            # Yield curve inversion risk
            yield_spread = derived.get("yield_curve_spread", {})
            if yield_spread.get("value", 1.0) < 0:
                risk_factors["recession_risk"] = max(
                    risk_factors["recession_risk"], 0.7
                )
                risk_factors["risk_description"].append("Inverted yield curve warning")

            # Interest rate risk
            interest_env = analysis.get("interest_rate_environment", {})
            if interest_env.get("environment") == "restrictive":
                risk_factors["interest_rate_risk"] = 0.7
                risk_factors["risk_description"].append("Restrictive rate environment")
            elif interest_env.get("risk_level") == "high":
                risk_factors["interest_rate_risk"] = 0.6

            # Inflation risk
            inflation_env = analysis.get("inflation_environment", {})
            if inflation_env.get("environment") == "high":
                risk_factors["inflation_risk"] = 0.7
                risk_factors["risk_description"].append("High inflation environment")
            elif inflation_env.get("trend") == "worsening":
                risk_factors["inflation_risk"] = 0.5

            # Calculate overall risk as weighted average
            weights = {
                "recession_risk": 0.3,
                "inflation_risk": 0.25,
                "interest_rate_risk": 0.25,
                "currency_risk": 0.1,
                "geopolitical_risk": 0.1,
            }

            overall_risk = sum(
                risk_factors[factor] * weight for factor, weight in weights.items()
            )
            risk_factors["overall_risk"] = overall_risk

        except Exception as e:
            risk_factors["error"] = str(e)

        return risk_factors

    def _generate_sector_guidance(self, macro_data: Dict) -> Dict:
        """Generate sector rotation guidance based on macro conditions"""
        sector_guidance = {
            "overweight": [],
            "neutral": [],
            "underweight": [],
            "rationale": {},
        }

        try:
            outlook = macro_data.get("outlook", {})
            analysis = macro_data.get("analysis", {})

            # Get investment regime
            overall_assessment = analysis.get("overall_assessment", {})
            investment_regime = overall_assessment.get("investment_regime", "balanced")

            # Sector preferences based on investment regime
            if investment_regime == "growth":
                sector_guidance["overweight"] = [
                    "Technology",
                    "Consumer Discretionary",
                    "Communication Services",
                ]
                sector_guidance["underweight"] = ["Utilities", "Consumer Staples"]
                sector_guidance["rationale"][
                    "Technology"
                ] = "Low rates support high-growth sectors"
                sector_guidance["rationale"][
                    "Consumer Discretionary"
                ] = "Strong consumer environment"

            elif investment_regime == "defensive":
                sector_guidance["overweight"] = [
                    "Utilities",
                    "Consumer Staples",
                    "Healthcare",
                ]
                sector_guidance["underweight"] = ["Technology", "Materials", "Energy"]
                sector_guidance["rationale"][
                    "Utilities"
                ] = "Defensive characteristics in uncertain environment"
                sector_guidance["rationale"][
                    "Healthcare"
                ] = "Non-cyclical demand and pricing power"

            else:  # balanced
                sector_guidance["neutral"] = [
                    "Technology",
                    "Healthcare",
                    "Financials",
                    "Consumer Discretionary",
                    "Industrials",
                    "Consumer Staples",
                    "Utilities",
                    "Materials",
                    "Energy",
                ]
                sector_guidance["rationale"][
                    "Balanced"
                ] = "Diversified exposure across sectors"

            # Interest rate sensitive adjustments
            interest_env = analysis.get("interest_rate_environment", {})
            if interest_env.get("environment") == "restrictive":
                if "Financials" not in sector_guidance["overweight"]:
                    sector_guidance["overweight"].append("Financials")
                sector_guidance["rationale"][
                    "Financials"
                ] = "Banks benefit from higher net interest margins"

                # Remove REITs-like exposure
                if "Real Estate" in sector_guidance["overweight"]:
                    sector_guidance["overweight"].remove("Real Estate")
                    sector_guidance["underweight"].append("Real Estate")
                    sector_guidance["rationale"][
                        "Real Estate"
                    ] = "Rate sensitivity and funding costs"

            # Inflation adjustments
            inflation_env = analysis.get("inflation_environment", {})
            if inflation_env.get("environment") == "high":
                if "Energy" not in sector_guidance["overweight"]:
                    sector_guidance["overweight"].append("Energy")
                sector_guidance["rationale"][
                    "Energy"
                ] = "Commodity exposure provides inflation hedge"

                if "Materials" not in sector_guidance["overweight"]:
                    sector_guidance["overweight"].append("Materials")
                sector_guidance["rationale"][
                    "Materials"
                ] = "Pricing power in inflationary environment"

            # Employment strength adjustments
            employment_env = analysis.get("employment_environment", {})
            if employment_env.get("strength") == "strong":
                if "Consumer Discretionary" not in sector_guidance["overweight"]:
                    sector_guidance["overweight"].append("Consumer Discretionary")
                sector_guidance["rationale"][
                    "Consumer Discretionary"
                ] = "Strong employment supports discretionary spending"

        except Exception as e:
            sector_guidance["error"] = str(e)

        return sector_guidance

    def _assess_market_timing_factors(self, macro_data: Dict) -> Dict:
        """Assess market timing factors based on macro indicators"""
        timing_factors = {
            "market_phase": "neutral",  # early, mid, late, recession
            "trend_direction": "neutral",  # bullish, bearish, neutral
            "volatility_regime": "medium",  # low, medium, high
            "momentum_indicators": [],
            "contrarian_indicators": [],
            "timing_score": 0.0,  # -1.0 (bearish) to +1.0 (bullish)
        }

        try:
            analysis = macro_data.get("analysis", {})
            derived = macro_data.get("derived_indicators", {})

            # Economic cycle assessment
            overall_assessment = analysis.get("overall_assessment", {})
            economic_cycle = overall_assessment.get("economic_cycle", "mid-cycle")
            recession_prob = overall_assessment.get("recession_probability", "low")

            timing_factors["market_phase"] = economic_cycle

            # Momentum indicators
            growth_env = analysis.get("growth_environment", {})
            if growth_env.get("momentum") == "strong":
                timing_factors["momentum_indicators"].append("Strong economic momentum")
                timing_factors["timing_score"] += 0.3
            elif growth_env.get("momentum") == "weak":
                timing_factors["momentum_indicators"].append("Weak economic momentum")
                timing_factors["timing_score"] -= 0.3

            employment_env = analysis.get("employment_environment", {})
            if employment_env.get("strength") == "strong":
                timing_factors["momentum_indicators"].append("Strong employment")
                timing_factors["timing_score"] += 0.2
            elif employment_env.get("strength") == "weak":
                timing_factors["momentum_indicators"].append("Weak employment")
                timing_factors["timing_score"] -= 0.2

            # Contrarian indicators
            if recession_prob in ["elevated", "high"]:
                timing_factors["contrarian_indicators"].append(
                    "Recession fears may create opportunities"
                )
                # Don't adjust timing_score for contrarian indicators

            # Yield curve analysis
            yield_spread = derived.get("yield_curve_spread", {})
            if yield_spread.get("value", 1.0) < 0:
                timing_factors["contrarian_indicators"].append(
                    "Inverted yield curve - historically preceded opportunities"
                )
                timing_factors["timing_score"] -= 0.2  # Near-term bearish

            # Interest rate cycle
            interest_env = analysis.get("interest_rate_environment", {})
            if interest_env.get("environment") == "restrictive":
                timing_factors["contrarian_indicators"].append(
                    "Peak rate environment may signal cycle turn"
                )
            elif interest_env.get("environment") == "accommodative":
                timing_factors["momentum_indicators"].append(
                    "Accommodative rates support asset prices"
                )
                timing_factors["timing_score"] += 0.2

            # Determine overall trend direction
            if timing_factors["timing_score"] > 0.3:
                timing_factors["trend_direction"] = "bullish"
            elif timing_factors["timing_score"] < -0.3:
                timing_factors["trend_direction"] = "bearish"
            else:
                timing_factors["trend_direction"] = "neutral"

            # Volatility regime assessment (simplified)
            market_risk_env = analysis.get("market_risk_environment", {})
            timing_factors["volatility_regime"] = market_risk_env.get(
                "volatility", "medium"
            )

        except Exception as e:
            timing_factors["error"] = str(e)

        return timing_factors

    def _determine_investment_regime(self, macro_data: Dict) -> Dict:
        """Determine the current investment regime"""
        regime = {
            "regime_type": "balanced",  # growth, value, defensive, balanced
            "confidence": 0.5,  # 0.0 to 1.0
            "duration": "medium_term",  # short_term, medium_term, long_term
            "key_drivers": [],
            "regime_description": "",
        }

        try:
            analysis = macro_data.get("analysis", {})
            overall_assessment = analysis.get("overall_assessment", {})

            investment_regime = overall_assessment.get("investment_regime", "balanced")
            regime["regime_type"] = investment_regime

            # Determine confidence based on indicator alignment
            indicators_aligned = 0
            total_indicators = 0

            # Check alignment of different macro environments
            environments = [
                analysis.get("interest_rate_environment", {}),
                analysis.get("inflation_environment", {}),
                analysis.get("employment_environment", {}),
                analysis.get("growth_environment", {}),
            ]

            for env in environments:
                if env:
                    total_indicators += 1
                    # Check if environment supports current regime
                    # This is a simplified alignment check
                    if investment_regime == "growth":
                        if (
                            env.get("environment") in ["accommodative", "low"]
                            or env.get("strength") == "strong"
                            or env.get("momentum") == "strong"
                        ):
                            indicators_aligned += 1
                    elif investment_regime == "defensive":
                        if (
                            env.get("environment") in ["restrictive", "high"]
                            or env.get("strength") == "weak"
                            or env.get("momentum") == "weak"
                        ):
                            indicators_aligned += 1
                    else:  # balanced
                        indicators_aligned += 0.5  # Neutral always partially aligned

            if total_indicators > 0:
                regime["confidence"] = indicators_aligned / total_indicators

            # Key drivers
            recession_prob = overall_assessment.get("recession_probability", "low")
            if recession_prob in ["elevated", "high"]:
                regime["key_drivers"].append("Recession risk")

            interest_env = analysis.get("interest_rate_environment", {})
            if interest_env.get("environment") in ["restrictive", "accommodative"]:
                regime["key_drivers"].append(
                    f"Interest rate environment: {interest_env.get('environment')}"
                )

            # Regime description
            if regime["regime_type"] == "growth":
                regime["regime_description"] = (
                    "Favorable conditions for growth-oriented investments"
                )
            elif regime["regime_type"] == "defensive":
                regime["regime_description"] = (
                    "Defensive positioning warranted by macro headwinds"
                )
            elif regime["regime_type"] == "value":
                regime["regime_description"] = (
                    "Value opportunities emerging from macro dislocations"
                )
            else:
                regime["regime_description"] = (
                    "Balanced approach suitable for current conditions"
                )

        except Exception as e:
            regime["error"] = str(e)

        return regime

    def _assess_analysis_confidence(self, macro_data: Dict) -> Dict:
        """Assess confidence level of macro analysis"""
        confidence = {
            "overall_confidence": 0.7,  # 0.0 to 1.0
            "data_quality": 0.8,  # 0.0 to 1.0
            "indicator_alignment": 0.6,  # 0.0 to 1.0
            "uncertainty_factors": [],
            "confidence_level": "medium",  # low, medium, high
        }

        try:
            # Check data quality
            metadata = macro_data.get("metadata", {})
            data_sources = metadata.get("data_sources", {})

            if "FRED" in str(data_sources):
                confidence["data_quality"] = 0.9
            elif "Yahoo Finance" in str(data_sources):
                confidence["data_quality"] = 0.8
            else:
                confidence["data_quality"] = 0.6
                confidence["uncertainty_factors"].append(
                    "Limited real-time data availability"
                )

            # Check for errors in analysis
            analysis = macro_data.get("analysis", {})
            error_count = 0
            total_sections = 0

            for section_name, section_data in analysis.items():
                total_sections += 1
                if isinstance(section_data, dict) and "error" in section_data:
                    error_count += 1

            if total_sections > 0:
                confidence["indicator_alignment"] = 1.0 - (error_count / total_sections)

            # Overall confidence calculation
            confidence["overall_confidence"] = (
                confidence["data_quality"] * 0.4
                + confidence["indicator_alignment"] * 0.6
            )

            # Confidence level classification
            if confidence["overall_confidence"] > 0.8:
                confidence["confidence_level"] = "high"
            elif confidence["overall_confidence"] > 0.6:
                confidence["confidence_level"] = "medium"
            else:
                confidence["confidence_level"] = "low"
                confidence["uncertainty_factors"].append(
                    "Low overall analysis confidence"
                )

            # Add specific uncertainty factors
            derived = macro_data.get("derived_indicators", {})
            if "calculation_error" in derived:
                confidence["uncertainty_factors"].append(
                    "Issues calculating derived indicators"
                )

        except Exception as e:
            confidence["error"] = str(e)
            confidence["overall_confidence"] = 0.5
            confidence["confidence_level"] = "low"

        return confidence

    def _get_neutral_macro_analysis(self) -> Dict:
        """Return neutral macro analysis when data is unavailable"""
        return {
            "strategy_adjustments": {
                "risk_adjustment": 0.0,
                "duration_bias": 0.0,
                "growth_value_tilt": 0.0,
                "size_bias": 0.0,
                "rationale": ["Neutral positioning due to limited macro data"],
            },
            "risk_factors": {
                "recession_risk": 0.3,
                "inflation_risk": 0.3,
                "interest_rate_risk": 0.3,
                "currency_risk": 0.2,
                "geopolitical_risk": 0.3,
                "overall_risk": 0.3,
                "risk_description": ["Moderate risk assumed due to data limitations"],
            },
            "sector_guidance": {
                "neutral": ["Diversified sector exposure recommended"],
                "rationale": {
                    "Neutral": "Balanced approach due to limited macro insights"
                },
            },
            "timing_factors": {
                "market_phase": "neutral",
                "trend_direction": "neutral",
                "volatility_regime": "medium",
                "timing_score": 0.0,
            },
            "investment_regime": {
                "regime_type": "balanced",
                "confidence": 0.3,
                "regime_description": "Balanced approach due to limited macro data",
            },
            "confidence_level": {
                "overall_confidence": 0.3,
                "confidence_level": "low",
                "uncertainty_factors": ["Limited macro-economic data availability"],
            },
        }
