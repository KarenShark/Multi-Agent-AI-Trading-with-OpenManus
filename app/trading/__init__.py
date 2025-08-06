"""
Trading Package

This package provides the complete trading system for StockSynergy AI:
- Trading Environment (Gym-compatible)
- Supervised Learning Traders (Random Forest, XGBoost, Ensemble)
- Reinforcement Learning Agents (PPO, SAC, Custom)
- Model Training and Evaluation Tools
"""

from .trading_env import TradingEnvironment, create_trading_environment
from .supervised_trader import (
    SupervisedTrader,
    EnsembleTrader,
    create_supervised_trader,
    create_ensemble_trader
)
from .rl_agent import (
    PPOAgent,
    TradingRLAgent,
    create_ppo_agent,
    create_trading_rl_manager
)

__all__ = [
    # Trading Environment
    "TradingEnvironment",
    "create_trading_environment",

    # Supervised Learning
    "SupervisedTrader",
    "EnsembleTrader",
    "create_supervised_trader",
    "create_ensemble_trader",

    # Reinforcement Learning
    "PPOAgent",
    "TradingRLAgent",
    "create_ppo_agent",
    "create_trading_rl_manager",
]