from app.agent.market_analyst import MarketAnalystAgent
from app.agent.sentiment_analyzer import SentimentAnalyzerAgent
from app.agent.technical_trader import TechnicalTraderAgent
from app.agent.risk_manager import RiskManagerAgent
from app.task.trading_task import TradingTask
import json

def test_trading_system():
    """Test the complete trading system with different objectives"""
    
    print("🚀 Testing StockSynergy Trading System")
    print("=" * 50)
    
    # Initialize agents
    agents = {
        "analyst":   MarketAnalystAgent(),
        "sentiment": SentimentAnalyzerAgent(),
        "technical": TechnicalTraderAgent(),
        "risk":      RiskManagerAgent(),
    }
    
    task = TradingTask(agents)
    
    # Test different objectives
    objectives = ["growth", "value", "balanced"]
    
    for objective in objectives:
        print(f"\n📊 Testing {objective.upper()} Strategy")
        print("-" * 30)
        
        result = task.run_once({
            "universe": {"objective": objective, "max_candidates": 3},
            "portfolio": {"cash": 100000}
        })
        
        # Display results
        print("\n📈 Market Analysis Results:")
        analyst_data = result["analyst"]
        for ticker_data in analyst_data["tickers"]:
            symbol = ticker_data["symbol"]
            industry = ticker_data.get("industry", "N/A")
            market_cap = ticker_data.get("market_cap", 0)
            rationale = analyst_data["rationale"].get(symbol, "No rationale")
            
            print(f"  • {symbol}: {industry}")
            if market_cap:
                print(f"    Market Cap: ${market_cap/1e9:.1f}B")
            print(f"    Rationale: {rationale}")
        
        print("\n💭 Sentiment Analysis:")
        sentiment_data = result["sentiment"]
        for symbol, score in sentiment_data["scores"].items():
            sentiment_label = "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"
            print(f"  • {symbol}: {score:.3f} ({sentiment_label})")
            
            # Show news sources if available
            sources = sentiment_data["sources"].get(symbol, [])
            if sources:
                print(f"    News headlines: {', '.join(sources[:2])}")
        
        print("\n⚡ Technical Analysis:")
        technical_data = result["technical"]
        for symbol, signal_data in technical_data["signals"].items():
            action = signal_data["action"]
            confidence = signal_data["confidence"]
            print(f"  • {symbol}: {action.upper()} (confidence: {confidence:.2f})")
        
        print("\n💰 Risk Management & Orders:")
        risk_data = result["risk"]
        orders = risk_data["orders"]
        if orders:
            for order in orders:
                print(f"  • {order['side'].upper()} {order['qty']} shares of {order['symbol']}")
        else:
            print("  • No orders generated (all signals were FLAT or SHORT)")
            
        risk_metrics = risk_data["risk_metrics"]
        print(f"  • Portfolio exposure: {risk_metrics['exposure']} positions")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_trading_system()