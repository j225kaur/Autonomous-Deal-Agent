import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class SignalScore:
    ticker: str
    total_score: float  # 0.0 to 1.0
    components: Dict[str, float]
    explanation: List[str]

class SignalModel:
    """
    Computes deal signals based on:
    1. Statistical anomalies (Price Z-score, Volume shock, Volatility)
    2. ML-based text classification (Headline sentiment/deal probability)
    3. Metadata heuristics
    """
    
    def __init__(self):
        # In a real scenario, we'd load trained models here
        # self.clf = joblib.load("deal_classifier.pkl")
        self.headline_model = None # Placeholder for LogisticRegression
        self.filing_model = None   # Placeholder for RandomForestClassifier

    def predict_headline_proba(self, headline: str) -> float:
        """
        Predict deal probability from headline text.
        """
        if not self.headline_model:
            # Fallback heuristic
            deal_keywords = ["merger", "acquisition", "buyout", "talks", "rumor", "proposal", "explore strategic alternatives"]
            if any(k in headline.lower() for k in deal_keywords):
                return 0.7
            return 0.1
        
        # Real model prediction would go here
        # return self.headline_model.predict_proba([headline])[0][1]
        return 0.0

    def predict_filing_proba(self, filing_type: str, filing_text: str) -> float:
        """
        Predict deal probability from filing metadata/text.
        """
        # High signal filings
        if filing_type in ["SC 13D", "SC 13D/A", "425", "DEFM14A"]:
            return 0.9
        if filing_type in ["8-K"]:
            # Check 8-K content for specific items (e.g. 1.01 Entry into a Material Definitive Agreement)
            if "Entry into a Material Definitive Agreement" in filing_text:
                return 0.95
        return 0.05

    def compute_statistical_features(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, float]:
        """
        Compute rolling stats.
        """
        if len(prices) < 30:
            return {"z_score": 0.0, "volatility": 0.0, "volume_shock": 0.0}

        # 1. Price Z-Score (vs 20-day MA)
        ma_20 = prices.rolling(window=20).mean().iloc[-1]
        std_20 = prices.rolling(window=20).std().iloc[-1]
        current_price = prices.iloc[-1]
        
        z_score = 0.0
        if std_20 > 0:
            z_score = (current_price - ma_20) / std_20

        # 2. Rolling Volatility (annualized)
        returns = prices.pct_change().dropna()
        vol_20 = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)

        # 3. Volume Shock (vs 20-day avg)
        avg_vol = volumes.rolling(window=20).mean().iloc[-1]
        current_vol = volumes.iloc[-1]
        vol_shock = 0.0
        if avg_vol > 0:
            vol_shock = current_vol / avg_vol

        return {
            "z_score": float(z_score),
            "volatility": float(vol_20),
            "volume_shock": float(vol_shock)
        }

    def score_ticker(self, ticker: str, market_data: Dict[str, List[float]], news_items: List[str]) -> SignalScore:
        """
        Generate a composite deal score.
        market_data expected keys: 'close', 'volume' (lists of floats)
        """
        # Convert to pandas
        prices = pd.Series(market_data.get("close", []))
        volumes = pd.Series(market_data.get("volume", []))

        stats = self.compute_statistical_features(prices, volumes)
        
        # Heuristic scoring logic
        # Deal signals often accompanied by:
        # - High volume shock (> 2.0)
        # - High volatility spike
        # - Price jump (Z-score > 2.0)
        
        score = 0.0
        explanations = []

        # Volume contribution
        if stats["volume_shock"] > 3.0:
            score += 0.3
            explanations.append(f"Massive volume spike ({stats['volume_shock']:.1f}x avg)")
        elif stats["volume_shock"] > 1.5:
            score += 0.1
            explanations.append(f"Elevated volume ({stats['volume_shock']:.1f}x avg)")

        # Price contribution
        if abs(stats["z_score"]) > 3.0:
            score += 0.3
            explanations.append(f"Extreme price move (Z={stats['z_score']:.1f})")
        elif abs(stats["z_score"]) > 2.0:
            score += 0.15
            explanations.append(f"Significant price move (Z={stats['z_score']:.1f})")

        # Volatility contribution
        if stats["volatility"] > 0.5: # > 50% annualized
            score += 0.1
            explanations.append(f"High volatility ({stats['volatility']:.1%})")

        # News/Text contribution (Placeholder for ML)
        # Simple keyword boost for now
        deal_keywords = ["merger", "acquisition", "buyout", "talks", "rumor", "proposal"]
        news_score = 0.0
        for item in news_items:
            if any(k in item.lower() for k in deal_keywords):
                news_score = 0.3
                explanations.append("News contains deal keywords")
                break
        
        score += news_score

        # Cap at 1.0
        score = min(1.0, score)

        return SignalScore(
            ticker=ticker,
            total_score=score,
            components=stats,
            explanation=explanations
        )
