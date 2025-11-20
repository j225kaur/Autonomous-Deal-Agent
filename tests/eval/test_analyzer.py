import pytest
import pandas as pd
import numpy as np
from src.analysis.signal_model import SignalModel

def test_signal_model_scoring():
    model = SignalModel()
    
    # Mock data: 30 days of flat price, then a jump
    prices = [100.0] * 29 + [110.0] # 10% jump
    volumes = [1000] * 29 + [5000]  # 5x volume
    
    # Convert to pandas Series
    p_series = pd.Series(prices)
    v_series = pd.Series(volumes)
    
    stats = model.compute_statistical_features(p_series, v_series)
    
    # Expect high Z-score and Volume Shock
    assert stats["z_score"] > 2.0
    assert stats["volume_shock"] > 3.0

    # Test scoring
    score_obj = model.score_ticker("TEST", {"close": prices, "volume": volumes}, ["Some random news"])
    
    # Should be high score due to shocks
    assert score_obj.total_score > 0.5
    assert any("volume spike" in e.lower() for e in score_obj.explanation)

def test_signal_model_keywords():
    model = SignalModel()
    
    # Flat data
    prices = [100.0] * 30
    volumes = [1000] * 30
    
    # News with deal keywords
    news = ["Rumors of acquisition by BigCorp"]
    
    score_obj = model.score_ticker("TEST", {"close": prices, "volume": volumes}, news)
    
    # Should have some score from news
    assert score_obj.total_score >= 0.3
    assert any("deal keywords" in e.lower() for e in score_obj.explanation)
