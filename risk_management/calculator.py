import numpy as np
from scipy.stats import norm

class RiskManager:
    def __init__(self, confidence=0.95):
        self.confidence = confidence
    
    def value_at_risk(self, returns):
        mean = np.mean(returns)
        std = np.std(returns)
        z = norm.ppf(1 - self.confidence)
        return mean + z * std
    
    def position_size(self, balance, risk_pct, entry, stop_loss):
        risk_amount = balance * (risk_pct / 100)
        return risk_amount / abs(entry - stop_loss)
