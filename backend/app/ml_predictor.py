"""
ML Prediction utilities
"""

import joblib
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

class MLPredictor:
    """ML model predictor for win probability"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        
        # Try to find default model if not provided
        if not model_path:
            # Look in root directory or app directory
            potential_paths = [
                Path("win_probability_model.pkl"),
                Path(__file__).parent.parent / "win_probability_model.pkl",
                Path(__file__).parent / "win_probability_model.pkl"
            ]
            for p in potential_paths:
                if p.exists():
                    model_path = str(p)
                    break
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print(f"⚠️ Model file not found. Fallback rules will be used.")
    
    def load_model(self, model_path: str):
        """Load trained ML model"""
        try:
            self.model = joblib.load(model_path)
            print(f"✓ ML model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            self.model = None
    
    def predict(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Make prediction using ML model"""
        
        X = np.array(feature_vector).reshape(1, -1)
        
        if self.model:
            try:
                prediction = self.model.predict(X)[0]
                probability = self.model.predict_proba(X)[0]
                
                return {
                    "outcome": "WIN" if prediction == 1 else "LOSS",
                    "win_probability": float(probability[1]) if len(probability) > 1 else float(probability[0]),
                    "prediction_class": int(prediction),
                    "method": "ml_model"
                }
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}")
                return self._fallback_prediction(feature_vector)
        else:
            return self._fallback_prediction(feature_vector)
    
    def _fallback_prediction(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Fallback rule-based prediction"""
        
        # Extract key features
        compliance = feature_vector[2] if len(feature_vector) > 2 else 50
        score = feature_vector[3] if len(feature_vector) > 3 else 50
        gaps = feature_vector[6] if len(feature_vector) > 6 else 5
        
        # Simple formula
        win_prob = (compliance * 0.4 + score * 0.4 + max(0, 100 - gaps * 10) * 0.2)
        win_prob = min(max(win_prob, 0), 100)
        
        return {
            "outcome": "WIN" if win_prob >= 60 else "LOSS",
            "win_probability": win_prob / 100,
            "prediction_class": 1 if win_prob >= 60 else 0,
            "method": "fallback_rule_based"
        }

# Singleton instance
_ml_predictor_instance = None

def get_ml_predictor(model_path: Optional[str] = None) -> MLPredictor:
    """Get singleton ML predictor instance"""
    global _ml_predictor_instance
    if _ml_predictor_instance is None:
        _ml_predictor_instance = MLPredictor(model_path)
    return _ml_predictor_instance