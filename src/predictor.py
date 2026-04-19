import joblib
import pandas as pd
import re
from src.preprocess import clean_text

class MentalHealthPredictor:
    def __init__(self, model_path, vectorizer_path):
        # Loading the artifacts we created in Phase 2 and 3
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def _extract_behavioral_features(self, raw_text):
        """Extracts the same features used during model training."""
        i_usage = len(re.findall(r'\bi\b|\bme\b|\bmy\b|\bmine\b', raw_text.lower()))
        abs_words = len(re.findall(r'\balways\b|\bnever\b|\bcompletely\b|\beveryone\b|\bnobody\b', raw_text.lower()))
        text_len = len(raw_text)
        return i_usage, abs_words, text_len

    def predict(self, raw_text):
        # 1. Preprocess input
        cleaned = clean_text(raw_text)
        
        # 2. Transform to TF-IDF
        tfidf_matrix = self.vectorizer.transform([cleaned])
        feature_df = pd.DataFrame(tfidf_matrix.toarray(), 
                                  columns=self.vectorizer.get_feature_names_out())
        
        # 3. Add Behavioral Features
        i_use, abs_w, t_len = self._extract_behavioral_features(raw_text)
        feature_df['feat_i_usage'] = i_use
        feature_df['feat_abs_words'] = abs_w
        feature_df['feat_len'] = t_len
        
        # 4. Get Probability of Distress (Class 1)
        probability = self.model.predict_proba(feature_df)[0][1]
        return round(probability * 100, 2)

def get_empathetic_advice(score):
    """
    Translates the numerical score into supportive feedback for the UI.
    """
    if score > 70:
        return {
            "status": "High Burnout Risk",
            "advice": "Your patterns suggest you're carrying a lot right now. Please consider taking a break or talking to someone you trust.",
            "color": "#e74c3c" # Red
        }
    elif 40 <= score <= 70:
        return {
            "status": "Moderate Stress",
            "advice": "You're showing signs of stress. Maybe it's time for some self-care this evening?",
            "color": "#f39c12" # Orange
        }
    else:
        return {
            "status": "Steady & Balanced",
            "advice": "You seem to be in a good headspace. Keep maintaining this healthy balance!",
            "color": "#2ecc71" # Green
        }