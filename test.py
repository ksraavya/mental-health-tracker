from src.predictor import MentalHealthPredictor

# Initialize the predictor (make sure paths match your folder)
predictor = MentalHealthPredictor(
    model_path='models/ensemble_model.pkl',
    vectorizer_path='models/tfidf_vectorizer.pkl'
)

# Test 1: High Stress Input
text_1 = "I am so overwhelmed with these exams and I feel like I'm failing everyone. I can't sleep and everything feels hopeless."
score_1 = predictor.predict(text_1)
print(f"Test 1 (Distress): {score_1}%")

# Test 2: Healthy/Neutral Input
text_2 = "I had a great day at MAIT today! The weather was nice and I finally finished my project. Feeling good."
score_2 = predictor.predict(text_2)
print(f"Test 2 (Healthy): {score_2}%")