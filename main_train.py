from src.features import build_feature_matrix
import joblib
import os

# Paths
PROCESSED_DATA = "data/processed/cleaned_mental_health.csv"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
FEATURES_OUTPUT = "data/processed/final_feature_matrix.pkl"

def run_phase_2():
    if not os.path.exists(PROCESSED_DATA):
        print("❌ Error: Cleaned data not found. Run Phase 1 first!")
        return

    print("🚀 Starting Phase 2: Feature Engineering...")
    X, y = build_feature_matrix(PROCESSED_DATA, VECTORIZER_PATH)
    
    # Saving X and y as a combined object for Phase 3 (Training)
    # Using joblib because the matrix is too large for a standard CSV comfortably
    joblib.dump((X, y), FEATURES_OUTPUT)
    
    print(f"✅ Phase 2 Complete! Feature Matrix Shape: {X.shape}")
    print(f"📦 Saved features to {FEATURES_OUTPUT}")

if __name__ == "__main__":
    run_phase_2()