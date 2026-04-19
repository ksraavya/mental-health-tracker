import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC # Faster for text
from sklearn.calibration import CalibratedClassifierCV # To get probabilities
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_mental_health_model():
    print("🧠 Loading feature matrix...")
    X, y = joblib.load('data/processed/final_feature_matrix.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🏗️ Initializing Optimized Ensemble Members...")
    
    # 1. Random Forest (Still fast)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)
    
    # 2. Faster SVM: LinearSVC + Calibration
    # LinearSVC is O(n), much faster than SVC's O(n^2)
    base_svm = LinearSVC(class_weight='balanced', max_iter=1000, dual=False)
    svm_calibrated = CalibratedClassifierCV(base_svm, cv=3) # This provides the 'predict_proba'
    
    # 3. XGBoost (Still fast)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    print("🤝 Creating the Soft-Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm_calibrated), ('xgb', xgb)],
        voting='soft'
    )

    print("🔥 Training Optimized Ensemble (Estimated time: < 2 mins)...")
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    print("\n📊 Model Performance:")
    print(classification_report(y_test, y_pred))

    joblib.dump(ensemble, 'models/ensemble_model.pkl')
    print("\n✅ Success! Model saved.")

if __name__ == "__main__":
    train_mental_health_model()