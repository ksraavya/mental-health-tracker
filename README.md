# 🌿 MindSpace — Daily Mental Wellness Companion

> A burnout-detection and daily wellness app powered by a hypothesis-driven NLP ensemble, built as a research tool for understanding mental health patterns in text.

**[Live Demo →](https://mind-space-project.streamlit.app)** · **[EDA Notebook →](https://github.com/ksraavya/mental-health-tracker/blob/main/notebooks/exploration.ipynb)**

---

## What it does

MindSpace is a personal wellness companion that helps you check in with yourself daily — through journaling, mood tracking, breathwork, and gratitude logging. Under the hood, it runs your journal entries through a trained NLP ensemble to estimate burnout risk and energy levels in real time.

It's not a clinical tool. It's a mirror.

---

## App Features

| Tab | What it does |
|---|---|
| 🔥 Journal | Write freely. NLP model analyzes your entry for burnout risk, energy index, and positive/negative language |
| 😊 Mood | Log how you're feeling across 6 states (Heavy → Great) with context notes |
| 🫁 Breathe | Guided breathwork sessions (Box, 4-7-8, Relaxing Breath) with round timers |
| 🌱 Gratitude | Daily gratitude prompts + affirmations + self-care ideas |
| 📊 Insights | Mood history, streak tracking, energy trends |

---

## The Research Problem

**Can linguistic patterns in text reliably signal mental distress?**

This project started with a hypothesis: people in distress don't just say different *things* — they write differently. They use more self-referential language ("I", "me", "my"), more absolutist words ("always", "never", "everyone"), and tend to write longer posts. These aren't just content signals — they're behavioural and cognitive markers documented in psychology literature.

The goal was to validate these hypotheses through EDA, engineer them as features, and combine them with TF-IDF text representations in an ensemble classifier.

---

## EDA & Feature Validation (`notebooks/exploration.ipynb`)

### Step 1 — Text Length Analysis
Distressed posts (label 1) have a significantly flatter, heavier-tailed word count distribution compared to healthy posts (label 0), which cluster sharply around shorter lengths. This confirmed `feat_len` as a meaningful signal worth including.

### Step 2 — The Noise Check: Linguistic Noise vs. Emotional Signals
A side-by-side comparison of the top 20 words before and after preprocessing:

- **Raw data (noise):** dominated by stop-words and contractions — "im", "ive", "cant", "it" — which appear equally across both classes and carry zero predictive value.
- **Post-preprocessing (signal):** after lemmatization and stop-word removal, the vocabulary shifts to action and state words — "feel", "want", "think", "life", "friend" — which are critical for sentiment-based modeling.

The preprocessing reduced vocabulary size while increasing signal-to-noise ratio, ensuring the TF-IDF vectorizer focused on high-entropy features rather than grammatical structure.

### Step 3 — Behavioural Feature Correlation (The "I-Usage Theory")
Feature matrix shape after combining TF-IDF (5,000 features) + behavioural features: **27,977 × 5,003**.

Correlation heatmap of engineered features vs. target label:

| Feature | Correlation with Target | Interpretation |
|---|---|---|
| `feat_i_usage` | 0.25 | Self-referential language increases with emotional distress — validates the psychological hypothesis |
| `feat_abs_words` | 0.26 | Absolutist thinking ("always", "never", "nobody") is a documented cognitive distortion marker |
| `feat_len` | 0.22 | Distressed posts tend to be longer; more words = more self-reference opportunities |

Note: `feat_len` and `feat_i_usage` inter-correlate at 0.63 (longer posts naturally have more self-references), but since they're not perfectly correlated, each contributes unique information to the ensemble.

### Step 4 — TF-IDF Word Cloud (Mean Feature Weights)
Generated from mean TF-IDF weights across the corpus (behavioral features excluded). Dominant terms — "feel", "want", "life", "know", "think" — align with the action/state vocabulary expected post-preprocessing. The presence of distress-adjacent terms in the high-weight cluster further validates the feature space.

---

## Model Architecture

```
raw_text
  → clean_text()              # lowercase, lemmatize, stopwords (spaCy en_core_web_sm)
  → vectorizer.transform()    # TF-IDF: 5,000 features, ngram (1,2)
  → behavioural features      # feat_i_usage, feat_abs_words, feat_len (from raw text)
  → feature_df (5,003 cols)
  → ensemble.predict_proba()  # RF + LinearSVC + XGB soft vote
  → [0][1] × 100              # burnout risk %
```

**Note:** Behavioural features are extracted from *raw* text before cleaning — spaCy lemmatization strips pronouns like "I/me/my" and contractions, so we preserve them from the original input.

### Ensemble Components

| Model | Role | Voting |
|---|---|---|
| Random Forest (n=100) | Robust base; handles non-linearity in sparse TF-IDF space | Soft |
| LinearSVC + CalibratedClassifierCV | Fast linear text classifier; calibrated for probability output | Soft |
| XGBoost | Gradient boosting on residual errors from linear model | Soft |

### Final Model Performance

```
                precision    recall  f1-score   support
Healthy              0.91      0.92      0.91      2828
Distress             0.92      0.90      0.91      2768

accuracy                                0.91      5596
macro avg            0.91      0.91      0.91      5596
weighted avg         0.91      0.91      0.91      5596
```

91% F1 on a balanced test set of 5,596 samples. Confusion matrix shows 2,599 true healthy and 2,500 true distress predictions, with roughly symmetric false positive/negative rates (~229 and ~268 respectively).

---

## Tech Stack

- **Frontend:** Streamlit
- **NLP:** spaCy (`en_core_web_sm`), scikit-learn (TF-IDF, ngram)
- **ML:** scikit-learn (Random Forest, LinearSVC, CalibratedClassifierCV), XGBoost
- **Serialization:** joblib
- **EDA:** pandas, seaborn, matplotlib, WordCloud
- **Dataset:** Kaggle mental health Reddit dataset (27,977 samples, binary classification)

---

## Project Structure

```
mental-health-tracker/
├── app.py                  # Streamlit entry point
├── src/
│   ├── features.py         # Behavioural feature extraction + TF-IDF pipeline
│   ├── model.py            # Ensemble training + serialization
│   └── preprocess.py       # Text cleaning (lowercase, lemmatize, stopwords)
├── notebooks/
│   └── exploration.ipynb   # Full EDA: hypothesis validation, noise analysis, feature correlation
├── models/                 # Serialized vectorizer + ensemble (joblib)
└── requirements.txt
```

---

## Running Locally

```bash
git clone https://github.com/ksraavya/mental-health-tracker
cd mental-health-tracker
pip install -r requirements.txt
streamlit run app.py
```

---

## Disclaimer

> MindSpace is a research tool, not a clinical diagnostic.
> If you're struggling, please reach out:
> - **iCall:** 9152987821
> - **Vandrevala Foundation:** 1860-2662-345 (24×7)
