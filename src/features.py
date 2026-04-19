import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def extract_behavioral_features(df):
    """
    Adds psychological markers to the dataset.
    Note: We use the 'text' (raw) column here because 'cleaned_text' 
    lost the 'I/me' pronouns during preprocessing.
    """
    # 1. Self-Reference Count
    df['i_usage'] = df['text'].str.lower().str.count(r'\bi\b|\bme\b|\bmy\b|\bmine\b')
    
    # 2. Absolute Words (Cognitive Distortion Markers)
    abs_pattern = r'\balways\b|\bnever\b|\bcompletely\b|\beveryone\b|\bnobody\b'
    df['abs_word_count'] = df['text'].str.lower().str.count(abs_pattern)
    
    # 3. Text Length (sometimes distressed posts are much longer/shorter)
    df['text_len'] = df['text'].str.len()
    
    return df

def build_feature_matrix(input_path, vectorizer_save_path):
    # Load data
    df = pd.read_csv(input_path).fillna("")
    
    print("📊 Extracting behavioral features...")
    df = extract_behavioral_features(df)
    
    print("🧮 Running TF-IDF Vectorization...")
    # max_features=5000 prevents the model from getting too bulky
    # ngram_range=(1,2) captures phrases like "not happy" as well as "happy"
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Transform cleaned text into numbers
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    
    # Save the vectorizer so the Streamlit app can use it later
    joblib.dump(vectorizer, vectorizer_save_path)
    
    # Combine TF-IDF features with our behavioral features
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df['feat_i_usage'] = df['i_usage'].values
    tfidf_df['feat_abs_words'] = df['abs_word_count'].values
    tfidf_df['feat_len'] = df['text_len'].values
    
    # Target label: usually 'label' or 'status' in the Kaggle dataset
    # Make sure your dataset column name matches here!
    y = df['label'] 
    
    return tfidf_df, y

if __name__ == "__main__":
    print("Feature logic is ready for the training pipeline.")