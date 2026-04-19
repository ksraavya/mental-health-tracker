import re
import spacy
import pandas as pd
from nltk.corpus import stopwords

# Load the small English model for speed
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Cleans raw text for NLP: lowercase, removes URLs/Handles, 
    and reduces words to their root form (lemmatization).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase and remove URLs/User Handles (@user)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", "", text)
    
    # 2. Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # 3. Lemmatization (e.g., "running" -> "run")
    # We use SpaCy because it's faster and more accurate than NLTK for this
    doc = nlp(text)
    
    # 4. Remove Stopwords (the, is, at) as they carry little sentiment info
    tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]
    
    return " ".join(tokens)

def prepare_data(input_file, output_file):
    print("🚀 Starting Preprocessing...")
    df = pd.read_csv(input_file)
    
    # Apply cleaning to the text column
    # The 'text' column in the Kaggle dataset contains the posts
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Save the processed data for Phase 2 (Feature Engineering)
    df.to_csv(output_file, index=False)
    print(f"✅ Preprocessing Complete! Saved to: {output_file}")

if __name__ == "__main__":
    # Test run
    sample = "I am feeling so burnt out and tired today @university... http://help.com"
    print(f"Sample Before: {sample}")
    print(f"Sample After:  {clean_text(sample)}")