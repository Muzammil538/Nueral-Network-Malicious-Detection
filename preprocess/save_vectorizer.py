import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Create output directory if not exists
os.makedirs("preprocess", exist_ok=True)

# Load dataset
df = pd.read_csv("../dataset/full_urls.csv")
urls = df["url"].astype(str)

# Train vectorizer
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=4096)
vectorizer.fit(urls)

# Save it
with open("preprocess/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Saved: preprocess/tfidf_vectorizer.pkl")
