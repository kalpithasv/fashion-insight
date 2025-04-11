import pandas as pd
import numpy as np
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df.columns = df.columns.str.strip()  # <-- this fixes your KeyError

# DEBUG: Check column names
print("Columns in dataset:", df.columns.tolist())

# Use correct column names
df['Review'] = df['Review'].fillna('')
df['Rating'] = df['Rating'].fillna(0)
df['Recommended IND'] = df['Recommended IND'].fillna(0)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in text.split() if w not in stop_words])

df['Processed_Review'] = df['Review'].apply(clean_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['Processed_Review'])

# Target
y = df['Recommended IND']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(rf, "rf_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
