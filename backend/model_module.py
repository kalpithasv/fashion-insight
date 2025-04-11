import pandas as pd
import re
import joblib
import torch
from nltk.corpus import stopwords
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained models
rf_model = joblib.load(os.path.join(BASE_DIR, "models", "rf_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

# Load GPT-2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Clean the review text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Predict recommendations using Random Forest
def predict_trends(df):
    df = df.copy()
    df.columns = df.columns.str.strip()  # Clean column names
    if 'Review' not in df.columns:
        df['Review'] = df.iloc[:, -1]  # Fallback for CSVs with unknown structure

    df['Review'] = df['Review'].fillna('')
    df['Processed_Review'] = df['Review'].apply(clean_text)
    X = vectorizer.transform(df['Processed_Review'])
    df['Predicted_Recommendation'] = rf_model.predict(X)
    return df

# Generate campaign using GPT-2
def generate_campaign(product_category, sentiment_keywords, demographic):
    try:
        prompt = (
            f"Our {product_category} collection is perfect for {demographic}. "
            f"Featuring {sentiment_keywords}, it's designed for both style and comfort."
        )
        inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
        output = gpt2_model.generate(
            inputs,
            max_length=50,
            top_p=0.9,
            temperature=0.9,
            do_sample=True,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
        result = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        return result
    except Exception as e:
        print(f"🔥 Error generating campaign: {e}")
        return "Shop our latest styles, designed just for you."

