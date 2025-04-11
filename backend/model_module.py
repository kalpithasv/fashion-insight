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
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_text = "Create a marketing campaign for fashion targeting young adults with keywords: comfortable, stylish."
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=100)
print(tokenizer.decode(outputs[0]))

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
        input_text = f"Create a marketing campaign for a {product_category} targeting {demographic}. Use keywords: {sentiment_keywords}."

        inputs = gpt2_tokenizer.encode(input_text, return_tensors='pt')
        outputs = gpt2_model.generate(inputs, max_length=100, num_return_sequences=1)
        campaign_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return campaign_text

    except Exception as e:
        print("❌ Error in generate_campaign:", str(e))
        return "Error generating campaign"
