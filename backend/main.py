from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from model_module import predict_trends, generate_campaign

app = FastAPI()

# CORS (allow frontend to access backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CampaignRequest(BaseModel):
    product_category: str
    sentiment_keywords: str
    demographic: str

@app.get("/api/metrics")
def get_metrics():
    return {
        "average_rating": "4.2/5",
        "recommendation_rate": "86%",
        "total_reviews": 23419,
        "positive_feedback": 12846
    }

@app.post("/api/campaign")
def campaign(req: CampaignRequest):
    print(f"📩 Received request: {req}")
    try:
        # Log inputs
        print(f"Product Category: {req.product_category}")
        print(f"Sentiment Keywords: {req.sentiment_keywords}")
        print(f"Demographic: {req.demographic}")

        # Call the generate_campaign function from the model module
        text = generate_campaign(req.product_category, req.sentiment_keywords, req.demographic)
        
        # Log the generated campaign text
        print(f"📦 Generated campaign text: {text}")
        
        # Return the response
        return {"campaign_text": text}
    
    except Exception as e:
        # Log the error and return a failure message
        print(f"❌ Error: {e}")
        return {"campaign_text": "Campaign generation failed."}


@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df = predict_trends(df)
    return {"message": "File uploaded and processed.", "predictions": df.head().to_dict(orient="records")}
