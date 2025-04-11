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
    print("📩 /api/campaign hit")
    print("🛬 Received request body:")
    print("   ➤ product_category:", req.product_category)
    print("   ➤ sentiment_keywords:", req.sentiment_keywords)
    print("   ➤ demographic:", req.demographic)
    
    # Dummy response
    return {
        "campaign_text": f"Generated dummy campaign for {req.product_category}, targeting {req.demographic}, using keywords: {req.sentiment_keywords}"
    }


@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df = predict_trends(df)
    return {"message": "File uploaded and processed.", "predictions": df.head().to_dict(orient="records")}
