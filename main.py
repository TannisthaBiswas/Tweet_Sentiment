from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Request schema
class TweetRequest(BaseModel):
    text: str

# Root
@app.get("/")
def read_root():
    return {"message": "Tweet Sentiment API is running."}

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    vector = vectorizer.transform([request.text])
    prediction = model.predict(vector)[0]
    label = "positive" if prediction == 1 else "negative"
    return {"sentiment": label}
