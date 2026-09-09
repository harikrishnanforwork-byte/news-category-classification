from pathlib import Path
import re
import html
import unicodedata

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

PROJECT_DIR = Path(__file__).resolve().parent
CATEGORIES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def clean_text(value):
    text = html.unescape(str(value)).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", text)).strip()


def load_training_data():
    source = PROJECT_DIR / "data" / "train.csv"
    frame = pd.read_csv(source)
    text = frame["Title"].fillna("").astype(str) + " " + frame["Description"].fillna("").astype(str)
    labels = frame["Class Index"].astype(int).map(lambda value: CATEGORIES.get(value - 1))
    data = pd.DataFrame({"text": text.map(clean_text), "label": labels}).dropna().drop_duplicates()
    return data[data["text"].str.len() >= 20].reset_index(drop=True)


training_data = load_training_data()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True)
features = vectorizer.fit_transform(training_data["text"])
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42)
model.fit(features, training_data["label"])

app = FastAPI(title="NewsLens API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    text: str


@app.get("/api/health")
def health():
    return {"status": "online", "articles": len(training_data), "features": len(vectorizer.vocabulary_)}


@app.post("/api/predict")
def predict(request: PredictionRequest):
    text = clean_text(request.text)
    if len(text) < 5:
        return {"error": "Enter a headline or article excerpt first."}
    probabilities = model.predict_proba(vectorizer.transform([text]))[0]
    ranked = probabilities.argsort()[::-1]
    return {
        "category": model.classes_[ranked[0]],
        "confidence": round(float(probabilities[ranked[0]]), 4),
        "signals": [
            {"category": model.classes_[index], "confidence": round(float(probabilities[index]), 4)}
            for index in ranked[1:]
        ],
    }
