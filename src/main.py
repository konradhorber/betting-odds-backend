from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from src.models.neg_binomial import NBExpectedPointsModel
from src.models.model import Model
from src.feature_engineering import FeatureEngineer
import pandas as pd
from scripts.download_model import download_model


class Odds(BaseModel):
    Div: str
    Date: str
    HomeTeam: str
    AwayTeam: str
    AvgH: float
    AvgD: float
    AvgA: float
    AvgOver2_5: float
    AvgUnder2_5: float
    AHh: float
    AvgAHH: float
    AvgAHA: float


class Prediction(BaseModel):
    home_goals: int
    away_goals: int
    home_team: str
    away_team: str
    date: str


feature_engineer: FeatureEngineer = FeatureEngineer()
model: Model = NBExpectedPointsModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_engineer
    path = Path("models/trained_model.pkl")
    if path.exists():
        model.load(path)
    else:
        success = download_model()
        if not success:
            raise RuntimeError("Model not found and download failed") 
        model.load(path)
    feature_engineer = FeatureEngineer()
    yield


app = FastAPI(
    title="Bundesliga Score Predictor",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=Prediction)
async def predict_score(odds: Odds):
    """Predict score using ML model with betting market data"""
    X = feature_engineer.engineer_X(pd.DataFrame([odds.model_dump()]))
    try:
        prediction = model.predict(X)
        result = Prediction(
            home_goals=int(prediction[0, 0]),
            away_goals=int(prediction[0, 1]),
            home_team=odds.HomeTeam,
            away_team=odds.AwayTeam,
            date=odds.Date,
        )
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")
