from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Outcome(BaseModel):
    name: str
    price: float
    point: Optional[float] = None

class Market(BaseModel):
    key: str
    last_update: str
    outcomes: List[Outcome]

class MatchData(BaseModel):
    home_team: str
    away_team: str
    markets: List[Market]

class PredictedScore(BaseModel):
    home_goals: int
    away_goals: int
    confidence: float

# Initialize predictor (will be loaded on startup)
predictor = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global predictor
    try:
        from .models.score_predictor import ScorePredictor
    except ImportError:
        from models.score_predictor import ScorePredictor
    
    try:
        # Always train on startup to ensure fresh model
        print("Training model on startup...")
        predictor = ScorePredictor(auto_train=True)
        # Force retrain to ensure model is fresh
        predictor.train_model(retrain=True)
        print("Score predictor initialized and trained successfully")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        predictor = None
    
    yield
    
    # Shutdown
    print("Shutting down prediction service")

app = FastAPI(title="Bundesliga Score Predictor", version="1.0.0", lifespan=lifespan)

@app.post("/predict", response_model=PredictedScore)
async def predict_score(match_data: MatchData):
    """Predict score using ML model with betting market data"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        prediction = predictor.predict_single_match(match_data)
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-matchday", response_model=List[PredictedScore])
async def predict_matchday(matches: List[MatchData]):
    """Predict entire matchday with 2/3 tendency constraint optimization"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    if len(matches) == 0:
        raise HTTPException(status_code=400, detail="No matches provided")
    
    try:
        predictions = predictor.predict_matchday(matches)
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matchday prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the current prediction model"""
    if predictor is None:
        return {"error": "Predictor not initialized"}
    
    try:
        info = predictor.get_model_info()
        return info
        
    except Exception as e:
        return {"error": f"Failed to get model info: {str(e)}"}

