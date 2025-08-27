import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import os
import logging
from datetime import datetime

from .feature_engineering import FeatureEngineer
from .ml_model import ScoreDistributionPredictor
from .constraint_optimizer import TendencyConstraintOptimizer

try:
    from ..main import MatchData, PredictedScore
except ImportError:
    from main import MatchData, PredictedScore


class ScorePredictor:
    """Main interface for score prediction with constraint optimization"""
    
    def __init__(self, model_path: str = "models/trained_model.pkl", 
                 auto_train: bool = True):
        self.model_path = model_path
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.constraint_optimizer = TendencyConstraintOptimizer()
        
        # Initialize ML model
        self.predictor = ScoreDistributionPredictor()
        
        # Try to load existing model or train new one
        if os.path.exists(model_path):
            try:
                self.predictor.load_model(model_path)
                logging.info(f"Loaded existing model from {model_path}")
            except Exception as e:
                logging.warning(f"Failed to load model: {e}")
                if auto_train:
                    self._train_model()
                else:
                    raise RuntimeError("Model loading failed and auto_train is disabled")
        else:
            if auto_train:
                logging.info("No existing model found, training new model...")
                self._train_model()
            else:
                raise RuntimeError("No model found and auto_train is disabled")
    
    def _train_model(self):
        """Internal method to train the model"""
        try:
            # Load historical data
            print("Loading historical data...")
            historical_df = self.feature_engineer.load_historical_data()
            
            if historical_df.empty:
                raise ValueError("No historical data found")
            
            print(f"Loaded {len(historical_df)} historical matches")
            
            # Prepare training data
            X, y = self.feature_engineer.prepare_training_data(historical_df)
            
            if len(X) == 0:
                raise ValueError("No valid training data after feature extraction")
            
            print(f"Prepared {len(X)} training samples with {X.shape[1]} features")
            
            # Train model
            print("Training model...")
            metrics = self.predictor.train(X, y, self.feature_engineer.feature_names)
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.predictor.save_model(self.model_path)
            
            print(f"Model trained and saved to {self.model_path}")
            print(f"Training metrics: {metrics}")
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}")
    
    def predict_single_match(self, match_data: MatchData) -> PredictedScore:
        """Predict score for a single match"""
        # Extract features
        features = self.feature_engineer.extract_market_features(match_data)
        
        # Create feature vector and predict
        feature_vector = self.feature_engineer.create_feature_vector(features)
        prediction = self.predictor.predict_with_confidence(
            feature_vector.reshape(1, -1)
        )[0]
        
        # Convert to PredictedScore format
        home_goals, away_goals = prediction['primary_score']
        confidence = prediction['primary_confidence']
        
        return PredictedScore(
            home_goals=int(home_goals),
            away_goals=int(away_goals),
            confidence=float(confidence)
        )
    
    def predict_matchday(self, matches: List[MatchData]) -> List[PredictedScore]:
        """Predict scores for entire matchday with constraint optimization"""
        
        # Get initial predictions
        initial_predictions = []
        
        for match_data in matches:
            features = self.feature_engineer.extract_market_features(match_data)
            feature_vector = self.feature_engineer.create_feature_vector(features)
            pred = self.predictor.predict_with_confidence(
                feature_vector.reshape(1, -1)
            )[0]
            initial_predictions.append(pred)
        
        # Apply constraint optimization
        optimized_predictions = self.constraint_optimizer.optimize_matchday_predictions(
            initial_predictions
        )
        
        # Convert to PredictedScore format
        results = []
        for pred in optimized_predictions:
            home_goals, away_goals = pred['primary_score']
            confidence = pred['primary_confidence']
            
            results.append(PredictedScore(
                home_goals=int(home_goals),
                away_goals=int(away_goals),
                confidence=float(confidence)
            ))
        
        return results
    
    def train_model(self, retrain: bool = False) -> Dict[str, float]:
        """Train/retrain the ML model using historical data"""
        
        if retrain:
            # Create new model instance for retraining
            self.predictor = ScoreDistributionPredictor()
            return self._train_model_external()
        elif not self.predictor.is_trained:
            return self._train_model_external()  
        else:
            return {'message': 'Model already trained, use retrain=True to force retrain'}
    
    def _train_model_external(self) -> Dict[str, float]:
        """External training method for API endpoint"""
        # Load historical data
        print("Loading historical data...")
        historical_df = self.feature_engineer.load_historical_data()
        
        if historical_df.empty:
            raise ValueError("No historical data found")
        
        print(f"Loaded {len(historical_df)} historical matches")
        
        # Prepare training data
        X, y = self.feature_engineer.prepare_training_data(historical_df)
        
        if len(X) == 0:
            raise ValueError("No valid training data after feature extraction")
        
        print(f"Prepared {len(X)} training samples with {X.shape[1]} features")
        
        # Train model
        print("Training model...")
        metrics = self.predictor.train(X, y, self.feature_engineer.feature_names)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.predictor.save_model(self.model_path)
        
        print(f"Model saved to {self.model_path}")
        print(f"Training metrics: {metrics}")
        
        return metrics
    
    def get_prediction_details(self, match_data: MatchData) -> Dict:
        """Get detailed prediction information including alternatives"""
        
        features = self.feature_engineer.extract_market_features(match_data)
        feature_vector = self.feature_engineer.create_feature_vector(features)
        prediction = self.predictor.predict_with_confidence(
            feature_vector.reshape(1, -1)
        )[0]
        
        # Add extracted features to response
        result = {
            'match': {
                'home_team': match_data.home_team,
                'away_team': match_data.away_team
            },
            'prediction': prediction,
            'extracted_features': features,
            'model_type': 'ml',
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def evaluate_predictions(self, test_matches: List[MatchData],
                           true_scores: List[Tuple[int, int]]) -> Dict[str, float]:
        """Evaluate prediction accuracy on test data"""
        
        if len(test_matches) != len(true_scores):
            raise ValueError("Number of matches and true scores must match")
        
        predictions = []
        for match in test_matches:
            pred = self.predict_single_match(match)
            predictions.append((pred.home_goals, pred.away_goals))
        
        # Calculate metrics
        exact_matches = 0
        correct_tendencies = 0
        correct_goal_diffs = 0
        total_points = 0
        
        for (pred_home, pred_away), (true_home, true_away) in zip(predictions, true_scores):
            # Exact match
            if pred_home == true_home and pred_away == true_away:
                exact_matches += 1
                total_points += 7
            # Goal difference
            elif (pred_home - pred_away) == (true_home - true_away):
                correct_goal_diffs += 1
                if true_home == true_away:  # Draw
                    total_points += 4
                else:  # Win
                    total_points += 5
            # Tendency
            elif (np.sign(pred_home - pred_away) == np.sign(true_home - true_away)):
                correct_tendencies += 1
                total_points += 3
        
        n_matches = len(predictions)
        return {
            'exact_match_rate': exact_matches / n_matches,
            'goal_difference_rate': correct_goal_diffs / n_matches,
            'tendency_rate': correct_tendencies / n_matches,
            'average_points': total_points / n_matches,
            'total_points': total_points
        }
    
    def get_model_info(self) -> Dict[str, Union[str, bool, int]]:
        """Get information about the current model"""
        
        feature_count = 0
        if hasattr(self.predictor, 'feature_names') and self.predictor.feature_names:
            feature_count = len(self.predictor.feature_names)
        elif self.feature_engineer.feature_names:
            feature_count = len(self.feature_engineer.feature_names)
        
        info = {
            'model_type': 'ml',
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path),
            'is_trained': self.predictor.is_trained,
            'feature_count': feature_count
        }
        
        if hasattr(self.predictor, 'score_combinations'):
            info['max_goals'] = self.predictor.max_goals
            info['score_combinations'] = len(self.predictor.score_combinations)
        
        return info


class PredictionBatch:
    """Helper class for batch predictions with caching"""
    
    def __init__(self, predictor: ScorePredictor):
        self.predictor = predictor
        self.cache = {}
    
    def predict_multiple_matchdays(self, matchdays: List[List[MatchData]]) -> List[List[PredictedScore]]:
        """Predict multiple matchdays efficiently"""
        
        results = []
        for matchday in matchdays:
            matchday_key = self._create_matchday_key(matchday)
            
            if matchday_key in self.cache:
                results.append(self.cache[matchday_key])
            else:
                predictions = self.predictor.predict_matchday(matchday)
                self.cache[matchday_key] = predictions
                results.append(predictions)
        
        return results
    
    def _create_matchday_key(self, matchday: List[MatchData]) -> str:
        """Create cache key for matchday"""
        key_parts = []
        for match in matchday:
            key_parts.append(f"{match.home_team}-{match.away_team}")
        return "|".join(sorted(key_parts))