import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple, List, Dict, Optional
import pickle
import os


class PointsOptimizedRegressor(BaseEstimator, RegressorMixin):
    """Custom regressor that optimizes for the 7/5/4/3 points scoring system"""
    
    def __init__(self, base_estimator=None, n_estimators=100, max_depth=10, random_state=42):
        self.base_estimator = base_estimator or RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth, 
            random_state=random_state,
            n_jobs=-1
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
    def fit(self, X, y):
        """Fit the model with points-based weighting"""
        # Create score combinations for weighting
        max_goals = 6
        self.score_combinations = [
            (i, j) for i in range(max_goals + 1) for j in range(max_goals + 1)
        ]
        
        # Calculate points-based sample weights
        sample_weights = self._calculate_sample_weights(y)
        
        # Use standard training (sample weights not supported by MultiOutputRegressor easily)
        self.estimator_ = MultiOutputRegressor(self.base_estimator)
        self.estimator_.fit(X, y)
        
        return self
    
    def _calculate_sample_weights(self, y):
        """Calculate weights based on how likely each score is to get points"""
        weights = np.ones(len(y))
        
        for i, true_score in enumerate(y):
            try:
                if isinstance(true_score, np.ndarray):
                    true_home, true_away = true_score[0], true_score[1]
                elif isinstance(true_score, (list, tuple)) and len(true_score) == 2:
                    true_home, true_away = true_score
                else:
                    continue
            except (ValueError, TypeError, IndexError):
                continue
                
            # Higher weight for more "predictable" scores that could earn points
            # Common scores get higher weights
            common_scores = [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (2,2)]
            if (int(true_home), int(true_away)) in common_scores:
                weights[i] = 1.5
            # Very high-scoring games get lower weights (harder to predict exactly)
            elif true_home + true_away > 4:
                weights[i] = 0.7
        
        return weights
        
    def predict(self, X):
        """Predict using the fitted estimator"""
        return self.estimator_.predict(X)


class ScoreDistributionPredictor:
    """ML model for predicting football score distributions"""
    
    def __init__(self, max_goals: int = 6):
        self.max_goals = max_goals
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.feature_names = []
        
        # Score combinations (0-0 to max_goals-max_goals)
        self.score_combinations = [
            (i, j) for i in range(max_goals + 1) for j in range(max_goals + 1)
        ]
        
    def _create_distribution_targets(self, y: np.ndarray) -> np.ndarray:
        """Convert exact scores to probability distributions"""
        n_samples = len(y)
        n_combinations = len(self.score_combinations)
        distributions = np.zeros((n_samples, n_combinations))
        
        for i, (home_goals, away_goals) in enumerate(y):
            # Find matching score combination
            for j, (combo_home, combo_away) in enumerate(self.score_combinations):
                if home_goals == combo_home and away_goals == combo_away:
                    distributions[i, j] = 1.0
                    break
            else:
                # If exact score not in combinations, assign to closest
                if home_goals > self.max_goals:
                    home_goals = self.max_goals
                if away_goals > self.max_goals:
                    away_goals = self.max_goals
                
                for j, (combo_home, combo_away) in enumerate(self.score_combinations):
                    if home_goals == combo_home and away_goals == combo_away:
                        distributions[i, j] = 1.0
                        break
        
        return distributions
    
    def _create_points_optimized_targets(self, y: np.ndarray) -> np.ndarray:
        """Create targets that optimize for the 7/5/4/3 points scoring system"""
        n_samples = len(y)
        n_combinations = len(self.score_combinations)
        distributions = np.zeros((n_samples, n_combinations))
        
        for i, true_score in enumerate(y):
            try:
                if isinstance(true_score, np.ndarray):
                    true_home, true_away = true_score[0], true_score[1]
                elif isinstance(true_score, (list, tuple)) and len(true_score) == 2:
                    true_home, true_away = true_score
                else:
                    # Skip invalid scores, use exact match fallback
                    for j, (pred_home, pred_away) in enumerate(self.score_combinations):
                        if j == 0:  # Default to first score combination
                            distributions[i, j] = 1.0
                            break
                    continue
            except (ValueError, TypeError, IndexError):
                # Skip invalid scores, use exact match fallback
                for j, (pred_home, pred_away) in enumerate(self.score_combinations):
                    if j == 0:  # Default to first score combination
                        distributions[i, j] = 1.0
                        break
                continue
                
            # Instead of just setting target score to 1.0, distribute probability
            # based on expected points for each possible prediction
            total_expected_points = 0.0
            points_for_each_score = []
            
            for j, (pred_home, pred_away) in enumerate(self.score_combinations):
                points = self._calculate_points_for_prediction(
                    (true_home, true_away), (pred_home, pred_away)
                )
                points_for_each_score.append(points)
                total_expected_points += points
            
            # Normalize to create a probability distribution based on points
            if total_expected_points > 0:
                for j, points in enumerate(points_for_each_score):
                    # Give higher probability to scores that would earn more points
                    distributions[i, j] = points / total_expected_points
            else:
                # Fallback: if no points possible, use exact match
                for j, (pred_home, pred_away) in enumerate(self.score_combinations):
                    if true_home == pred_home and true_away == pred_away:
                        distributions[i, j] = 1.0
                        break
        
        return distributions
    
    def _calculate_points_for_prediction(self, true_score: Tuple, pred_score: Tuple) -> float:
        """Calculate points according to the 7/5/4/3 scoring system"""
        true_home, true_away = true_score
        pred_home, pred_away = pred_score
        
        # Exact score: 7 points
        if true_home == pred_home and true_away == pred_away:
            return 7.0
        
        # Correct goal difference
        true_diff = true_home - true_away
        pred_diff = pred_home - pred_away
        
        if true_diff == pred_diff:
            if true_diff == 0:  # Draw
                return 4.0
            else:  # Win with correct goal difference
                return 5.0
        
        # Correct tendency
        if np.sign(true_diff) == np.sign(pred_diff):
            return 3.0
        
        return 0.0
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: List[str] = None) -> Dict[str, float]:
        """Train the score distribution predictor"""
        
        if feature_names:
            self.feature_names = feature_names
        
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create distribution targets optimized for points scoring
        y_dist = self._create_points_optimized_targets(y)
        
        # Use points-optimized regressor for probability distribution
        self.model = PointsOptimizedRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train on distribution targets weighted by expected points
        print("Training with points-based optimization...")
        self.model.fit(X_scaled, y_dist)
        
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_dist = self.model.predict(X_scaled)
        y_pred_scores = self.predict_scores(X)
        
        mae_home = mean_absolute_error(y[:, 0], y_pred_scores[:, 0])
        mae_away = mean_absolute_error(y[:, 1], y_pred_scores[:, 1])
        
        # Calculate expected points on training data
        expected_points = self._calculate_expected_points(y, y_pred_dist)
        
        return {
            'mae_home_goals': mae_home,
            'mae_away_goals': mae_away,
            'avg_expected_points': np.mean(expected_points)
        }
    
    def predict_distribution(self, X: np.ndarray) -> np.ndarray:
        """Predict score probability distributions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        distributions = self.model.predict(X_scaled)
        
        # Ensure probabilities are non-negative and sum to 1
        distributions = np.maximum(distributions, 0)
        row_sums = distributions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        distributions = distributions / row_sums
        
        return distributions
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely exact scores"""
        distributions = self.predict_distribution(X)
        
        scores = []
        for dist in distributions:
            best_idx = np.argmax(dist)
            home_goals, away_goals = self.score_combinations[best_idx]
            scores.append([home_goals, away_goals])
        
        return np.array(scores)
    
    def predict_with_confidence(self, X: np.ndarray) -> List[Dict]:
        """Predict scores with confidence and alternatives"""
        distributions = self.predict_distribution(X)
        
        predictions = []
        for dist in distributions:
            # Get top 3 most likely scores
            top_indices = np.argsort(dist)[-3:][::-1]
            
            prediction = {
                'primary_score': self.score_combinations[top_indices[0]],
                'primary_confidence': float(dist[top_indices[0]]),
                'alternatives': [
                    {
                        'score': self.score_combinations[idx],
                        'probability': float(dist[idx])
                    } for idx in top_indices[1:]
                ],
                'tendency_probs': self._calculate_tendency_probs(dist)
            }
            predictions.append(prediction)
        
        return predictions
    
    def _calculate_tendency_probs(self, distribution: np.ndarray) -> Dict[str, float]:
        """Calculate win/draw/loss probabilities from score distribution"""
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        
        for i, (home_goals, away_goals) in enumerate(self.score_combinations):
            prob = distribution[i]
            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_win_prob += prob
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob
        }
    
    def _calculate_expected_points(self, y_true: np.ndarray, 
                                  y_pred_dist: np.ndarray) -> np.ndarray:
        """Calculate expected points using scoring system"""
        points = np.zeros(len(y_true))
        
        for i, true_score in enumerate(y_true):
            try:
                if isinstance(true_score, np.ndarray):
                    true_home, true_away = true_score[0], true_score[1]
                elif isinstance(true_score, (list, tuple)) and len(true_score) == 2:
                    true_home, true_away = true_score
                else:
                    continue
            except (ValueError, TypeError, IndexError):
                # Skip invalid scores
                continue
            expected = 0.0
            
            for j, (pred_home, pred_away) in enumerate(self.score_combinations):
                prob = y_pred_dist[i, j]
                
                # Calculate points for this prediction
                if true_home == pred_home and true_away == pred_away:
                    # Exact score: 7 points
                    expected += 7.0 * prob
                elif self._correct_goal_difference(true_home, true_away, 
                                                 pred_home, pred_away):
                    # Correct goal difference
                    if true_home == true_away:  # Draw
                        expected += 4.0 * prob
                    else:  # Win
                        expected += 5.0 * prob
                elif self._correct_tendency(true_home, true_away, 
                                          pred_home, pred_away):
                    # Correct tendency: 3 points
                    expected += 3.0 * prob
            
            points[i] = expected
        
        return points
    
    def _correct_goal_difference(self, true_home: int, true_away: int,
                               pred_home: int, pred_away: int) -> bool:
        """Check if goal difference is correct"""
        return (true_home - true_away) == (pred_home - pred_away)
    
    def _correct_tendency(self, true_home: int, true_away: int,
                         pred_home: int, pred_away: int) -> bool:
        """Check if tendency (win/draw/loss) is correct"""
        true_tendency = np.sign(true_home - true_away)
        pred_tendency = np.sign(pred_home - pred_away)
        return true_tendency == pred_tendency
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'max_goals': self.max_goals,
            'score_combinations': self.score_combinations
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.max_goals = model_data['max_goals']
        self.score_combinations = model_data['score_combinations']
        self.is_trained = True


