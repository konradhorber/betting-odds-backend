import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linprog
import warnings


class TendencyConstraintOptimizer:
    """Optimize predictions to respect the 2/3 tendency constraint"""
    
    def __init__(self, max_same_tendency_ratio: float = 2/3):
        self.max_same_tendency_ratio = max_same_tendency_ratio
    
    def optimize_matchday_predictions(self, predictions: List[Dict], 
                                    max_iterations: int = 10) -> List[Dict]:
        """Optimize all predictions for a matchday to respect constraints"""
        
        if len(predictions) <= 2:
            return predictions  # Constraint not applicable for small matchdays
        
        max_same_tendency = int(len(predictions) * self.max_same_tendency_ratio)
        
        # Count tendencies in current predictions
        tendency_counts = self._count_tendencies(predictions)
        
        # Check if constraint is violated
        max_count = max(tendency_counts.values())
        if max_count <= max_same_tendency:
            return predictions  # No violation
        
        # Find which tendency is over the limit
        violating_tendency = None
        for tendency, count in tendency_counts.items():
            if count > max_same_tendency:
                violating_tendency = tendency
                break
        
        if not violating_tendency:
            return predictions
        
        # Optimize predictions
        optimized = self._balance_tendencies(
            predictions, max_same_tendency, violating_tendency, max_iterations
        )
        
        return optimized
    
    def _count_tendencies(self, predictions: List[Dict]) -> Dict[str, int]:
        """Count how many predictions have each tendency"""
        counts = {'home_win': 0, 'draw': 0, 'away_win': 0}
        
        for pred in predictions:
            home_goals, away_goals = pred['primary_score']
            if home_goals > away_goals:
                counts['home_win'] += 1
            elif home_goals == away_goals:
                counts['draw'] += 1
            else:
                counts['away_win'] += 1
        
        return counts
    
    def _balance_tendencies(self, predictions: List[Dict], 
                          max_same_tendency: int, violating_tendency: str,
                          max_iterations: int) -> List[Dict]:
        """Balance tendencies by changing lowest confidence predictions"""
        
        optimized = predictions.copy()
        
        # Get predictions with the violating tendency
        violating_indices = []
        for i, pred in enumerate(optimized):
            home_goals, away_goals = pred['primary_score']
            current_tendency = self._get_tendency(home_goals, away_goals)
            if current_tendency == violating_tendency:
                violating_indices.append((i, pred['primary_confidence']))
        
        # Sort by confidence (lowest first)
        violating_indices.sort(key=lambda x: x[1])
        
        # Change predictions starting from lowest confidence
        changes_needed = len(violating_indices) - max_same_tendency
        changed_count = 0
        
        for i, confidence in violating_indices:
            if changed_count >= changes_needed:
                break
            
            # Try to change this prediction to next best alternative
            new_prediction = self._find_best_alternative(
                optimized[i], violating_tendency
            )
            
            if new_prediction:
                optimized[i] = new_prediction
                changed_count += 1
        
        return optimized
    
    def _get_tendency(self, home_goals: int, away_goals: int) -> str:
        """Get tendency from score"""
        if home_goals > away_goals:
            return 'home_win'
        elif home_goals == away_goals:
            return 'draw'
        else:
            return 'away_win'
    
    def _find_best_alternative(self, prediction: Dict, 
                             avoid_tendency: str) -> Optional[Dict]:
        """Find best alternative prediction avoiding specific tendency"""
        
        # Check if we have alternatives
        if not prediction.get('alternatives'):
            # Generate alternatives from tendency probabilities
            alternatives = self._generate_alternatives_from_tendencies(prediction)
        else:
            alternatives = prediction['alternatives']
        
        # Find best alternative that doesn't have the avoiding tendency
        best_alternative = None
        best_prob = 0.0
        
        for alt in alternatives:
            score = alt['score']
            prob = alt['probability']
            tendency = self._get_tendency(score[0], score[1])
            
            if tendency != avoid_tendency and prob > best_prob:
                best_alternative = alt
                best_prob = prob
        
        if best_alternative:
            return {
                'primary_score': best_alternative['score'],
                'primary_confidence': best_alternative['probability'],
                'alternatives': [a for a in alternatives 
                               if a != best_alternative],
                'tendency_probs': prediction['tendency_probs']
            }
        
        return None
    
    def _generate_alternatives_from_tendencies(self, prediction: Dict) -> List[Dict]:
        """Generate score alternatives from tendency probabilities"""
        alternatives = []
        tendency_probs = prediction.get('tendency_probs', {})
        
        # Generate representative scores for each tendency
        score_examples = {
            'home_win': [(2, 1), (1, 0), (3, 1)],
            'draw': [(1, 1), (0, 0), (2, 2)],
            'away_win': [(1, 2), (0, 1), (1, 3)]
        }
        
        for tendency, prob in tendency_probs.items():
            if prob > 0:
                for score in score_examples[tendency]:
                    alternatives.append({
                        'score': score,
                        'probability': prob / len(score_examples[tendency])
                    })
        
        # Sort by probability
        alternatives.sort(key=lambda x: x['probability'], reverse=True)
        return alternatives[:5]  # Keep top 5
    
    def calculate_expected_points_optimized(self, predictions: List[Dict],
                                          true_scores: List[Tuple[int, int]] = None) -> float:
        """Calculate expected points for optimized predictions"""
        
        if not true_scores:
            # Can't calculate without true scores
            return 0.0
        
        total_points = 0.0
        
        for pred, true_score in zip(predictions, true_scores):
            predicted_score = pred['primary_score']
            confidence = pred['primary_confidence']
            
            points = self._calculate_points(true_score, predicted_score)
            total_points += points * confidence
        
        return total_points
    
    def _calculate_points(self, true_score: Tuple[int, int], 
                         pred_score: Tuple[int, int]) -> float:
        """Calculate points according to scoring system"""
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
        true_tendency = np.sign(true_diff)
        pred_tendency = np.sign(pred_diff)
        
        if true_tendency == pred_tendency:
            return 3.0
        
        return 0.0


class LinearProgrammingOptimizer:
    """Alternative optimizer using linear programming"""
    
    def __init__(self, max_same_tendency_ratio: float = 2/3):
        self.max_same_tendency_ratio = max_same_tendency_ratio
    
    def optimize_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Optimize using linear programming approach"""
        
        if len(predictions) <= 2:
            return predictions
        
        n_matches = len(predictions)
        max_same_tendency = int(n_matches * self.max_same_tendency_ratio)
        
        try:
            return self._solve_linear_program(predictions, max_same_tendency)
        except Exception:
            # Fallback to constraint optimizer if LP fails
            optimizer = TendencyConstraintOptimizer(self.max_same_tendency_ratio)
            return optimizer.optimize_matchday_predictions(predictions)
    
    def _solve_linear_program(self, predictions: List[Dict], 
                            max_same_tendency: int) -> List[Dict]:
        """Solve optimization as linear program"""
        
        n_matches = len(predictions)
        
        # For each match, we have 3 possible tendencies
        # Variables: x[i,j] = 1 if match i gets tendency j, 0 otherwise
        # j=0: home_win, j=1: draw, j=2: away_win
        n_vars = n_matches * 3
        
        # Objective: maximize expected points
        c = self._build_objective_coefficients(predictions, n_matches)
        
        # Constraints
        A_ub, b_ub = self._build_constraints(n_matches, max_same_tendency)
        A_eq, b_eq = self._build_equality_constraints(n_matches)
        
        # Variable bounds (0 <= x <= 1)
        bounds = [(0, 1) for _ in range(n_vars)]
        
        # Solve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method='highs')
        
        if result.success:
            return self._extract_solution(predictions, result.x, n_matches)
        else:
            # Return original if optimization fails
            return predictions
    
    def _build_objective_coefficients(self, predictions: List[Dict], 
                                    n_matches: int) -> np.ndarray:
        """Build objective function coefficients"""
        c = np.zeros(n_matches * 3)
        
        tendency_map = {'home_win': 0, 'draw': 1, 'away_win': 2}
        
        for i, pred in enumerate(predictions):
            tendency_probs = pred.get('tendency_probs', {})
            for tendency, prob in tendency_probs.items():
                j = tendency_map[tendency]
                # Negative because linprog minimizes
                c[i * 3 + j] = -prob
        
        return c
    
    def _build_constraints(self, n_matches: int, 
                         max_same_tendency: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build inequality constraints (tendency limits)"""
        
        # 3 constraints: one for each tendency type
        A_ub = np.zeros((3, n_matches * 3))
        b_ub = np.full(3, max_same_tendency)
        
        for tendency_idx in range(3):
            for match_idx in range(n_matches):
                var_idx = match_idx * 3 + tendency_idx
                A_ub[tendency_idx, var_idx] = 1
        
        return A_ub, b_ub
    
    def _build_equality_constraints(self, n_matches: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build equality constraints (each match gets exactly one tendency)"""
        
        A_eq = np.zeros((n_matches, n_matches * 3))
        b_eq = np.ones(n_matches)
        
        for i in range(n_matches):
            for j in range(3):
                A_eq[i, i * 3 + j] = 1
        
        return A_eq, b_eq
    
    def _extract_solution(self, predictions: List[Dict], solution: np.ndarray,
                        n_matches: int) -> List[Dict]:
        """Extract optimized predictions from LP solution"""
        
        optimized = predictions.copy()
        tendency_names = ['home_win', 'draw', 'away_win']
        score_examples = {
            'home_win': (2, 1),
            'draw': (1, 1),
            'away_win': (1, 2)
        }
        
        for i in range(n_matches):
            # Find which tendency has highest value for this match
            tendency_values = solution[i*3:(i+1)*3]
            best_tendency_idx = np.argmax(tendency_values)
            best_tendency = tendency_names[best_tendency_idx]
            
            # Update prediction if different from current
            current_score = predictions[i]['primary_score']
            current_tendency = self._get_tendency_from_score(current_score)
            
            if current_tendency != best_tendency:
                new_score = score_examples[best_tendency]
                optimized[i] = {
                    'primary_score': new_score,
                    'primary_confidence': tendency_values[best_tendency_idx],
                    'alternatives': predictions[i].get('alternatives', []),
                    'tendency_probs': predictions[i].get('tendency_probs', {})
                }
        
        return optimized
    
    def _get_tendency_from_score(self, score: Tuple[int, int]) -> str:
        """Get tendency from score tuple"""
        home_goals, away_goals = score
        if home_goals > away_goals:
            return 'home_win'
        elif home_goals == away_goals:
            return 'draw'
        else:
            return 'away_win'