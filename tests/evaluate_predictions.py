"""
Evaluation module for testing prediction API performance against historical data.

This module provides functionality to:
1. Aggregate betting odds from multiple bookmakers
2. Test predictions against a running server
3. Evaluate prediction accuracy using standard football metrics
4. Generate detailed performance reports
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path


class OddsAggregator:
    """Handles aggregation of betting odds from multiple bookmakers"""
    
    def __init__(self):
        pass
    
    def aggregate_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate betting odds by taking median across different bookmakers
        for H2H, spreads (Asian Handicap), and totals markets
        """
        result_df = df.copy()
        
        # H2H (1X2) odds columns - pre-closing
        h2h_home_cols = ['B365H', 'BFDH', 'BMGMH', 'BVH', 'BWH', 'CLH', 'LBH', 'PSH', 'BFEH']
        h2h_draw_cols = ['B365D', 'BFDD', 'BMGMD', 'BVD', 'BWD', 'CLD', 'LBD', 'PSD', 'BFED']
        h2h_away_cols = ['B365A', 'BFDA', 'BMGMA', 'BVA', 'BWA', 'CLA', 'LBA', 'PSA', 'BFEA']
        
        # H2H (1X2) odds columns - closing
        h2h_home_close_cols = ['B365CH', 'BFDCH', 'BMGMCH', 'BVCH', 'BWCH', 'CLCH', 'LBCH', 'PSCH', 'BFECH']
        h2h_draw_close_cols = ['B365CD', 'BFDCD', 'BMGMCD', 'BVCD', 'BWCD', 'CLCD', 'LBCD', 'PSCD', 'BFECD']
        h2h_away_close_cols = ['B365CA', 'BFDCA', 'BMGMCA', 'BVCA', 'BWCA', 'CLCA', 'LBCA', 'PSCA', 'BFECA']
        
        # Totals (Over/Under 2.5) odds columns
        totals_over_cols = ['B365>2.5', 'P>2.5', 'BFE>2.5']
        totals_under_cols = ['B365<2.5', 'P<2.5', 'BFE<2.5']
        totals_over_close_cols = ['B365C>2.5', 'PC>2.5', 'BFEC>2.5']
        totals_under_close_cols = ['B365C<2.5', 'PC<2.5', 'BFEC<2.5']
        
        # Asian Handicap odds columns
        ah_home_cols = ['B365AHH', 'PAHH', 'BFEAHH']
        ah_away_cols = ['B365AHA', 'PAHA', 'BFEAHA']
        ah_home_close_cols = ['B365CAHH', 'PCAHH', 'BFECAHH']
        ah_away_close_cols = ['B365CAHA', 'PCAHA', 'BFECAHA']
        
        def calculate_median_odds(row, columns):
            """Calculate median odds from available bookmaker columns"""
            values = []
            for col in columns:
                if col in row.index and pd.notna(row[col]):
                    values.append(row[col])
            return np.median(values) if values else np.nan
        
        # Calculate aggregated odds
        result_df['median_h2h_home'] = result_df.apply(lambda row: calculate_median_odds(row, h2h_home_cols), axis=1)
        result_df['median_h2h_draw'] = result_df.apply(lambda row: calculate_median_odds(row, h2h_draw_cols), axis=1)
        result_df['median_h2h_away'] = result_df.apply(lambda row: calculate_median_odds(row, h2h_away_cols), axis=1)
        
        result_df['median_h2h_home_close'] = result_df.apply(lambda row: calculate_median_odds(row, h2h_home_close_cols), axis=1)
        result_df['median_h2h_draw_close'] = result_df.apply(lambda row: calculate_median_odds(row, h2h_draw_close_cols), axis=1)
        result_df['median_h2h_away_close'] = result_df.apply(lambda row: calculate_median_odds(row, h2h_away_close_cols), axis=1)
        
        result_df['median_totals_over'] = result_df.apply(lambda row: calculate_median_odds(row, totals_over_cols), axis=1)
        result_df['median_totals_under'] = result_df.apply(lambda row: calculate_median_odds(row, totals_under_cols), axis=1)
        result_df['median_totals_over_close'] = result_df.apply(lambda row: calculate_median_odds(row, totals_over_close_cols), axis=1)
        result_df['median_totals_under_close'] = result_df.apply(lambda row: calculate_median_odds(row, totals_under_close_cols), axis=1)
        
        result_df['median_ah_home'] = result_df.apply(lambda row: calculate_median_odds(row, ah_home_cols), axis=1)
        result_df['median_ah_away'] = result_df.apply(lambda row: calculate_median_odds(row, ah_away_cols), axis=1)
        result_df['median_ah_home_close'] = result_df.apply(lambda row: calculate_median_odds(row, ah_home_close_cols), axis=1)
        result_df['median_ah_away_close'] = result_df.apply(lambda row: calculate_median_odds(row, ah_away_close_cols), axis=1)
        
        # Get handicap value
        result_df['handicap_value'] = result_df['AHh'].fillna(result_df.get('AHCh', np.nan))
        
        return result_df
    
    def prepare_for_prediction_api(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare aggregated data for prediction API"""
        prediction_df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG',
                           'median_h2h_home', 'median_h2h_draw', 'median_h2h_away',
                           'median_h2h_home_close', 'median_h2h_draw_close', 'median_h2h_away_close',
                           'median_totals_over', 'median_totals_under',
                           'median_totals_over_close', 'median_totals_under_close',
                           'median_ah_home', 'median_ah_away',
                           'median_ah_home_close', 'median_ah_away_close',
                           'handicap_value']].copy()
        return prediction_df


class PredictionEvaluator:
    """Handles evaluation of prediction accuracy"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
    
    def create_match_data_for_api(self, row: pd.Series) -> dict:
        """Convert a row from aggregated dataset to JSON format for API call"""
        markets = []
        
        # Create H2H market if odds are available
        if pd.notna(row['median_h2h_home']) and pd.notna(row['median_h2h_draw']) and pd.notna(row['median_h2h_away']):
            h2h_market = {
                "key": "h2h",
                "last_update": datetime.now().isoformat(),
                "outcomes": [
                    {"name": "home", "price": float(row['median_h2h_home'])},
                    {"name": "draw", "price": float(row['median_h2h_draw'])},
                    {"name": "away", "price": float(row['median_h2h_away'])}
                ]
            }
            markets.append(h2h_market)
        
        # Create totals market if odds are available
        if pd.notna(row['median_totals_over']) and pd.notna(row['median_totals_under']):
            totals_market = {
                "key": "totals",
                "last_update": datetime.now().isoformat(),
                "outcomes": [
                    {"name": "over", "price": float(row['median_totals_over']), "point": 2.5},
                    {"name": "under", "price": float(row['median_totals_under']), "point": 2.5}
                ]
            }
            markets.append(totals_market)
        
        # Create spread/Asian Handicap market if odds are available
        if pd.notna(row['median_ah_home']) and pd.notna(row['median_ah_away']) and pd.notna(row['handicap_value']):
            spread_market = {
                "key": "spreads",
                "last_update": datetime.now().isoformat(),
                "outcomes": [
                    {"name": "home", "price": float(row['median_ah_home']), "point": float(row['handicap_value'])},
                    {"name": "away", "price": float(row['median_ah_away']), "point": float(-row['handicap_value'])}
                ]
            }
            markets.append(spread_market)
        
        return {
            "home_team": row['HomeTeam'],
            "away_team": row['AwayTeam'],
            "markets": markets
        }
    
    def make_prediction_request(self, match_data: dict) -> Optional[dict]:
        """Make API request to predict endpoint"""
        try:
            response = requests.post(f"{self.server_url}/predict", json=match_data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
    
    def check_server_status(self) -> bool:
        """Check if server is running and model is loaded"""
        try:
            response = requests.get(f"{self.server_url}/model-info", timeout=10)
            response.raise_for_status()
            model_info = response.json()
            print(f"Server status: {model_info}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Server not accessible: {e}")
            return False
    
    def evaluate_prediction_accuracy(self, predictions: List[dict], actual_results: List[dict]) -> dict:
        """Evaluate prediction accuracy using standard football prediction scoring"""
        if len(predictions) != len(actual_results):
            raise ValueError("Number of predictions must match actual results")
        
        exact_matches = 0
        correct_goal_differences = 0
        correct_tendencies = 0
        total_points = 0
        
        for pred, actual in zip(predictions, actual_results):
            if pred is None:  # Skip failed predictions
                continue
                
            pred_home = pred['home_goals']
            pred_away = pred['away_goals']
            actual_home = actual['FTHG']
            actual_away = actual['FTAG']
            
            # Exact score match (7 points)
            if pred_home == actual_home and pred_away == actual_away:
                exact_matches += 1
                total_points += 7
            # Correct goal difference (4 points for draw, 5 for win)
            elif (pred_home - pred_away) == (actual_home - actual_away):
                correct_goal_differences += 1
                if actual_home == actual_away:  # Draw
                    total_points += 4
                else:  # Win
                    total_points += 5
            # Correct tendency only (3 points)
            elif np.sign(pred_home - pred_away) == np.sign(actual_home - actual_away):
                correct_tendencies += 1
                total_points += 3
        
        valid_predictions = len([p for p in predictions if p is not None])
        if valid_predictions == 0:
            return {'error': 'No valid predictions received'}
        
        return {
            'total_matches': len(actual_results),
            'valid_predictions': valid_predictions,
            'exact_matches': exact_matches,
            'correct_goal_differences': correct_goal_differences,
            'correct_tendencies': correct_tendencies,
            'exact_match_rate': exact_matches / valid_predictions * 100,
            'goal_difference_rate': correct_goal_differences / valid_predictions * 100,
            'tendency_rate': correct_tendencies / valid_predictions * 100,
            'overall_accuracy': (exact_matches + correct_goal_differences + correct_tendencies) / valid_predictions * 100,
            'total_points': total_points,
            'average_points': total_points / valid_predictions
        }


class EvaluationReporter:
    """Handles generation of evaluation reports"""
    
    def __init__(self, output_dir: str = "tests/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def print_detailed_results(self, df: pd.DataFrame, predictions: List[dict], 
                              accuracy_metrics: dict, model_type: str = "Unknown"):
        """Print detailed match-by-match results"""
        
        print("\n" + "="*80)
        print(f"DETAILED MATCH RESULTS ({model_type.upper()} MODEL)")
        print("="*80)
        
        for i, (_, row) in enumerate(df.iterrows()):
            pred = predictions[i]
            if pred is None:
                print(f"{row['Date']:<10} {row['HomeTeam']:<15} vs {row['AwayTeam']:<15}")
                print("  PREDICTION FAILED")
                print()
                continue
                
            actual_result = 'H' if row['FTHG'] > row['FTAG'] else 'D' if row['FTHG'] == row['FTAG'] else 'A'
            pred_result = 'H' if pred['home_goals'] > pred['away_goals'] else 'D' if pred['home_goals'] == pred['away_goals'] else 'A'
            
            # Determine points scored
            points = 0
            result_type = ""
            
            if pred['home_goals'] == row['FTHG'] and pred['away_goals'] == row['FTAG']:
                points = 7
                result_type = "EXACT"
            elif (pred['home_goals'] - pred['away_goals']) == (row['FTHG'] - row['FTAG']):
                points = 4 if row['FTHG'] == row['FTAG'] else 5
                result_type = "GOAL DIFF"
            elif np.sign(pred['home_goals'] - pred['away_goals']) == np.sign(row['FTHG'] - row['FTAG']):
                points = 3
                result_type = "TENDENCY"
            
            status = "✓" if points > 0 else "✗"
            
            print(f"{row['Date']:<10} {row['HomeTeam']:<15} vs {row['AwayTeam']:<15}")
            print(f"  Predicted: {pred['home_goals']}-{pred['away_goals']} ({pred_result}) | "
                  f"Actual: {row['FTHG']}-{row['FTAG']} ({actual_result}) | "
                  f"Points: {points} ({result_type}) {status}")
            print(f"  Confidence: {pred['confidence']:.3f}")
            
            # Show key odds used
            print(f"  Odds - H2H: {row['median_h2h_home']:.2f}/{row['median_h2h_draw']:.2f}/{row['median_h2h_away']:.2f} | "
                  f"O/U 2.5: {row['median_totals_over']:.2f}/{row['median_totals_under']:.2f}")
            print()
        
        print("="*80)
        print("SUMMARY STATISTICS - 7/5/4/3 BUNDESLIGA SCORING SYSTEM")
        print("="*80)
        if 'error' in accuracy_metrics:
            print(f"ERROR: {accuracy_metrics['error']}")
        else:
            print(f"Total Matches: {accuracy_metrics['total_matches']}")
            print(f"Valid Predictions: {accuracy_metrics['valid_predictions']}")
            print()
            print("SCORING BREAKDOWN:")
            print(f"  Exact Matches (7 pts): {accuracy_metrics['exact_matches']} ({accuracy_metrics['exact_match_rate']:.1f}%)")
            print(f"  Correct Goal Differences (4/5 pts): {accuracy_metrics['correct_goal_differences']} ({accuracy_metrics['goal_difference_rate']:.1f}%)")
            print(f"  Correct Tendencies (3 pts): {accuracy_metrics['correct_tendencies']} ({accuracy_metrics['tendency_rate']:.1f}%)")
            print(f"  Overall Accuracy: {accuracy_metrics['overall_accuracy']:.1f}%")
            print()
            print("POINTS PERFORMANCE:")
            print(f"  Total Points: {accuracy_metrics['total_points']}")
            print(f"  Average Points per Match: {accuracy_metrics['average_points']:.2f}")
            max_possible = accuracy_metrics['valid_predictions'] * 7
            efficiency = (accuracy_metrics['total_points'] / max_possible * 100) if max_possible > 0 else 0
            print(f"  Points Efficiency: {efficiency:.1f}% (vs max possible {max_possible} points)")
        print("="*80)
    
    def save_results(self, df: pd.DataFrame, predictions: List[dict], 
                    filename: str = "prediction_results.csv") -> str:
        """Save detailed results to CSV file"""
        results_data = []
        for i, (_, row) in enumerate(df.iterrows()):
            pred = predictions[i]
            result_row = row.to_dict()
            if pred:
                result_row['predicted_home_goals'] = pred['home_goals']
                result_row['predicted_away_goals'] = pred['away_goals']
                result_row['prediction_confidence'] = pred['confidence']
            else:
                result_row['predicted_home_goals'] = None
                result_row['predicted_away_goals'] = None
                result_row['prediction_confidence'] = None
            results_data.append(result_row)
        
        results_df = pd.DataFrame(results_data)
        filepath = self.output_dir / filename
        results_df.to_csv(filepath, index=False)
        return str(filepath)


def run_evaluation(test_file: str = "tests/evaluation/test_set.csv", 
                  server_url: str = "http://localhost:8000") -> dict:
    """
    Main evaluation function
    
    Args:
        test_file: Path to test dataset CSV file
        server_url: URL of the running prediction server
    
    Returns:
        Dictionary containing accuracy metrics
    """
    print("="*60)
    print("PREDICTION API EVALUATION")
    print("="*60)
    
    # Initialize components
    aggregator = OddsAggregator()
    evaluator = PredictionEvaluator(server_url)
    reporter = EvaluationReporter()
    
    # Check server status
    print(f"Testing against server: {server_url}")
    if not evaluator.check_server_status():
        print("❌ Server is not running or not accessible. Please start the server first.")
        return {'error': 'Server not accessible'}
    
    # Load and aggregate test data
    print(f"Loading test data from: {test_file}")
    try:
        df = pd.read_csv(test_file)
        print(f"✅ Loaded {len(df)} matches from test set")
    except FileNotFoundError:
        print(f"❌ Test file not found: {test_file}")
        return {'error': 'Test file not found'}
    
    # Check if data is already aggregated
    if 'median_h2h_home' not in df.columns:
        print("📊 Aggregating odds from multiple bookmakers...")
        df = aggregator.aggregate_odds(df)
        df = aggregator.prepare_for_prediction_api(df)
        
        # Save aggregated data
        aggregated_file = reporter.output_dir / "aggregated_test_set.csv"
        df.to_csv(aggregated_file, index=False)
        print(f"✅ Aggregated odds saved to: {aggregated_file}")
    else:
        print("✅ Using pre-aggregated odds data")
    
    # Make predictions
    print("🔮 Making predictions via API...")
    predictions = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  Predicting match {i+1}/{len(df)}: {row['HomeTeam']} vs {row['AwayTeam']}")
        match_data = evaluator.create_match_data_for_api(row)
        prediction = evaluator.make_prediction_request(match_data)
        predictions.append(prediction)
    
    valid_predictions = len([p for p in predictions if p is not None])
    print(f"✅ Received {valid_predictions}/{len(predictions)} valid predictions")
    
    # Prepare actual results
    actual_results = [
        {'FTHG': row['FTHG'], 'FTAG': row['FTAG'], 'FTR': row['FTR']}
        for _, row in df.iterrows()
    ]
    
    # Evaluate accuracy
    print("📈 Evaluating prediction accuracy...")
    accuracy_metrics = evaluator.evaluate_prediction_accuracy(predictions, actual_results)
    
    # Generate reports
    model_type = "ML" if valid_predictions > 0 else "Unknown"
    reporter.print_detailed_results(df, predictions, accuracy_metrics, model_type)
    
    # Save results
    results_file = reporter.save_results(df, predictions, "evaluation_results.csv")
    print(f"💾 Detailed results saved to: {results_file}")
    
    return accuracy_metrics


if __name__ == "__main__":
    # Run evaluation with default settings
    metrics = run_evaluation()