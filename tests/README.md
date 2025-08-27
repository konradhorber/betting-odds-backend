# Prediction API Evaluation

This module provides comprehensive testing and evaluation functionality for the Betting Odds Backend prediction API.

## Directory Structure

```
tests/
├── __init__.py                    # Package init
├── README.md                      # This file
├── evaluate_predictions.py        # Main evaluation module
└── evaluation/                    # Test data and results
    ├── test_set.csv              # Original test dataset
    ├── aggregated_test_set.csv   # Processed odds data
    └── evaluation_results.csv    # Latest test results
```

## Usage

### Quick Evaluation

Run a complete evaluation against your running server:

```bash
# Make sure your server is running first
python -m uvicorn src.main:app --reload --port 8000

# Then run evaluation
python tests/evaluate_predictions.py
```

### Custom Evaluation

```python
from tests.evaluate_predictions import run_evaluation

# Evaluate with custom test file and server
metrics = run_evaluation(
    test_file="path/to/your/test_data.csv",
    server_url="http://localhost:8000"
)

print(f"Overall accuracy: {metrics['overall_accuracy']:.1f}%")
```

## Components

### OddsAggregator
- Aggregates betting odds from multiple bookmakers using median values
- Supports H2H (1X2), Totals (Over/Under), and Asian Handicap markets
- Handles both pre-closing and closing odds

### PredictionEvaluator
- Makes API calls to your running prediction server
- Converts historical data to API-compatible format
- Calculates prediction accuracy using standard football metrics

### EvaluationReporter
- Generates detailed match-by-match reports
- Provides summary statistics
- Saves results to CSV for further analysis

## Evaluation Metrics

The evaluation uses standard football prediction scoring:

- **Exact Score Match**: 7 points
- **Correct Goal Difference**: 4 points (draws) or 5 points (wins)
- **Correct Tendency**: 3 points (win/draw/loss)
- **Wrong Prediction**: 0 points

## Test Data Format

Your test CSV file should include these columns:

**Required:**
- `HomeTeam`, `AwayTeam`: Team names
- `FTHG`, `FTAG`: Full-time home/away goals
- `FTR`: Full-time result (H/D/A)

**H2H Odds (multiple bookmakers):**
- `B365H/D/A`, `BFDH/DD/DA`, `BMGMH/MD/MA`, etc.

**Totals Odds:**
- `B365>2.5/<2.5`, `P>2.5/<2.5`, `BFE>2.5/<2.5`

**Asian Handicap Odds:**
- `B365AHH/AHA`, `PAHH/AHA`, `BFEAHH/AHA`
- `AHh`: Handicap line

## Example Output

```
============================================================
PREDICTION API EVALUATION
============================================================
Testing against server: http://localhost:8000
Server status: {'model_type': 'ml', 'model_path': 'models/trained_model.pkl', 'model_exists': True}
✅ Loaded 9 matches from test set
✅ Received 9/9 valid predictions

================================================================================
SUMMARY STATISTICS
================================================================================
Total Matches: 9
Valid Predictions: 9
Exact Matches: 0 (0.0%)
Correct Goal Differences: 0 (0.0%)
Correct Tendencies: 4 (44.4%)
Overall Accuracy: 44.4%
Total Points: 12
Average Points per Match: 1.33
================================================================================
```

## Notes

- The server must be running before evaluation
- Results are automatically saved to `tests/evaluation/evaluation_results.csv`
- Odds aggregation happens automatically if not pre-processed
- The evaluation works with the trained ML model