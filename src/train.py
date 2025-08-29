from training_data_loader import TrainingDataLoader
from feature_engineering import FeatureEngineer
import src.models
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging
logging.getLogger().setLevel(logging.ERROR)


def calculate_points(predicted_home, predicted_away, actual_home, actual_away):
    """
    Calculate points based on the scoring system:
    - 7 points: Exact score match
    - 5 points: Correct goal difference for wins
    - 4 points: Correct goal difference for draws
    - 3 points: Correct tendency only
    - 0 points: Incorrect prediction
    """
    pred_diff = predicted_home - predicted_away
    actual_diff = actual_home - actual_away
    
    # Exact score match
    if predicted_home == actual_home and predicted_away == actual_away:
        return 7
    
    # Correct goal difference
    if pred_diff == actual_diff:
        if actual_diff == 0:  # Draw
            return 4
        else:  # Win
            return 5
    
    # Correct tendency
    if (pred_diff > 0 and actual_diff > 0) or \
       (pred_diff < 0 and actual_diff < 0) or \
       (pred_diff == 0 and actual_diff == 0):
        return 3
    
    # Incorrect prediction
    return 0


directory = Path("src/data/")
loader = TrainingDataLoader(directory=directory)
feature_engineer = FeatureEngineer()
# model = models.ExpectedPointsModel(K=5, alpha=0.6, epochs=25, batch_size=512)
# model = models.PoissonGoalsRegressor(max_iter=2000)
model = src.models.neg_binomial.NBExpectedPointsModel(
    K=5,
    epochs=40,
    lr=2e-3,
    weight_decay=1e-4,
    hidden=128,
)

raw_data = loader.load_data(b1_only=False)
X, y = feature_engineer.engineer_full(raw_data)
full_set = pd.concat([X, y], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

model.fit(X_train, y_train)
model.save(Path("models/trained_model.pkl"))

# Model evaluation
print("Evaluating model on test set...")
predictions = model.predict(X_test)

# Calculate points for each prediction
points = []
for i in range(len(predictions)):
    pred_home = int(round(predictions[i, 0]))
    pred_away = int(round(predictions[i, 1]))
    actual_home = int(y_test.iloc[i]['FTHG'])
    actual_away = int(y_test.iloc[i]['FTAG'])
    
    score = calculate_points(pred_home, pred_away, actual_home, actual_away)
    points.append(score)

# Show 40 predictions vs actuals
print("\n40 Predictions vs Actuals:")
print("Pred | Actual | Points")
print("-" * 23)
for i in range(min(40, len(predictions))):
    pred_home = int(round(predictions[i, 0]))
    pred_away = int(round(predictions[i, 1]))
    actual_home = int(y_test.iloc[i]['FTHG'])
    actual_away = int(y_test.iloc[i]['FTAG'])
    score = points[i]
    
    print(f"{pred_home}-{pred_away}  | {actual_home}-{actual_away}    | {score}")

# Calculate and display average score
average_score = np.mean(points)
print(f"\nAverage Score on Test Set: {average_score:.2f} points")

