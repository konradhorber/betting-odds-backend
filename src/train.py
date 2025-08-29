from training_data_loader import TrainingDataLoader
from feature_engineering import FeatureEngineer
import src.models
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
logging.getLogger().setLevel(logging.ERROR)


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
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
model.save(Path("models/trained_model.pkl"))
