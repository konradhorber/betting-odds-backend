
from pathlib import Path
from src.training_data_loader import TrainingDataLoader
from src.feature_engineering import FeatureEngineer


def test_feature_engineering():
    data_directory = Path("src/data/")
    loader = TrainingDataLoader(data_directory)
    df = loader.load_data()

    feature_engineer = FeatureEngineer()
    X, Y = feature_engineer.engineer_full(df)
