from pathlib import Path
from src.training_data_loader import TrainingDataLoader


def test_load_data():
    data_directory = Path("src/data/")
    loader = TrainingDataLoader(data_directory)
    loader.load_data()
