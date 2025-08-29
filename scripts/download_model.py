#!/usr/bin/env python3
"""Download trained model from Google Cloud Storage"""

import os
import sys
from pathlib import Path
from google.cloud import storage


def download_model():
    """Download model from GCS bucket if MODEL_BUCKET_NAME is set"""
    bucket_name = os.getenv('MODEL_BUCKET_NAME')
    
    if not bucket_name:
        print("MODEL_BUCKET_NAME not set, skipping model download")
        return False
    
    try:        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Download model file
        model_path = Path("models/trained_model.pkl")
        model_path.parent.mkdir(exist_ok=True)
        
        blob = bucket.blob("trained_model.pkl")
        
        if blob.exists():
            print(f"Downloading model from gs://{bucket_name}/trained_model.pkl")
            blob.download_to_filename(model_path)
            print(f"Model downloaded to {model_path}")
            return True
        else:
            print(f"Model file not found in bucket gs://{bucket_name}")
            return False
            
    except ImportError:
        print("google-cloud-storage not installed, skipping model download")
        return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    if not success:
        print("Model download failed or skipped - will train on startup")
    sys.exit(0)  # Don't fail startup if model download fails