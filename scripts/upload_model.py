#!/usr/bin/env python3
"""Upload trained model to Google Cloud Storage"""

import sys
from pathlib import Path


def upload_model(bucket_name: str):
    """Upload model to GCS bucket"""
    try:
        from google.cloud import storage

        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Upload model file
        model_path = Path("models/trained_model.pkl")

        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            return False

        blob = bucket.blob("trained_model.pkl")

        print(f"Uploading {model_path} to gs://{bucket_name}/trained_model.pkl")
        blob.upload_from_filename(model_path)
        print("Model uploaded successfully!")
        return True
   
    except ImportError:
        print("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
        return False
    except Exception as e:
        print(f"Error uploading model: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/upload_model.py BUCKET_NAME")
        sys.exit(1)

    bucket_name = sys.argv[1]
    success = upload_model(bucket_name)
    sys.exit(0 if success else 1)