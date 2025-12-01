# Image Prediction Pipeline

This project provides a minimal end-to-end workflow that sends an image to a machine-learning model running inside a Docker container and saves the prediction output as a JSON file. The entire process can also be executed automatically using GitHub Actions.

## Features

* Uses your existing Docker image: `guardvision/ml_project:v1.0`
* Automatically starts the container and exposes the API on port 8000
* Sends a test image located in the `images/` folder to the `/predict` endpoint
* Saves the model output to `result.json`
* Uploads the result as a GitHub Actions artifact
* Clean, simple, and extendable structure

---

## Project Structure

```
image-predictor/
│
├── images/
│   └── test.jpg
│
├── predictor.py
├── requirements.txt
│
└── .github/
    └── workflows/
        └── predict.yml
```

---

## How It Works

1. GitHub Actions pulls your Docker image.
2. It starts the container and exposes the ML API.
3. The `predictor.py` script sends a POST request with the image file.
4. The model returns predictions.
5. The script saves the result to `result.json`.
6. GitHub Actions uploads the file as an artifact.

---

## predictor.py

This script sends the `test.jpg` image to the model API and saves the output.

```python
import requests
import json

API_URL = "http://localhost:8000/predict"
IMAGE_PATH = "images/test.jpg"

with open(IMAGE_PATH, "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    response = requests.post(API_URL, files=files)

result = response.json()

with open("result.json", "w") as outfile:
    json.dump(result, outfile, indent=4)

print("Prediction completed:")
print(result)
```

---

## GitHub Actions Workflow

The workflow automatically runs the prediction and uploads the result.

```yaml
name: ML Prediction Test

on:
  workflow_dispatch:

jobs:
  run-prediction:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repo
      uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install Dependencies
      run: |
        pip install -r requirements.txt

    - name: Pull Your ML Docker Image
      run: |
        docker pull guardvision/ml_project:v1.0

    - name: Run Docker ML API
      run: |
        docker run -d -p 8000:8000 --name ml_api guardvision/ml_project:v1.0
        sleep 5

    - name: Run Prediction Python Script
      run: |
        python predictor.py

    - name: Upload result.json as artifact
      uses: actions/upload-artifact@v3
      with:
        name: prediction-output
        path: result.json

    - name: Stop container
      run: |
        docker stop ml_api
        docker rm ml_api
```

---

## Running Locally

### 1. Start the container

```bash
docker run -p 8000:8000 guardvision/ml_project:v1.0
```

### 2. Run the predictor script

```bash
pip install -r requirements.txt
python predictor.py
```

---

## Result Example

The output `result.json` will look like:

```json
{
  "detections": [
    {
      "class_id": 11,
      "label": "leaf_blower",
      "confidence": 0.989,
      "bbox": {
        "x1": 4.30,
        "y1": 16.23,
        "x2": 1447.89,
        "y2": 818.96
      }
    }
  ],
  "num_detections": 1
}
```

---

## Future Improvements

* Support multiple input images
* Send results to cloud storage (S3, Azure Blob, etc.)
* Add scheduled daily/weekly predictions
* Include notifications (Slack, email, Teams)

---

## License

MIT License