import requests
import json
import os

API_URL = "http://localhost:8000/predict"
IMAGE_PATH = "C:\\Users\\Admin\\Documents\\guard_vision\\test\\images\\test_image.jpg"

def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    with open(IMAGE_PATH, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        response = requests.post(API_URL, files=files)

    result = response.json()
    
    # Save the prediction to result.json
    with open("result.json", "w") as out:
        json.dump(result, out, indent=4)

    print("Prediction saved to result.json")
    print(result)

if __name__ == "__main__":
    main()
