import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def is_image_file(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def send_image(api_url: str, image_path: Path, timeout: int = 30):
    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "application/octet-stream")}
        resp = requests.post(api_url, files=files, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


def extract_max_confidence(resp_json: dict) -> float:
    max_conf = 0.0
    if not isinstance(resp_json, dict):
        return 0.0
    if "detections" in resp_json and isinstance(resp_json["detections"], list):
        for d in resp_json["detections"]:
            if isinstance(d, dict):
                for key in ("confidence", "score", "probability"):
                    if key in d and isinstance(d[key], (int, float)):
                        max_conf = max(max_conf, float(d[key]))
    else:
        if "confidence" in resp_json and isinstance(resp_json["confidence"], (int, float)):
            max_conf = float(resp_json["confidence"])
        elif "predictions" in resp_json and isinstance(resp_json["predictions"], list):
            for p in resp_json["predictions"]:
                if isinstance(p, dict):
                    for key in ("confidence", "score", "probability"):
                        if key in p and isinstance(p[key], (int, float)):
                            max_conf = max(max_conf, float(p[key]))
    return max_conf


def handle_image(api_url: str, image_path: Path, threshold: float, review_dir: Path, results_dir: Path):
    logging.info("Processing %s", image_path)
    try:
        resp = send_image(api_url, image_path)
    except Exception as e:
        logging.error("Failed to send %s: %s", image_path, e)
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / (image_path.stem + ".json")
    with out_path.open("w") as f:
        json.dump({"image": str(image_path), "response": resp}, f, indent=2)

    max_conf = extract_max_confidence(resp)
    logging.info("Max confidence for %s = %.4f", image_path.name, max_conf)

    if max_conf < threshold:
        review_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_dir / (image_path.stem + ".json")
        with review_path.open("w") as f:
            json.dump({"image": str(image_path), "response": resp, "max_confidence": max_conf}, f, indent=2)
        logging.info("Saved low-confidence result to %s", review_path)


class NewImageHandler(FileSystemEventHandler):
    def __init__(self, api_url: str, threshold: float, review_dir: Path, results_dir: Path):
        self.api_url = api_url
        self.threshold = threshold
        self.review_dir = review_dir
        self.results_dir = results_dir

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if is_image_file(path):
            handle_image(self.api_url, path, self.threshold, self.review_dir, self.results_dir)


def scan_existing_and_process(watch_dir: Path, api_url: str, threshold: float, review_dir: Path, results_dir: Path):
    for p in sorted(watch_dir.iterdir() if watch_dir.exists() else []):
        if p.is_file() and is_image_file(p):
            handle_image(api_url, p, threshold, review_dir, results_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simple image prediction agent")
    parser.add_argument("--watch-dir", default="images", help="Directory to watch for images")
    parser.add_argument("--api-url", default="http://localhost:8000/predict", help="Model API predict endpoint")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for review")
    parser.add_argument("--review-dir", default="for_review", help="Where to store low-confidence results")
    parser.add_argument("--results-dir", default="results", help="Where to store all prediction outputs")
    parser.add_argument("--no-watch", action="store_true", help="Do not start filesystem watcher; only scan existing files and exit")
    args = parser.parse_args(argv)

    watch_dir = Path(args.watch_dir)
    review_dir = Path(args.review_dir)
    results_dir = Path(args.results_dir)

    logging.info("Agent starting. Watching %s; API=%s; thresh=%.2f", watch_dir, args.api_url, args.threshold)

    scan_existing_and_process(watch_dir, args.api_url, args.threshold, review_dir, results_dir)

    if args.no_watch:
        logging.info("No-watch set; exiting after initial pass")
        return

    event_handler = NewImageHandler(args.api_url, args.threshold, review_dir, results_dir)
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
