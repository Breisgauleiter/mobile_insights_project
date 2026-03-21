#!/usr/bin/env python3
"""
Dummy Highlight Detector for Mobile Insights MVP.

This script reads a video file and returns a list of "highlight" timestamps.  
In a real implementation, replace the random selection with computer-vision models.
"""
import argparse
import random
import os

import cv2


def detect_highlights(video_path, num_highlights=5):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    
    # Generate random highlight timestamps for demonstration
    highlights = sorted([round(random.uniform(0, max(0, duration_sec - 1)), 2) for _ in range(num_highlights)])
    cap.release()
    return highlights


def main():
    parser = argparse.ArgumentParser(description="Dummy Highlight Detector")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--n", type=int, default=5, help="Number of highlight timestamps to return")
    args = parser.parse_args()
    
    highlights = detect_highlights(args.video, args.n)
    for t in highlights:
        print(f"Highlight at {t} seconds")


if __name__ == "__main__":
    main()
