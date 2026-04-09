import cv2
import base64
import requests
import os

'''
This script processes a video file and sends them to an Ollama model for action labeling.

Usage:
1. Connect to ollama server:
    ssh -R 11434:localhost:11434 your_ollama_server_user@your_ollama_server_ip
2. Run this script:
    python3 generate_vid_prompt_ollama.py
'''

# Configuration
VIDEO_PATH = "/home/shared/openpi/rosbag2_2025_09_25-13_49_34.mp4"
MODEL = "kimi-k2.5:cloud"
OLLAMA_URL = "http://localhost:11434/api/generate"
TARGET_FPS = 3  # Extract 3 frames per second

def process_video(video_path, target_fps, model_name, api_url):
    print(f"Processing video: {video_path}")
    
    # 1. Extract Frames directly to memory (no temp dir needed)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    base64_images = []
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame interval to match target FPS
    # e.g., if video is 30fps and we want 3fps, we grab every 10th frame
    frame_interval = int(original_fps / target_fps)
    if frame_interval == 0: frame_interval = 1 # Handle low fps videos

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process frames at the specific interval
        if frame_count % frame_interval == 0:
            # Encode frame to JPG in memory
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64 string
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            base64_images.append(jpg_as_text)
            
        frame_count += 1

    cap.release()
    print(f"Extracted {len(base64_images)} frames.")

    # 2. Send to Ollama
    print("Sending payload to Ollama...")
    
    payload = {
        "model": model_name,
        "prompt": "Your task is to label video actions. You must output ONLY the instruction. No preamble, no 'Here is the instruction', and no extra text.",
        "stream": False,
        "images": base64_images,
        "options": {
            "temperature": 0.3
        }
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status() # Raise error for bad status codes
        
        result = response.json()
        print("\n--- Response ---")
        print(result.get("response", "No response found"))
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    process_video(VIDEO_PATH, TARGET_FPS, MODEL, OLLAMA_URL)
