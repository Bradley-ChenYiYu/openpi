#!/bin/bash

VIDEO="rosbag2_2025_09_25-13_49_34.mp4"
MODEL="kimi-k2.5:cloud"

# 1. Create a temporary directory for frames
TMP_DIR=$(mktemp -d)
echo "Extracting frames to $TMP_DIR..."

# Extract 1 frame per second. Adjust 'fps=1' as needed.
ffmpeg -i "$VIDEO" -vf "fps=1" "$TMP_DIR/frame_%03d.jpg" -loglevel error

# 2. Convert each image to Base64 and wrap in quotes for a JSON array
# This loop builds a string like: "base64_1", "base64_2", "base64_3"
IMAGES_JSON=""
for img in "$TMP_DIR"/*.jpg; do
    B64=$(base64 -w 0 < "$img")
    IMAGES_JSON+="\"$B64\","
done

# Remove the trailing comma
IMAGES_JSON=${IMAGES_JSON%?}

# 3. Send to Ollama
echo "Sending multiple frames to Ollama..."

cat <<EOF | curl -X POST http://172.17.0.2:11434/api/generate -d @-
{
  "model": "$MODEL",
  "prompt": "These are sequential frames from a ROS bag. Describe the scene and the camera motion.",
  "stream": false,
  "images": [$IMAGES_JSON]
}
EOF

# 4. Cleanup
rm -rf "$TMP_DIR"