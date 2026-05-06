
#!/bin/bash

set -euo pipefail

VIDEO=""
TIME=""
INDEX=""
MODEL="gemma4:31b"
PROMPT="Describe the scene in this frame."

usage() {
		cat <<USAGE
Usage: $0 -v <video> (-t <time> | -n <frame_index>) [-m <model>] [-p <prompt>]

Options:
	-v <video>         Path to video file
	-t <time>          Timestamp (seconds or HH:MM:SS) to extract the frame
	-n <frame_index>   Frame index (0-based) to extract
	-m <model>         Ollama model (default: $MODEL)
	-p <prompt>        Prompt text to send to the model
	-h                 Show this help
USAGE
		exit 1
}

while getopts ":v:t:n:m:p:h" opt; do
	case ${opt} in
		v ) VIDEO=$OPTARG ;;
		t ) TIME=$OPTARG ;;
		n ) INDEX=$OPTARG ;;
		m ) MODEL=$OPTARG ;;
		p ) PROMPT=$OPTARG ;;
		h ) usage ;;
		\? ) echo "Invalid Option: -$OPTARG" 1>&2; usage ;;
		: ) echo "Invalid option: -$OPTARG requires an argument" 1>&2; usage ;;
	esac
done

if [ -z "$VIDEO" ]; then
	echo "Error: video path is required." >&2
	usage
fi

if [ -z "$TIME" ] && [ -z "$INDEX" ]; then
	echo "Error: either -t <time> or -n <frame_index> is required." >&2
	usage
fi

if [ ! -f "$VIDEO" ]; then
	echo "Error: video file '$VIDEO' not found." >&2
	exit 1
fi

TMP_DIR=$(mktemp -d)
OUT_IMG="$TMP_DIR/frame.jpg"

echo "Extracting frame to $OUT_IMG..."

if [ -n "$TIME" ]; then
	ffmpeg -y -ss "$TIME" -i "$VIDEO" -frames:v 1 "$OUT_IMG" -loglevel error
elif [ -n "$INDEX" ]; then
	ffmpeg -y -i "$VIDEO" -vf "select=eq(n\,$INDEX)" -vframes 1 "$OUT_IMG" -loglevel error
fi

if [ ! -f "$OUT_IMG" ]; then
	echo "Failed to extract frame." >&2
	rm -rf "$TMP_DIR"
	exit 1
fi

B64=$(base64 -w 0 < "$OUT_IMG")

echo "Sending frame to Ollama model $MODEL..."

cat <<EOF | curl -s -X POST http://localhost:11434/api/generate -d @- | python3 -c "import sys, json; print(json.load(sys.stdin).get('response', ''))"
{
	"model": "$MODEL",
	"prompt": "$PROMPT",
	"stream": false,
	"images": ["$B64"]
}
EOF

rm -rf "$TMP_DIR"
