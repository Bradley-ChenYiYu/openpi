#!/bin/bash

# --- Configuration ---
HOST="140.116.154.4"
PORT_SOURCE=30707
PORT_SINK=30501
LOCAL_PORT=11434
KEY_SOURCE="$HOME/.ssh/ollama_id_ed25519"
KEY_SINK="$HOME/.ssh/openpi_id_ed25519"
RECONNECT_DELAY=3

SSH_COMMON_OPTS=(
    -o AddKeysToAgent=yes
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o TCPKeepAlive=yes
)

# --- Cleanup Function ---
cleanup() {
    echo -e "\n[!] Shutting down tunnels..."

    # Kill supervisor loops and any child ssh processes.
    for PID in "$PID1" "$PID2"; do
        [ -z "$PID" ] && continue
        pkill -TERM -P "$PID" 2>/dev/null
        kill "$PID" 2>/dev/null
    done

    wait "$PID1" "$PID2" 2>/dev/null

    # Kill the ssh-agent ONLY if we started it in this script
    if [ "$STARTED_AGENT" = true ] && [ -n "$SSH_AGENT_PID" ]; then
        echo "[!] Killing ssh-agent (PID: $SSH_AGENT_PID)..."
        kill "$SSH_AGENT_PID" 2>/dev/null
    fi

    echo "[+] Done."
    exit
}

trap cleanup SIGINT SIGTERM

start_persistent_tunnel() {
    local label="$1"
    shift

    (
        trap 'exit 0' SIGTERM SIGINT
        while true; do
            echo "[$label] Tunnel starting..."
            ssh "${SSH_COMMON_OPTS[@]}" "$@"
            exit_code=$?
            echo "[$label] Tunnel exited (code $exit_code). Reconnecting in ${RECONNECT_DELAY}s..."
            sleep "$RECONNECT_DELAY"
        done
    ) &

    LAST_TUNNEL_PID=$!
}

# --- SSH Agent & Passphrase Handling ---
echo "--- Passphrase Authentication ---"

# Check if an agent is already running
if [ -z "$SSH_AUTH_SOCK" ]; then
    echo "[*] No agent found. Starting new ssh-agent..."
    eval "$(ssh-agent -s)"
    STARTED_AGENT=true
else
    echo "[*] Using existing ssh-agent."
fi

# 2. Add keys to the agent (This is where the script waits for your input)
echo "[*] Please unlock your SSH keys:"
ssh-add "$KEY_SOURCE"
ssh-add "$KEY_SINK"

echo -e "\n--- Starting Ollama Bridge ---"

# --- Launch Tunnels ---
# Now that keys are unlocked in the agent, these background supervisor loops
# will automatically reconnect if a tunnel drops.
echo "[1/2] Connecting to Source (Port $PORT_SOURCE)..."
start_persistent_tunnel "SOURCE" -N -i "$KEY_SOURCE" -L "$LOCAL_PORT:localhost:$LOCAL_PORT" "root@$HOST" -p "$PORT_SOURCE"
PID1=$LAST_TUNNEL_PID

echo "[2/2] Connecting to Sink (Port $PORT_SINK)..."
start_persistent_tunnel "SINK" -N -i "$KEY_SINK" -R "$LOCAL_PORT:localhost:$LOCAL_PORT" "root@$HOST" -p "$PORT_SINK"
PID2=$LAST_TUNNEL_PID

# Wait a moment to ensure they didn't crash
sleep 2
if ! ps -p $PID1 > /dev/null || ! ps -p $PID2 > /dev/null; then
    echo "Error: One of the tunnels failed to stay open."
    cleanup
fi

echo "------------------------------------------------"
echo "BRIDGE ACTIVE: [Container:30707] <-> [Local] <-> [Container:30501]"
echo "Press Ctrl+C to stop."
echo "------------------------------------------------"

wait