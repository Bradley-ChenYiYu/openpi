#!/bin/bash
set -Eeuo pipefail

on_error() {
    local exit_code="$?"
    echo "Error: commit training loop failed at line ${BASH_LINENO[0]} (exit code: ${exit_code})."
    exit "$exit_code"
}

trap on_error ERR

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")

cd "$repo_root"

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <commit> [<commit> ...]"
    echo "Runs scripts/run_training_from_raw_data.sh once for each commit in order."
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree must be clean before switching commits."
    exit 1
fi

original_ref=$(git rev-parse --abbrev-ref HEAD)
if [[ "$original_ref" == "HEAD" ]]; then
    original_ref=$(git rev-parse HEAD)
    restore_command=(git switch --detach --quiet "$original_ref")
else
    restore_command=(git switch --quiet "$original_ref")
fi

restore_repo() {
    "${restore_command[@]}"
}

trap restore_repo EXIT

total_commits=$#
commit_index=0

for commit in "$@"; do
    commit_index=$((commit_index + 1))
    resolved_commit=$(git rev-parse --verify "$commit^{commit}")

    echo ""
    echo "=== [$commit_index/$total_commits] Running training for commit $resolved_commit ==="

    git switch --detach --quiet "$resolved_commit"
    TRAIN_IN_BACKGROUND=0 bash scripts/run_training_from_raw_data.sh
done
