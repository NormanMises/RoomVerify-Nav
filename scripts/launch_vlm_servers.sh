#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
set -euo pipefail

# Start four model servers in a detached tmux window using the active environment.

command -v tmux >/dev/null || { echo "Error: tmux is not on PATH." >&2; exit 1; }

VLFM_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VLFM_PYTHON=${VLFM_PYTHON:-$(command -v python)}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export SAM_PORT=${SAM_PORT:-12183}
export YOLOV7_PORT=${YOLOV7_PORT:-12184}
VLM_TMUX_SESSION=${VLM_TMUX_SESSION:-vlm-servers-$GROUNDING_DINO_PORT}
VLM_TMUX_WINDOW=${VLM_TMUX_WINDOW:-vlm}

export LD_LIBRARY_PATH="$(dirname -- "$(dirname -- "$VLFM_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VLM_NUM_THREADS=${VLM_NUM_THREADS:-4}
export OMP_NUM_THREADS=$VLM_NUM_THREADS MKL_NUM_THREADS=$VLM_NUM_THREADS OPENBLAS_NUM_THREADS=$VLM_NUM_THREADS

env_exports="export"
for name in VLFM_PYTHON LD_LIBRARY_PATH OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS MOBILE_SAM_CHECKPOINT GROUNDING_DINO_CONFIG GROUNDING_DINO_WEIGHTS \
            CLASSES_PATH GROUNDING_DINO_PORT BLIP2ITM_PORT SAM_PORT YOLOV7_PORT; do
    printf -v env_exports '%s %s=%q' "$env_exports" "$name" "${!name}"
done

if tmux has-session -t "=$VLM_TMUX_SESSION" 2>/dev/null; then
    yolo_pane=$(tmux new-window -d -t "=$VLM_TMUX_SESSION" -n "$VLM_TMUX_WINDOW" -c "$VLFM_REPO_ROOT" -P -F '#{pane_id}')
else
    yolo_pane=$(tmux new-session -d -s "$VLM_TMUX_SESSION" -n "$VLM_TMUX_WINDOW" -c "$VLFM_REPO_ROOT" -x 240 -y 60 -P -F '#{pane_id}')
fi

gdino_pane=$(tmux split-window -d -h -p 50 -t "$yolo_pane" -c "$VLFM_REPO_ROOT" -P -F '#{pane_id}')
sam_pane=$(tmux split-window -d -v -p 50 -t "$yolo_pane" -c "$VLFM_REPO_ROOT" -P -F '#{pane_id}')
blip_pane=$(tmux split-window -d -v -p 50 -t "$gdino_pane" -c "$VLFM_REPO_ROOT" -P -F '#{pane_id}')

run_service() {
    local pane_id=$1 module=$2 port=$3 service_command
    printf -v service_command 'cd %q && %s && %q -m %q --port %q' \
        "$VLFM_REPO_ROOT" "$env_exports" "$VLFM_PYTHON" "$module" "$port"
    tmux send-keys -t "$pane_id" -l "$service_command"
    tmux send-keys -t "$pane_id" Enter
}
run_service "$yolo_pane" vlfm.vlm.yolov7 "$YOLOV7_PORT"
run_service "$gdino_pane" vlfm.vlm.grounding_dino "$GROUNDING_DINO_PORT"
run_service "$sam_pane" vlfm.vlm.sam "$SAM_PORT"
run_service "$blip_pane" vlfm.vlm.blip2itm "$BLIP2ITM_PORT"

printf 'Created window %s in tmux session %s. Model loading may take 90 seconds.\n' "$VLM_TMUX_WINDOW" "$VLM_TMUX_SESSION"
printf 'View server output: tmux attach -t %s\n' "$VLM_TMUX_SESSION"
