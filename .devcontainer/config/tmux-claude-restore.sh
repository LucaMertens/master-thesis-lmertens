#!/usr/bin/env bash
# Recreates tmux sessions/windows from the last save and resumes claude in each.
set -euo pipefail

IN="${1:-$HOME/.claude/tmux-sessions.tsv}"
[[ -f "$IN" ]] || { echo "no snapshot at $IN" >&2; exit 1; }

while IFS=$'\t' read -r sess win_idx win_name path sid; do
    if tmux has-session -t "$sess" 2>/dev/null; then
        tmux new-window -t "$sess:" -n "$win_name" -c "$path"
    else
        tmux new-session -d -s "$sess" -n "$win_name" -c "$path"
    fi
    target="$sess:$win_name"
    [[ "$sid" != "-" ]] && tmux send-keys -t "$target" "claude --resume $sid" Enter
done < "$IN"
