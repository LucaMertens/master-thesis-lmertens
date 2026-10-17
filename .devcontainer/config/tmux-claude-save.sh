#!/usr/bin/env bash
# Snapshots tmux window -> claude session id mappings, for restore after a container restart.
# Claude Code tracks pid/sessionId/tmux-pane itself in ~/.claude/sessions/<pid>.json.
set -euo pipefail

[[ -f /.dockerenv ]] || exit 0
command -v tmux >/dev/null 2>&1 || exit 0
tmux list-sessions >/dev/null 2>&1 || exit 0

OUT="${1:-$HOME/.claude/tmux-sessions.tsv}"
TMP="$(mktemp)"

for f in "$HOME"/.claude/sessions/*.json; do
    [[ -e "$f" ]] || continue
    pid=$(jq -r '.pid' "$f")
    kill -0 "$pid" 2>/dev/null || continue
    sid=$(jq -r '.sessionId' "$f")
    tmux_ref=$(jq -r '.tmux // empty' "$f")
    [[ -n "$tmux_ref" ]] || continue
    pane_id="${tmux_ref##*.}"
    info=$(tmux display-message -p -t "$pane_id" $'#{session_name}\t#{window_index}\t#{window_name}\t#{pane_current_path}' 2>/dev/null) || continue
    printf '%s\t%s\n' "$info" "$sid"
done > "$TMP"

mv "$TMP" "$OUT"
