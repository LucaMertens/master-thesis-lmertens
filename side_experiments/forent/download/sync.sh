#!/usr/bin/env bash
# Mirror the ForEnt Google Drive folder to $FORENT_ROOT/ForEnt, minus the WVN overlays.
# Idempotent: re-run any time to pick up files that appear on Drive later.
set -euo pipefail

SRC_ID=13An85RVF46wO96enk4qrD1-s1sfLW3p8
REMOTE="${FORENT_REMOTE:-gdrive}"
SRC="${REMOTE},root_folder_id=${SRC_ID}:"
MIRROR="${REMOTE}:forent_mirror/"
DEST="${FORENT_ROOT:-/datahome/lmertens/forent}/ForEnt"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON=(--filter-from "$HERE/forent.filter" --fast-list --transfers 16 --checkers 32
        --drive-acknowledge-abuse --stats 30s --stats-one-line)

# --size-only on copies: ~120k small files on NFS, otherwise rclone rewrites every mtime.
# Not used for `check`, which needs the MD5 comparison.
COPY=("${COMMON[@]}" --size-only)

# raw/ bags are 0 bytes on Drive but complete locally (Zenodo); --min-size 1 protects them.
min_size() { case "$1" in raw/*) echo "--min-size 1" ;; esac; }

# Recursive listings of the public folder sometimes drop files, so go one sequence at a time.
# All chunks, or only those named on the command line (e.g. `pull processed/seq_3`).
chunks() {
  if [ "$#" -gt 0 ]; then printf '%s\n' "$@"; return; fi
  rclone lsf "${SRC}raw" --dirs-only | sed 's|^|raw/|; s|/$||'
  rclone lsf "${SRC}processed" --dirs-only | sed 's|^|processed/|; s|/$||'
}

# README, setup_env.sh, calibration/, code/: kilobytes, and never download-capped.
pull_small() { rclone copy "$1" "$DEST" "${COPY[@]}" --exclude '/processed/**' --exclude '/raw/**'; }

pull() {  # pull <remote> [chunk...]
  local remote="$1"
  shift
  [ "$#" -gt 0 ] || pull_small "$remote"
  for c in $(chunks "$@"); do
    echo "== pull $c"
    rclone copy "$remote$c" "$DEST/$c" "${COPY[@]}" $(min_size "$c")
  done
}

# Copy one subtree into your own Drive, pull it, purge it. Works around Google's per-file
# download cap (a 2009-byte HTML page instead of data) and keeps Drive usage at a few GB.
stage() {
  local c="$1" list
  list=$(mktemp)
  # Missing or wrong-sized files
  rclone check "${SRC}${c}" "${DEST}/${c}" "${COMMON[@]}" $(min_size "$c") --one-way --size-only \
    --missing-on-dst "$list.miss" --differ "$list.diff" >/dev/null 2>&1 || true
  sort -u "$list.miss" "$list.diff" >"$list" 2>/dev/null || true
  if [ -s "$list" ]; then
    echo "== stage $c ($(wc -l <"$list") files)"
    rclone copy "${SRC}${c}" "${MIRROR}${c}" "${COPY[@]}" --files-from "$list" \
      --drive-server-side-across-configs
    rclone copy "${MIRROR}${c}" "${DEST}/${c}" "${COPY[@]}" --files-from "$list"
    rclone purge "${MIRROR}${c}" 2>/dev/null || true
  fi
  rm -f "$list" "$list.miss" "$list.diff"
}

cmd="${1:-pull}"
shift || true

case "$cmd" in
  pull) pull "$SRC" "$@" ;;              # straight from the public folder
  pull-mirror) pull "$MIRROR" "$@" ;;    # resume after an interrupted stage

  staged)                                # quota fallback, sequence by sequence
    [ "$#" -gt 0 ] || pull_small "$SRC"
    for c in $(chunks "$@"); do stage "$c"; done
    ;;

  # Size + MD5 against Drive, one way (extra local files, e.g. Zenodo bags, are fine).
  check)
    rc=0
    [ "$#" -gt 0 ] || rclone check "$SRC" "$DEST" "${COMMON[@]}" \
      --exclude '/processed/**' --exclude '/raw/**' --one-way || rc=1
    for c in $(chunks "$@"); do
      echo "== check $c"
      rclone check "${SRC}${c}" "$DEST/$c" "${COMMON[@]}" $(min_size "$c") --one-way || rc=1
    done
    exit $rc
    ;;

  *) echo "usage: $0 {pull|staged|pull-mirror|check} [chunk...]" >&2; exit 2 ;;
esac
