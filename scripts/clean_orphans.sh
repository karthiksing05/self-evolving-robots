#!/bin/sh
# Kill this repo's multiprocessing workers whose parent run has died.
# A worker or resource tracker adopted by launchd (PPID 1) can never receive
# work again; a live run's workers still have their run as parent, so they are
# never matched. Prints a JSON systemMessage for the Claude Code hook if any die.
root=$(cd "$(dirname "$0")/.." && pwd)
pids=$(ps -Ao pid=,ppid=,command= | awk -v py="$root/.venv/bin/python -c from multiprocessing" \
    '$2 == 1 && index($0, py) { print $1 }')
[ -z "$pids" ] && exit 0
n=$(echo "$pids" | wc -l | tr -d ' ')
echo "$pids" | xargs kill
echo "{\"systemMessage\": \"clean_orphans: killed $n orphaned multiprocessing processes\"}"
