#!/bin/sh
# Kill this repo's multiprocessing workers and System 2 calls whose parent run has died.
# One adopted by launchd (PPID 1) can never report back, and a call still spends
# the plan; a live run's have their run as parent, so they are never matched.
# Prints a JSON systemMessage for the Claude Code hook if any die.
root=$(cd "$(dirname "$0")/.." && pwd)
pids=$(ps -Ao pid=,ppid=,command= | awk -v py="$root/.venv/bin/python -c from multiprocessing" -v runs="$root/runs/evolve/" \
    '$2 == 1 && (index($0, py) || (/claude -p / && index($0, runs))) { print $1 }')
[ -z "$pids" ] && exit 0
n=$(echo "$pids" | wc -l | tr -d ' ')
echo "$pids" | xargs kill
echo "{\"systemMessage\": \"clean_orphans: killed $n orphaned processes\"}"
