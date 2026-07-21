#!/bin/bash
# Durable BabyVLM submission watcher.
#
# Polls the GitHub repo for new [Submission] issues and, for each new one, stages
# submissions/<submitter>__<name>/ and emails the organizers the exact L40S run command.
# Runs detached (nohup/setsid) so it survives after the launching session ends.
#
# Start:  runner/watch_daemon.sh start   [interval_seconds]   (default 300)
# Status: runner/watch_daemon.sh status
# Stop:   runner/watch_daemon.sh stop
# Log:    runner/watch_daemon.sh log
set -u

REPO_ROOT="/projectnb/ivc-ml/srikumar/babyvlm_work/BabyVLM-Leaderboard-2026"
PY="/projectnb/ivc-ml/wsashawn/miniconda3/envs/llava2/bin/python"
SECRETS="/projectnb/ivc-ml/srikumar/.secrets"
STATE_DIR="/projectnb/ivc-ml/srikumar/babyvlm_work/.watch"
PIDFILE="$STATE_DIR/watch.pid"
LOGFILE="$STATE_DIR/watch.log"
INTERVAL="${2:-300}"

mkdir -p "$STATE_DIR"

_running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

case "${1:-}" in
  start)
    if _running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    export GITHUB_TOKEN="$(cat "$SECRETS/.babyvlm_gh_pat" 2>/dev/null)"
    export HF_TOKEN="$(cat "$SECRETS/.babyvlm_hf_token" 2>/dev/null)"
    cd "$REPO_ROOT" || exit 1
    # setsid fully detaches so the loop outlives this shell/session
    setsid bash -c '
      echo "[watch_daemon] started pid $$ on $(hostname) interval '"$INTERVAL"'s at $(date)"
      while true; do
        '"$PY"' -m runner.watch_github 2>&1 | grep -viE "hf_[A-Za-z0-9]|ghp_[A-Za-z0-9]"
        sleep '"$INTERVAL"'
      done
    ' >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 1
    echo "started watcher pid $(cat "$PIDFILE") on $(hostname), every ${INTERVAL}s"
    echo "log: $LOGFILE"
    ;;
  stop)
    if _running; then
      PID="$(cat "$PIDFILE")"
      # kill the whole process group (setsid child + its subshells)
      kill -TERM -- "-$PID" 2>/dev/null || kill "$PID" 2>/dev/null
      pkill -f "runner.watch_github" 2>/dev/null
      rm -f "$PIDFILE"
      echo "stopped"
    else
      echo "not running"; rm -f "$PIDFILE"
    fi
    ;;
  status)
    if _running; then echo "RUNNING (pid $(cat "$PIDFILE")) on this node"; else echo "NOT running on this node"; fi
    ;;
  log)
    tail -n 40 "$LOGFILE" 2>/dev/null || echo "(no log yet)"
    ;;
  *)
    echo "usage: $0 {start [interval]|stop|status|log}"; exit 1 ;;
esac
