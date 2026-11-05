#!/usr/bin/env bash
# Manages the llama.cpp oracle (the ONLY valid correctness reference for the
# QAD checkpoint).  Runs on LK_ORACLE_GPU (see setenv.sh).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../setenv.sh" >/dev/null

PIDFILE="$LK_SCRATCH/oracle.pid"
# The timing lines that tools/oracle_bench.py parses only appear at -lv 4, so the
# default log is the verbose one: the numbers quoted in README.md have to be
# reproducible from a scripted `oracle.sh start`, not from a manual invocation.
LOGFILE="$LK_SCRATCH/oracle_verbose.log"
CTX="${LK_ORACLE_CTX:-8192}"

start() {
  if status >/dev/null 2>&1; then echo "oracle already running (pid $(cat "$PIDFILE"))"; return 0; fi
  if [ -n "$LK_ORACLE_LIBS" ]; then export LD_LIBRARY_PATH="$LK_ORACLE_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
  export CUDA_VISIBLE_DEVICES="$LK_ORACLE_GPU"
  nohup "$LK_ORACLE" \
      -m "$LK_MODEL" \
      --host 127.0.0.1 --port "$LK_ORACLE_PORT" \
      -ngl 99 -c "$CTX" \
      --metrics --no-webui -lv "${LK_ORACLE_VERBOSE:-4}" \
      >"${LK_ORACLE_LOG:-$LOGFILE}" 2>&1 &
  echo $! > "$PIDFILE"
  echo "starting oracle pid $(cat "$PIDFILE") on GPU $LK_ORACLE_GPU :$LK_ORACLE_PORT (ctx $CTX)"
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$LK_ORACLE_PORT/health" >/dev/null 2>&1; then
      echo "oracle healthy after ${i}s"; return 0
    fi
    sleep 1
  done
  echo "ERROR: oracle did not become healthy; last log:"; tail -30 "$LOGFILE"; return 1
}
stop() {
  if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null; rm -f "$PIDFILE"; echo "stopped"; else echo "not running"; fi
}
status() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  status) if status; then echo "running pid $(cat "$PIDFILE")"; else echo "not running"; fi ;;
  log) tail -"${2:-40}" "$LOGFILE" ;;
  *) echo "usage: oracle.sh {start|stop|status|log [n]}"; exit 1 ;;
esac
