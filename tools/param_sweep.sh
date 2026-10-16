#!/usr/bin/env bash
# Parameter sweep driver.
#
# Every measurement records the md5 of the binary that produced it, and the
# binary is rebuilt from scratch-checked flags each time.  Two rounds of
# conclusions in this project were drawn from stale binaries -- once because
# make does not rebuild on a -D change, once because `make | grep -q` SIGPIPE'd
# make mid-build -- so the rule here is that a number without its binary hash is
# not a number.
#
# usage: tools/param_sweep.sh "SPLIT=8 ABLATE=0" "SPLIT=16 ABLATE=0" ...
set -u
cd "$(dirname "$0")/.."
source setenv.sh >/dev/null 2>&1

CTX=${CTX:-208,8128}
NGEN=${NGEN:-64}
REPS=${REPS:-1}
LOG=/tmp/param_sweep.log

for spec in "$@"; do
    if ! make -j8 $spec engine > "$LOG" 2>&1; then
        echo "BUILD FAILED for [$spec]"; tail -20 "$LOG"; exit 1
    fi
    HASH=$(md5sum build/engine | cut -c1-8)
    RATES=$(python3 tools/sweep.py --ngen "$NGEN" --ctx "$CTX" --reps "$REPS" 2>/dev/null \
            | tail -n +3 | awk '{printf "%s ", $2}')
    # attention is profiled REPS times too; report the best, since a single
    # sample of a 0.4 ms kernel against a 3 ms step drifts with SM clock.
    BEST=""
    for _ in $(seq "$REPS"); do
        A=$(LK_PROFILE=1 ./build/engine prof $(cat /tmp/ids8k.txt) 2>/dev/null \
            | grep -E "^attention " | awk '{print $3}')
        BEST=$(python3 -c "
import sys
v=[float(x) for x in '$BEST $A'.split() if x]
print(min(v) if v else '')")
    done
    printf '%-22s bin=%s  tok/s[%s] = %s  attn8k(min/%d)=%s ms\n' \
           "[$spec]" "$HASH" "$CTX" "$RATES" "$REPS" "${BEST:-n/a}"
done
