#!/usr/bin/env bash
# Interleaved A/B benchmark.
#
# Sequential sweeps confound the parameter with whatever drifts over the run
# (SM clock, L2 state, page placement).  SPLIT=192 measured 6% faster than
# SPLIT=128 on throughput while its attention kernel measured 22% SLOWER, which
# cannot both be true; the only way to tell signal from drift is to alternate
# the binaries within the same time window.
#
# usage: tools/abbench.sh BIN1 BIN2 [BIN3 ...]   (env: CTX, NGEN, REPS)
set -u
cd "$(dirname "$0")/.."
source setenv.sh >/dev/null 2>&1

CTX=${CTX:-8128}
NGEN=${NGEN:-64}
REPS=${REPS:-5}

BINS=("$@")
N=${#BINS[@]}

echo "interleaved: ${N} binaries x ${REPS} rounds, ctx=${CTX} ngen=${NGEN}"
for r in $(seq "$REPS"); do
    for b in "${BINS[@]}"; do
        export LK_ENGINE="$b"
        R=$(python3 tools/sweep.py --ngen "$NGEN" --ctx "$CTX" 2>/dev/null \
            | tail -1 | awk '{print $2}')
        printf '%s\t%s\t%s\n' "$r" "$(basename "$b")" "$R"
    done
done | tee /tmp/abbench.tsv | awk -F'\t' '{printf "  round %s  %-16s %s tok/s\n", $1, $2, $3}'

echo "--- summary ---"
awk -F'\t' '
{ n[$2]++; s[$2]+=$3; if(min[$2]==""||$3<min[$2])min[$2]=$3; if($3>max[$2])max[$2]=$3 }
END { for (k in n) printf "  %-16s mean=%.2f  min=%.2f  max=%.2f  spread=%.2f%%\n",
      k, s[k]/n[k], min[k], max[k], 100*(max[k]-min[k])/min[k] }' /tmp/abbench.tsv | sort
