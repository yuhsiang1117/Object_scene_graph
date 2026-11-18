#!/usr/bin/env bash
# Supervise the full v1 val run until every episode is scored.
#
#   bash scripts/run_full_split.sh            # start (or adopt a run in flight)
#   bash scripts/run_full_split.sh --status   # what is left, nothing else
#
# WHY THIS EXISTS. The 2000-episode run takes ~60 h and has been killed twice by
# things that have nothing to do with the agent: a non-reentrant model server
# returning HTTP 500 under a concurrent eval, and a session teardown that took
# the whole process tree. Each time the fix was mechanical -- work out which
# episodes are missing, restart -- so it should not need a person.
#
# The remaining set is re-derived from episodes.jsonl before every attempt, and
# the runner appends, so a restart can neither re-run a scored episode nor skip
# an unscored one no matter how the previous attempt died.
#
# WHAT IT CANNOT DO. It is a process in this container, so it dies with the
# container or with a session teardown that sweeps user processes -- which is
# what happened on 2026-09-13. Surviving that needs a supervisor outside the
# session (the container's restart policy, or an entrypoint hook). On restart,
# run this script again; it picks up wherever the record left off.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-outputs/full_final_resume}
PRIOR=${PRIOR:-outputs/full_final}          # earlier partial records to count as done
EXPERIMENT=${EXPERIMENT:-final_sensor}
LOG=${LOG:-$OUT.log}
LOCK=${LOCK:-/tmp/osg_full_split.lock}
ARGFILE=${ARGFILE:-/tmp/osg_full_split_arg.txt}
MAX_STALL=${MAX_STALL:-3}                   # consecutive attempts that score nothing
BACKOFF=${BACKOFF:-60}
SERVERS="13182 13183 13184 13185 13186"

remaining() { python scripts/full_split_remaining.py --write "$ARGFILE" "$PRIOR" "$OUT"; }

if [ "${1:-}" = "--status" ]; then
    n=$(remaining) || exit 2
    echo "$n episodes remaining; override written to $ARGFILE"
    exit 0
fi

# One supervisor at a time, or two of them race to run the same episodes.
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another supervisor already holds $LOCK -- not starting a second"
    exit 1
fi

servers_up() {
    for p in $SERVERS; do
        [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://localhost:$p/" 2>/dev/null)" = "000" ] \
            && return 1
    done
    return 0
}

ensure_servers() {
    servers_up && return 0
    echo "$(date -Is) model servers down -- starting them"
    bash scripts/serve_perception.sh >/dev/null 2>&1
    for _ in $(seq 60); do
        servers_up && { echo "$(date -Is) servers up"; return 0; }
        sleep 5
    done
    echo "$(date -Is) servers did not come up"
    return 1
}

mkdir -p "$OUT"
stall=0
attempt=0

while true; do
    n=$(remaining) || { echo "cannot compute the remaining set"; exit 2; }
    if [ "$n" -eq 0 ]; then
        echo "$(date -Is) COMPLETE -- every episode scored"
        python - <<'PY'
import json, numpy as np, os
rows=[]
for d in (os.environ.get("PRIOR","outputs/full_final"), os.environ.get("OUT","outputs/full_final_resume")):
    f=os.path.join(d,"episodes.jsonl")
    if os.path.isfile(f): rows += [json.loads(l) for l in open(f) if l.strip()]
n=len(rows); p=sum(r["success"] for r in rows)/n
print(f"  {n} episodes  SR {100*p:.1f}%  SPL {np.mean([r['spl'] for r in rows]):.3f}")
PY
        exit 0
    fi

    # Adopt a run already in flight rather than starting a rival.
    if pgrep -f "run_eval.py \+experiment=$EXPERIMENT" >/dev/null; then
        echo "$(date -Is) a run is already in flight ($n left) -- waiting for it"
        while pgrep -f "run_eval.py \+experiment=$EXPERIMENT" >/dev/null; do sleep 60; done
    else
        ensure_servers || { sleep "$BACKOFF"; continue; }
        attempt=$((attempt + 1))
        echo "$(date -Is) attempt $attempt: $n episodes remaining"
        python scripts/run_eval.py "+experiment=$EXPERIMENT" "$(cat "$ARGFILE")" \
            eval.behaviour_log=true eval.save_viz=false "output_dir=$OUT" >> "$LOG" 2>&1
        echo "$(date -Is) attempt $attempt exited rc=$?"
    fi

    after=$(remaining)
    if [ "$after" -ge "$n" ]; then
        stall=$((stall + 1))
        echo "$(date -Is) no episode was scored ($stall/$MAX_STALL)"
        if [ "$stall" -ge "$MAX_STALL" ]; then
            echo "$(date -Is) giving up: $MAX_STALL attempts in a row scored nothing." \
                 "Something is broken that restarting will not fix -- see $LOG."
            exit 3
        fi
        sleep "$BACKOFF"
    else
        stall=0
    fi
done
