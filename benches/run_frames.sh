#!/usr/bin/env bash
# Runs the frame benchmark (benches/frame.rs): first a verification of every
# structure, mode and scene with seed 0, appended to OUT's `.verify.jsonl`,
# then the timed runs, each in its own process, the structures in reverse
# order on odd rounds. Appends one JSON line per run to OUT, with a hash of
# the sources; benches/summarize_frames.py turns them into tables.
# VERIFY_ARGS adds arguments to the verification, e.g. "--verify-every 1";
# RUN_ARGS to every run, e.g. "--pool" (not --seed or --verify, which this
# script sets). Every line also gets an id of this invocation, since rounds
# start at 0 in each, so that a summary pairs runs of the same invocation.
#
#   benches/run_frames.sh OUT [ROUNDS] [SEEDS...]
#
# STRUCTURES, MODES and SCENES override the lists below, e.g.
#   SCENES=10k MODES=pairs benches/run_frames.sh out.jsonl 1 0
set -euo pipefail
cd "$(dirname "$0")/.."

out=$1
rounds=${2:-2}
shift $(($# < 2 ? $# : 2))
seeds=("$@")
[ ${#seeds[@]} -gt 0 ] || seeds=(0 1 2)

read -r -a structures <<< "${STRUCTURES:-HaloOverflowBoundGrid SpanOverflowBoundGrid HierarchicalLooseGrid UniformGrid TwoLayerGrid SortAndSweep ParryBvh/rebuild-binned ParryBvh/rebuild-ploc ParryBvh/refit ParryBvh/refit-opt ParryBvh/refit-opt-m5 ParryBvh/refit-opt-m25 ParryBvh/reinsert-m25}"
read -r -a modes <<< "${MODES:-search pairs}"
read -r -a scenes <<< "${SCENES:-10k 160k}"

bin=$(cargo bench --bench frame --no-run 2>&1 | sed -n 's/.*Executable .*(\(.*\))/\1/p' | tail -1)
[ -n "$bin" ] || { echo "frame bench did not build" >&2; exit 1; }
# Files moved away but still in the index are listed too; skip them.
source=$(git ls-files -co --exclude-standard -- src benches Cargo.toml | sort |
    while IFS= read -r file; do if [ -f "$file" ]; then cat "$file"; fi; done | sha1sum | cut -c1-12)
verified="${out%.jsonl}.verify.jsonl"
invocation="$source-$(date +%s)-$$"

# The sweeps are not spatial indexes: their searches scan every item.
skip() {
    { [ "$2" = search ] && [[ "$1" == SortAndSweep* || "$1" == SpatialJoinSweep/* ]]; } ||
        # The hierarchical structures have no levels for the tall worlds.
        { [[ "$3" == tall* ]] && [[ "$1" =~ ^(Halo|Span)OverflowBoundGrid$|^HierarchicalLooseGrid$ ]]; }
}

for scene in "${scenes[@]}"; do
    for mode in "${modes[@]}"; do
        for structure in "${structures[@]}"; do
            skip "$structure" "$mode" "$scene" && continue
            # `Null` finds nothing by design.
            [ "$structure" = Null ] && continue
            # shellcheck disable=SC2086
            "$bin" "$structure" "$mode" "$scene" --verify ${VERIFY_ARGS:-} ${RUN_ARGS:-} >> "$verified"
            echo "verified $scene $mode $structure" >&2
        done
    done
done

for ((round = 0; round < rounds; round++)); do
    order=("${structures[@]}")
    if ((round % 2 == 1)); then
        order=()
        for ((i = ${#structures[@]} - 1; i >= 0; i--)); do order+=("${structures[i]}"); done
    fi
    for seed in "${seeds[@]}"; do
        for scene in "${scenes[@]}"; do
            for mode in "${modes[@]}"; do
                for structure in "${order[@]}"; do
                    skip "$structure" "$mode" "$scene" && continue
                    # shellcheck disable=SC2086
                    line=$("$bin" "$structure" "$mode" "$scene" --seed "$seed" ${RUN_ARGS:-})
                    echo "{\"round\":$round,\"source\":\"$source\",\"invocation\":\"$invocation\",${line#\{}" >> "$out"
                    echo "round $round seed $seed $scene $mode $structure" >&2
                done
            done
        done
    done
done
