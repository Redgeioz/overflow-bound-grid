"""Tables of the frame benchmark's JSON lines (benches/run_frames.sh).

    python benches/summarize_frames.py OUT.jsonl [BASELINE]

Per mode and scene, every structure's median over its runs (rounds and
seeds) of the mean time per frame, split into update (moves, updates and the
end of the frame) and queries. The scenes of mixed sizes and the scene of
small items (`small`) are listed apart and never compared. The ratio to
BASELINE (by default HaloOverflowBoundGrid, and HaloGrid in `small`, where
the first does not run) is the median of the ratios of runs with the same
seed and round of the same invocation of run_frames.sh (or, for lines
without an invocation id, the same sources, warmup and frames), so that
seeds with more work do not mix into it; a baseline run that would pair
ambiguously is reported and left out. Also the spread of the runs,
the share of items maintained per frame, the heap the structure holds after
setup and at the end of the run, and the allocations per timed frame. Lines
with a "build" field are kept apart per build. Checksums must agree across
the structures of a mode, scene and seed (except `Null`, which finds
nothing); disagreements are listed first.
"""

import json
import statistics
import sys
from collections import defaultdict

# The µ of the headers on consoles of other code pages.
sys.stdout.reconfigure(encoding="utf-8")


# Scenes whose items never double in size; all others mix sizes.
SMALL_SCENES = {"small"}
# Ratio baselines other than HaloOverflowBoundGrid, by scene.
SCENE_BASELINES = {"small": "HaloGrid"}


def scene(run):
    """The scene, with the share of moving balls where not all move."""
    moving = run.get("moving", 1)
    return run["scene"] if moving == 1 else f"{run['scene']}, moving {moving:g}"


def batch(run):
    """The runs that were timed together: one invocation of run_frames.sh."""
    if "invocation" in run:
        return run["invocation"]
    return (run.get("source"), run.get("warmup"), run.get("frames"), run.get("warmup_queries"))


def label(run):
    name = f"{run['structure']} [{run['build']}]" if "build" in run else run["structure"]
    return f"{name} [pool]" if run.get("pool") else name


def main():
    path = sys.argv[1]
    chosen = sys.argv[2] if len(sys.argv) > 2 else None
    baseline_of = lambda name: chosen or SCENE_BASELINES.get(name, "HaloOverflowBoundGrid")
    runs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

    checksums = defaultdict(set)
    for run in runs:
        if run["structure"] != "Null":
            checksums[(run["mode"], scene(run), run["seed"], run.get("warmup"), run.get("frames"))].add(run["checksum"])
    for key, values in sorted(checksums.items()):
        if len(values) > 1:
            print(f"CHECKSUMS DISAGREE for {key}: {sorted(values)}")

    # The baseline run of every seed, round and build.
    bases = {}
    ambiguous = set()
    for run in runs:
        if run["structure"] == baseline_of(run["scene"]):
            key = (run["mode"], scene(run), run["seed"], run.get("round"), run.get("build"), run.get("pool"), batch(run))
            if key in bases:
                ambiguous.add(key)
            bases[key] = run["frame_us"]["mean"]
    for key in sorted(ambiguous, key=str):
        print(f"AMBIGUOUS BASELINE, left out: {key}")
        del bases[key]

    groups = defaultdict(list)
    for run in runs:
        groups[(run["mode"], scene(run))].append(run)

    section = None
    order = lambda item: (item[0][1].split(",")[0] in SMALL_SCENES, item[0][1], item[0][0])
    for (mode, scene_name), group in sorted(groups.items(), key=order):
        base_scene = scene_name.split(",")[0]
        small = base_scene in SMALL_SCENES
        if small != section:
            section = small
            print("\n## Frames, small items (radii 25 to 50, never doubled)" if small
                  else "\n## Frames, mixed sizes (radii 25 to 50, doubled up to five times)")
        baseline = baseline_of(base_scene)
        by_structure = defaultdict(list)
        for run in group:
            by_structure[label(run)].append(run)
        rows = []
        for name, items in by_structure.items():
            frame = [item["frame_us"]["mean"] for item in items]
            ratios = [
                item["frame_us"]["mean"] / bases[key]
                for item in items
                if (key := (mode, scene_name, item["seed"], item.get("round"), item.get("build"), item.get("pool"), batch(item))) in bases
            ]
            rows.append({
                "name": name,
                "frame": statistics.median(frame),
                "ratio": statistics.median(ratios) if ratios else None,
                "low": min(frame),
                "high": max(frame),
                "update": statistics.median(item["update_us"]["mean"] for item in items),
                "query": statistics.median(item["query_us"]["mean"] for item in items),
                "maintained": (
                    statistics.median(item["maintained"]["mean"] for item in items)
                    if items[0]["maintained"] else None
                ),
                "setup": statistics.median(item["setup_bytes"] for item in items),
                "end": statistics.median(item["end_bytes"] for item in items),
                "allocs": statistics.median(item.get("allocs_per_frame", float("nan")) for item in items),
                "faults": statistics.median(
                    (item.get("faults_per_frame") if item.get("faults_per_frame") is not None else float("nan"))
                    for item in items
                ),
                "runs": len(items),
            })
        rows.sort(key=lambda row: row["frame"])
        unit, scale, digits = ("µs", 1.0, 0) if base_scene in ("10k", "small") else ("ms", 1e-3, 2)
        fmt = lambda value: f"{value * scale:.{digits}f}"
        print(f"\n### {mode}, {scene_name}\n")
        print(f"| structure | frame ({unit}) | vs {baseline} (paired) | update | query | runs, range "
              f"| maintained | heap setup / end (MB) | allocs/frame | faults/frame |")
        print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
        for row in rows:
            ratio = f"{row['ratio']:.2f}" if row["ratio"] is not None else "-"
            maintained = f"{row['maintained'] * 100:.1f}%" if row["maintained"] is not None else "-"
            print(
                f"| {row['name']} | {fmt(row['frame'])} | {ratio} | {fmt(row['update'])} | {fmt(row['query'])} "
                f"| {row['runs']}, {fmt(row['low'])}–{fmt(row['high'])} | {maintained} "
                f"| {row['setup'] / 1e6:.2f} / {row['end'] / 1e6:.2f} | {row['allocs']:.1f} | {row['faults']:.1f} |"
            )


if __name__ == "__main__":
    main()
