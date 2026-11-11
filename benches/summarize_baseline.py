"""Tables of a run of benches/run_baseline.py.

    python benches/summarize_baseline.py NAME

Prints what the run was, how long each step took, how busy the machine was while
it ran, the frame tables of benches/summarize_frames.py, and per criterion group
the median of every id, with the difference between its rounds, which is the
noise a comparison with this baseline has to clear.
"""

import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench-results"
# The groups of benches/search.rs and benches/update.rs by the sizes of their
# items, listed apart; `search` stands for the ungrouped ids of the search bench.
# `small_pairs` left the benches on 2026-11-11 and stays for the older runs.
SECTIONS = (
    ("mixed sizes", {"search", "moving", "self_query", "pairs", "pairs_160k", "search_160k", "update", "reinsert",
                     "update_compare", "update_160k"}),
    ("small items", {"small_balls", "small_pairs", "small_update"}),
    ("other static scenes", {"aligned", "gap", "adjacent"}),
)
GROUPS = set().union(*(groups for _, groups in SECTIONS)) - {"search"}


def minutes(seconds):
    return f"{seconds / 60:.1f} min"


def describe(name, meta):
    print(f"## {name}\n")
    print(f"- Commit `{meta['commit'][:12]}`, sources `{meta['source']}`; changed against the commit: "
          f"{', '.join(f'`{line.strip()}`' for line in meta['modified']) or 'nothing'}.")
    version = {line.split(': ')[0]: line.split(': ')[1] for line in meta['rustc'] if ': ' in line}
    print(f"- {meta['rustc'][0]}, LLVM {version.get('LLVM version')}, {version.get('host')}; RUSTFLAGS `{meta['rustflags']}`.")
    machine = meta["machine"]
    power = f"on AC power {machine['ac_power']}; " if "ac_power" in machine else ""
    print(f"- {machine.get('cpu')}, core {meta['core']}, high priority; {power}battery status {machine.get('battery_status')} (2 is on AC power); power scheme {machine.get('power_scheme')}.")
    for number, attempt in enumerate(meta["attempts"], 1):
        parts = [f"build {minutes(attempt['build_s'])}"] if "build_s" in attempt else []
        previous = attempt.get("build_s", 0)
        for key, label in (("verified_at_s", "verify"), ("settled_at_s", "settle"), ("frames_at_s", "frame"), ("total_s", "criterion")):
            if key in attempt:
                parts.append(f"{label} {minutes(attempt[key] - previous)}")
                previous = attempt[key]
        total = f", total **{minutes(attempt['total_s'])}**" if "total_s" in attempt else ", interrupted"
        print(f"- Attempt {number}: {', '.join(parts)}{total}.")


def describe_verification(name):
    path = RESULTS / f"{name}.verify.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"- Verified against the reference: {len(rows)} runs, {sum(row['frames_checked'] for row in rows)} frames, none differ.")


def describe_load(rows, cores):
    """How much else ran beside the pinned process, whose own share is 1/cores."""
    own = 1 / cores
    others = sorted(max(row["machine_busy"] - own, 0) for row in rows)
    at = lambda q: others[round((len(others) - 1) * q)]
    print(f"- CPU use of everything else during the frame runs, as a share of all {cores} logical CPUs: "
          f"median {at(0.5) * 100:.1f}%, 90th percentile {at(0.9) * 100:.1f}%, highest {others[-1] * 100:.1f}%.")


def criterion_tables(path):
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    by_id = defaultdict(dict)
    bench_of = {}
    for round_number, bench, full_id, median, low, high, _mean, _busy in rows:
        by_id[full_id][int(round_number)] = (float(median), float(low), float(high))
        bench_of[full_id] = bench
    groups = defaultdict(list)
    for full_id in by_id:
        group = full_id.split("/", 1)[0]
        # The ungrouped ids of `search` are named after their structure.
        groups[(bench_of[full_id], group if group in GROUPS else "search")].append(full_id)
    section_of = {group: number for number, (_, members) in enumerate(SECTIONS) for group in members}
    ordered = sorted(groups.items(), key=lambda item: (section_of.get(item[0][1], len(SECTIONS)), item[0]))
    differences = []
    print("\n## Criterion\n")
    print("Median per iteration, µs, of the first round, with its 95% interval; the last column is the second round over the first.\n")
    section = None
    for (bench, group), ids in ordered:
        number = section_of.get(group, len(SECTIONS))
        if number != section:
            section = number
            print(f"### {SECTIONS[number][0].capitalize() if number < len(SECTIONS) else 'Other'}\n")
        print(f"#### {bench}: {group}\n")
        print("| id | median | interval | round 2 / 1 |")
        print("|---|---:|---:|---:|")
        for full_id in sorted(ids):
            median, low, high = by_id[full_id][0]
            ratio = by_id[full_id][1][0] / median if 1 in by_id[full_id] else None
            if ratio is not None:
                differences.append((abs(ratio - 1), full_id, ratio))
            print(f"| {full_id} | {median:,.1f} | {low:,.1f}–{high:,.1f} | {f'{ratio:.3f}' if ratio is not None else '-'} |")
        print()
    if differences:
        print("### Noise between the rounds\n")
        values = sorted(item[0] for item in differences)
        at = lambda q: values[round((len(values) - 1) * q)]
        print(f"{len(values)} ids; absolute difference median {at(0.5) * 100:.1f}%, 90th percentile {at(0.9) * 100:.1f}%, highest {values[-1] * 100:.1f}%. Largest:\n")
        for difference, full_id, ratio in sorted(differences, reverse=True)[:10]:
            print(f"- {full_id}: {ratio:.3f}")
        print()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    name = sys.argv[1]
    meta = json.loads((RESULTS / f"{name}.meta.json").read_text(encoding="utf-8"))
    describe(name, meta)
    describe_verification(name)
    frame = RESULTS / f"{name}.frame.jsonl"
    if frame.exists():
        rows = [json.loads(line) for line in frame.read_text(encoding="utf-8").splitlines() if line.strip()]
        describe_load(rows, meta["machine"]["logical_cpus"])
        print(f"- {len(rows)} frame runs, {statistics.fmean(row['wall_s'] for row in rows):.1f} s on average.")
        sys.stdout.flush()
        subprocess.run([sys.executable, str(REPO / "benches" / "summarize_frames.py"), str(frame)], check=True)
    tsv = RESULTS / f"{name}.criterion.tsv"
    if tsv.exists():
        criterion_tables(tsv)


if __name__ == "__main__":
    main()
