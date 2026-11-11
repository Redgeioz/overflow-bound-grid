#!/usr/bin/env python3
"""The unified baseline: every structure on one commit, compiler and machine, in
one run of under 30 minutes.

    python benches/run_baseline.py NAME [--cpu 12] [--default-codegen]

Builds the benches with the code alignment forced, which takes most of the
layout out of the comparisons of later builds, then

1. checks the results of the frame benchmark against its reference (in parallel,
   since nothing is timed),
2. waits until the machine has settled,
3. times the frame benchmark, one process per run,
4. times the criterion benches, one process per bench and round; the first round
   is saved as criterion baseline NAME, under target/baseline/criterion.

Every timed process is pinned to one core at high priority and nothing runs
beside it, since the large scenes react to contention for the shared cache.
Writes bench-results/NAME.{meta.json,verify.jsonl,frame.jsonl,criterion.tsv,
criterion.log,Cargo.lock}. Rerunning with the same NAME continues an interrupted
run if the sources are the same, and refuses otherwise.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench-results"
TARGET = REPO / "target" / "baseline"
ALIGNED = "-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=5"

HALO = "HaloOverflowBoundGrid"
SEEDS = (0, 1)
ROUNDS = 2
SHARES = ("1", "0.1", "0.01", "0")
CRITERION_ROUNDS = 2
# See `settle`: the wait after the verification, the busy share of the machine to
# wait for after that, and for how long at most.
SETTLE_S = 120
QUIET_SHARE = 0.04
QUIET_WAIT_S = 180


def structures(scene, mode, share):
    """The structures of a cell of the frame matrix."""
    if scene == "small":
        # Items that all fit a cell, so the single-level grids, which the
        # mixed sizes would leave with crowded overflow nodes, against each
        # other; `MultiGrid/1` is a loose grid without bounds.
        return ["HaloGrid", "SpanGrid", "UniformGrid", "TwoLayerGrid", "MultiGrid/1"]
    names = [HALO, "SpanOverflowBoundGrid", "HierarchicalLooseGrid", "BoxedLooseGrid", "UniformGrid", "TwoLayerGrid",
             "ParryRaw/opt-refit"]
    if mode == "search":
        return names + ["PackedRTree/static-hilbert-coherent"]
    names += ["SortAndSweep", "PackedRTree/static-hilbert-join-coherent"]
    if share != "1":
        # Parry under Rapier's schedule, with partial refits, is its best setting
        # when few items move.
        names.append("ParryRapier/m4")
    if share == "0.01":
        names.append("PackedRTree/static-hilbert-join-coherent-defer5")
    return names


def frame_cells():
    """(scene, mode, share, structures, seeds, rounds) of every timed cell."""
    cells = []
    for scene in ("10k", "160k"):
        for mode in ("pairs", "search"):
            for share in SHARES:
                cells.append((scene, mode, share, structures(scene, mode, share), SEEDS, ROUNDS))
    # The small items, region queries only: what Halo's copies are for. Pairs
    # gain more from the handling of the bounds than from the copies.
    for share in SHARES:
        cells.append(("small", "search", share, structures("small", "search", share), SEEDS, ROUNDS))
    # Controls of the reports, once each per seed 0: the adapters as a library's
    # user would drive them, and the sweep with an index in y.
    for scene in ("10k", "160k"):
        cells.append((scene, "pairs", "1", ["ParryBvh/opt-refit", "PackedRTree/static-hilbert", "SpatialJoinSweep/kept-vec"], (0,), ROUNDS))
        cells.append((scene, "search", "1", ["ParryBvh/opt-refit", "PackedRTree/static-hilbert"], (0,), ROUNDS))
    # A world 16 times taller than wide, at the same density and finest cell
    # width; `tall4` left the matrix on 2026-11-11.
    cells.append(("tall16", "pairs", "1", [HALO, "SpanOverflowBoundGrid", "HierarchicalLooseGrid", "UniformGrid", "TwoLayerGrid", "SortAndSweep", "SpatialJoinSweep/sorted-btree"], (0,), ROUNDS))
    return cells


def timed_runs():
    """Every timed run, the structures in reverse order on odd rounds."""
    cells = frame_cells()
    runs = []
    for round_number in range(max(cell[5] for cell in cells)):
        for seed in SEEDS:
            for scene, mode, share, names, seeds, rounds in cells:
                if round_number >= rounds or seed not in seeds:
                    continue
                for name in (reversed(names) if round_number % 2 else names):
                    runs.append((name, mode, scene, share, seed, round_number))
    return runs


def verified_runs():
    """Every structure, mode and scene of the matrix, with all items and a few moving."""
    runs = set()
    for scene, mode, share, names, _, _ in frame_cells():
        if share in ("1", "0.01"):
            runs.update((name, mode, scene, share) for name in names)
    return sorted(runs)


def pin(cpu):
    """Pin this process, and so the processes it starts from now on, to one core."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1 << cpu):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")
    else:
        os.sched_setaffinity(0, {cpu})


def priority():
    """High priority for a timed process, which Windows does not inherit."""
    return {"creationflags": subprocess.HIGH_PRIORITY_CLASS} if sys.platform == "win32" else {}


def system_times():
    """CPU time of the whole machine so far, (busy, total), in any unit."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        value = lambda time: (time.dwHighDateTime << 32) | time.dwLowDateTime
        # Kernel time includes idle time.
        total = value(kernel) + value(user)
        return total - value(idle), total
    with open("/proc/stat", encoding="ascii") as file:
        fields = [int(field) for field in file.readline().split()[1:]]
    return sum(fields) - fields[3] - fields[4], sum(fields)


def busy_since(start):
    """The busy share of the whole machine since the `system_times` `start`."""
    busy, total = system_times()
    return (busy - start[0]) / max(total - start[1], 1)


def output(*command):
    # Windows reports the power scheme in the code page of its language.
    return subprocess.run(command, check=True, capture_output=True, text=True, errors="replace", cwd=REPO).stdout.strip()


def powershell(command):
    """The output of a PowerShell command, empty if it fails."""
    try:
        return output("powershell", "-NoProfile", "-Command", command)
    except (OSError, subprocess.CalledProcessError):
        return ""


def machine():
    """What the numbers depend on, beyond the sources. Each value is read on its
    own and is None where it cannot be: the CIM queries once printed nothing,
    which shifted the lines of a combined query (baseline-2026-11-11-rerun
    recorded the power scheme as its CPU)."""
    info = {"platform": platform.platform(), "logical_cpus": os.cpu_count()}
    if sys.platform == "win32":
        import ctypes
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            info["cpu"] = None

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ac_line", ctypes.c_ubyte), ("battery_flag", ctypes.c_ubyte), ("battery_percent", ctypes.c_ubyte),
                        ("saver", ctypes.c_ubyte), ("life_time", ctypes.c_ulong), ("full_life_time", ctypes.c_ulong)]

        status = PowerStatus()
        # ACLineStatus is 1 on AC power, 0 on battery and 255 if unknown.
        known = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
        info["ac_power"] = {0: False, 1: True}.get(status.ac_line) if known else None
        battery = powershell("(Get-CimInstance Win32_Battery).BatteryStatus").splitlines()
        # 2 is on AC power.
        info["battery_status"] = battery[0].strip() if battery else None
        # The GUID alone: the rest is in the language of the system.
        scheme = re.search(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", powershell("powercfg /getactivescheme"))
        info["power_scheme"] = scheme.group(0) if scheme else None
    elif Path("/proc/cpuinfo").exists():
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        info["cpu"] = next((line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("model name")), None)
    return info


def source_hash():
    """As in run_frames.sh, over the tracked and new files of the sources."""
    files = output("git", "ls-files", "-co", "--exclude-standard", "--", "src", "benches", "Cargo.toml").splitlines()
    digest = hashlib.sha1()
    for name in sorted(files):
        path = REPO / name
        if path.is_file() and "__pycache__" not in name:
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def build(rustflags):
    """Build the benches; returns their executables by name."""
    env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET), RUSTFLAGS=rustflags)
    locked = ["--locked"] if (REPO / "Cargo.lock").exists() else []
    result = subprocess.run(["cargo", "bench", "--no-run", "--message-format=json-render-diagnostics", *locked],
                            cwd=REPO, env=env, stdout=subprocess.PIPE, text=True, check=True)
    executables = {}
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact" and "bench" in message["target"]["kind"] and message.get("executable"):
            executables[message["target"]["name"]] = message["executable"]
    missing = {"frame", "search", "update"} - set(executables)
    if missing:
        sys.exit(f"benches not built: {sorted(missing)}")
    return executables


def clock(started):
    seconds = int(time.perf_counter() - started)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def settle(started):
    """Wait until the timed runs will not pay for what ran before them, and return
    the busy share of the machine at the end of the wait.

    After the verification the small scene is 5% slower, which fades in about
    100 s. Beside that, the large scene is 10% slower while the rest of the
    machine uses about 5% of it instead of 3%, as it does at times on a desktop in
    use (bench-results/baseline-2026-11-04.md), hence the measurement.
    """
    time.sleep(SETTLE_S)
    busy = None
    deadline = time.perf_counter() + QUIET_WAIT_S
    while True:
        start = system_times()
        time.sleep(10)
        busy = busy_since(start)
        if busy < QUIET_SHARE or time.perf_counter() >= deadline:
            break
    print(f"[{clock(started)}] settled, machine {busy * 100:.1f}% busy", flush=True)
    return busy


def verify(exe, path, chosen, started):
    """Check every result of selected frames against the reference."""
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["structure"], row["mode"], row["scene"], str(row.get("moving", "1"))))
    runs = [run for run in verified_runs() if run not in done and chosen(run[0])]

    def check(run):
        name, mode, scene, share = run
        result = subprocess.run([exe, name, mode, scene, "--verify", "--moving", share], capture_output=True, text=True, cwd=REPO)
        return run, result

    workers = max(1, min(8, (os.cpu_count() or 2) // 2))
    with concurrent.futures.ThreadPoolExecutor(workers) as pool, open(path, "a", encoding="utf-8") as sink:
        for (name, mode, scene, share), result in pool.map(check, runs):
            if result.returncode != 0:
                sys.exit(f"VERIFICATION FAILED: {name} {mode} {scene} moving {share}: {result.stderr.strip()}")
            row = json.loads(result.stdout)
            row["moving"] = share
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    print(f"[{clock(started)}] verified {len(runs)} runs", flush=True)


def time_frames(exe, path, meta, chosen, started):
    runs = [run for run in timed_runs() if chosen(run[0])]
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["structure"], row["mode"], row["scene"], f"{row['moving']:g}", row["seed"], row["round"]))
    remaining = [run for run in runs if run not in done]
    print(f"[{clock(started)}] frame: {len(runs)} runs, {len(remaining)} to do", flush=True)
    phase = time.perf_counter()
    with open(path, "a", encoding="utf-8") as sink:
        for index, (name, mode, scene, share, seed, round_number) in enumerate(remaining, 1):
            start = system_times()
            run = time.perf_counter()
            line = subprocess.run([exe, name, mode, scene, "--seed", str(seed), "--moving", share],
                                  check=True, capture_output=True, text=True, cwd=REPO, **priority()).stdout.strip()
            wall = time.perf_counter() - run
            prefix = {"round": round_number, "source": meta["source"], "invocation": meta["invocation"],
                      "wall_s": round(wall, 3), "machine_busy": round(busy_since(start), 4)}
            sink.write(json.dumps(prefix)[:-1] + "," + line[1:] + "\n")
            sink.flush()
            if index % 25 == 0 or index == len(remaining):
                elapsed = time.perf_counter() - phase
                left = elapsed / index * (len(remaining) - index)
                print(f"[{clock(started)}] frame {index}/{len(remaining)}, about {left / 60:.1f} min left", flush=True)


def read_estimates(home, since):
    """Statistics of the benchmarks that finished after `since`, by id, in ns."""
    results = {}
    for path in Path(home).rglob("new/benchmark.json"):
        estimates = path.with_name("estimates.json")
        if estimates.stat().st_mtime < since:
            continue
        results[json.loads(path.read_text(encoding="utf-8"))["full_id"]] = json.loads(estimates.read_text(encoding="utf-8"))
    return results


def time_criterion(executables, name, tsv, log, meta, id_filter, started):
    # A pinned process has no use for the threads of criterion's analysis.
    env = dict(os.environ, CRITERION_HOME=str(TARGET / "criterion"), RAYON_NUM_THREADS="1")
    done = set()
    if tsv.exists():
        for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split("\t")
            done.add((int(fields[0]), fields[1]))
    else:
        tsv.write_text("round\tbench\tid\tmedian_us\tlow_us\thigh_us\tmean_us\tmachine_busy\n", encoding="utf-8")
    for round_number in range(CRITERION_ROUNDS):
        for bench in ("search", "update"):
            if (round_number, bench) in done:
                continue
            selection = [id_filter] if id_filter else []
            listed = subprocess.run([executables[bench], "--bench", *selection, "--list"], check=True, capture_output=True,
                                    text=True, cwd=REPO, env=env).stdout
            expected = {line.rsplit(":", 1)[0] for line in listed.splitlines() if line.endswith(": benchmark")}
            # Later rounds compare with the first, so that the log shows the noise.
            flag = ["--save-baseline", name] if round_number == 0 else ["--baseline", name]
            since = time.time() - 1
            start = system_times()
            process = time.perf_counter()
            with open(log, "ab") as sink:
                sink.write(f"\n=== round {round_number} {bench} ===\n".encode())
                sink.flush()
                subprocess.run([executables[bench], "--bench", *selection, *flag], check=True, cwd=REPO, env=env,
                               stdout=sink, stderr=subprocess.STDOUT, **priority())
            busy = busy_since(start)
            results = read_estimates(env["CRITERION_HOME"], since)
            missing = expected - set(results)
            if missing:
                sys.exit(f"criterion {bench}: no results for {sorted(missing)}")
            with open(tsv, "a", encoding="utf-8") as sink:
                for full_id in sorted(expected):
                    estimate = results[full_id]
                    median = estimate["median"]
                    interval = median["confidence_interval"]
                    sink.write(f"{round_number}\t{bench}\t{full_id}\t{median['point_estimate'] / 1e3:.3f}\t"
                               f"{interval['lower_bound'] / 1e3:.3f}\t{interval['upper_bound'] / 1e3:.3f}\t"
                               f"{estimate['mean']['point_estimate'] / 1e3:.3f}\t{busy:.4f}\n")
            meta.setdefault("criterion_processes", []).append(
                {"round": round_number, "bench": bench, "ids": len(expected), "wall_s": round(time.perf_counter() - process, 1), "machine_busy": round(busy, 4)})
            print(f"[{clock(started)}] criterion round {round_number} {bench}: {len(expected)} ids in {time.perf_counter() - process:.0f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--cpu", type=int, default=12, help="the core of the timed runs; a performance core")
    parser.add_argument("--default-codegen", action="store_true", help="do not force the code alignment")
    parser.add_argument("--only", default="verify,frame,criterion", help="comma-separated steps")
    parser.add_argument("--filter", default="", help="regex of the frame structures and criterion ids to run, for a trial")
    parser.add_argument("--dry-run", action="store_true", help="only print the size of the matrix")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    chosen = lambda structure: re.search(args.filter, structure) is not None

    if args.dry_run:
        runs = timed_runs()
        print(f"{len(verified_runs())} verifications, {len(runs)} timed frame runs:")
        for scene in sorted({run[2] for run in runs}):
            print(f"  {scene}: {sum(run[2] == scene for run in runs)}")
        return

    started = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    meta_path = RESULTS / f"{args.name}.meta.json"
    rustflags = "" if args.default_codegen else ALIGNED
    source = source_hash()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["source"] != source or meta["rustflags"] != rustflags or meta["filter"] != args.filter:
            sys.exit(f"{meta_path} is of other sources, flags or a filter; use another name")
        print("continuing the run of this name", flush=True)
    else:
        meta = {
            "name": args.name,
            "source": source,
            "invocation": f"{source}-{int(time.time())}-{os.getpid()}",
            "commit": output("git", "rev-parse", "HEAD"),
            "modified": output("git", "status", "--short", "--", "src", "benches", "Cargo.toml").splitlines(),
            "rustc": output("rustc", "-vV").splitlines(),
            "rustflags": rustflags,
            "filter": args.filter,
            "core": args.cpu,
            "machine": machine(),
            "started": datetime.now(timezone.utc).isoformat(),
            "attempts": [],
        }
    attempt = {"started": datetime.now(timezone.utc).isoformat()}
    meta["attempts"].append(attempt)
    save = lambda: meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    save()

    executables = build(rustflags)
    attempt["build_s"] = round(time.perf_counter() - started, 1)
    print(f"[{clock(started)}] built", flush=True)
    if (REPO / "Cargo.lock").exists():
        shutil.copyfile(REPO / "Cargo.lock", RESULTS / f"{args.name}.Cargo.lock")

    steps = args.only.split(",")
    if "verify" in steps:
        verify(executables["frame"], RESULTS / f"{args.name}.verify.jsonl", chosen, started)
        attempt["verified_at_s"] = round(time.perf_counter() - started, 1)
        save()
    if not args.filter:
        attempt["settled_busy"] = round(settle(started), 4)
        attempt["settled_at_s"] = round(time.perf_counter() - started, 1)
        save()
    pin(args.cpu)
    if "frame" in steps:
        time_frames(executables["frame"], RESULTS / f"{args.name}.frame.jsonl", meta, chosen, started)
        attempt["frames_at_s"] = round(time.perf_counter() - started, 1)
        save()
    if "criterion" in steps:
        time_criterion(executables, args.name, RESULTS / f"{args.name}.criterion.tsv", RESULTS / f"{args.name}.criterion.log",
                       meta, args.filter, started)
    attempt["total_s"] = round(time.perf_counter() - started, 1)
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"[{clock(started)}] done", flush=True)


if __name__ == "__main__":
    main()
