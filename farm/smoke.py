"""Prove the farm works and that the second card is actually earning its place.

    python farm/smoke.py               # verify + 4 jobs on 1 card, then 4 on 2
    python farm/smoke.py --jobs 6
    python farm/smoke.py --verify-only # node/model checks, no GPU work

The scaling number is the point. Two workers on two cards should approach 2x on
an independent-job batch, because nothing crosses the PCIe bus between them.
If it comes back near 1.0, either a worker is down or --cuda-device did not
take and both processes are fighting over GPU 0.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import graphs  # noqa: E402
from comfy_client import ComfyClient  # noqa: E402
from dispatch import Farm, Job  # noqa: E402
from preflight import verify_worker  # noqa: E402

PROMPT = (
    "photograph, sharp focus, fine detail. A long empty municipal corridor at "
    "dusk, painted cinderblock walls, scuffed linoleum, one fluorescent fitting "
    "still lit at the far end. Deserted, long abandoned, every surface settled "
    "with dust. Overcast daylight from an unseen window on the left."
)


def verify(port: int, name: str) -> bool:
    client = ComfyClient(port)
    if not client.reachable():
        print(f"  {name}: NOT REACHABLE on {port}")
        return False

    devs = client.devices()
    dev_line = ", ".join(
        f"{d.get('name', '?')} {d.get('vram_free', 0) / 1024**3:.1f}/"
        f"{d.get('vram_total', 0) / 1024**3:.1f} GB free" for d in devs)
    print(f"  {name}: up, {len(devs)} device(s) - {dev_line}")
    if len(devs) != 1:
        print(f"  {name}: FAIL - expected exactly 1 visible device. "
              f"--cuda-device did not take, and on Windows two visible NVIDIA "
              f"cards in one process is the open CUDA bug (ComfyUI #15255).")
        return False

    # Node classes
    try:
        info = client.object_info()
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: could not read /object_info: {e}")
        return False
    wanted = sorted({n for names in graphs.REQUIRED_NODES.values() for n in names})
    missing = [n for n in wanted if n not in info]
    if missing:
        print(f"  {name}: MISSING node classes: {', '.join(missing)}")
    else:
        print(f"  {name}: all {len(wanted)} node classes present")

    # Model enums, live
    problems = verify_worker(port, graphs.MODEL_ENUMS)
    if problems:
        print(f"  {name}: model enum problems:")
        for p in problems:
            print(f"      - {p}")
    else:
        print(f"  {name}: all {len(graphs.MODEL_ENUMS)} model names resolve")

    # Video output node: core CreateVideo/SaveVideo vs the VHS suite.
    has_vhs = "VHS_VideoCombine" in info
    has_core = "CreateVideo" in info and "SaveVideo" in info
    print(f"  {name}: video out - VHS_VideoCombine={has_vhs} core CreateVideo/SaveVideo={has_core}")

    return not missing and not problems


def bench(n_jobs: int, workers: list[dict], out_dir: Path, label: str,
          steps: int, width: int, height: int, seed_base: int) -> float:
    """seed_base must differ per arm.

    ComfyUI caches node results, so a job whose graph is byte-identical to one
    already executed on that worker returns in ~1 s without sampling. The first
    version of this benchmark reused seeds 31337..n across the warmup and the
    single arm and got a free cache hit in each, which flattered one arm and
    not the other. Every arm now occupies its own seed range.
    """
    jobs = [
        Job(
            job_id=f"{label}_{i:02d}",
            graph=graphs.zimage_t2i(
                PROMPT, width=width, height=height, steps=steps,
                seed=seed_base + i, prefix=f"smoke_{label}_{i:02d}"),
            model_key="zimage",
            meta={"arm": label, "steps": steps, "res": f"{width}x{height}",
                  "seed_base": seed_base},
        )
        for i in range(n_jobs)
    ]
    farm = Farm(out_dir / label, workers=workers)
    t0 = time.monotonic()
    results = farm.run(jobs)
    wall = time.monotonic() - t0
    ok = [r for r in results if r.status == "ok"]
    per_card: dict[str, list[float]] = {}
    for r in ok:
        per_card.setdefault(r.worker, []).append(r.seconds)
    print(f"\n  {label}: {len(ok)}/{n_jobs} ok in {wall:.1f}s wall")
    for wname, times in sorted(per_card.items()):
        avg = sum(times) / len(times)
        print(f"      {wname}: {len(times)} jobs, {avg:.1f}s avg, "
              f"{min(times):.1f}-{max(times):.1f}s")
    for r in results:
        if r.status != "ok":
            print(f"      FAILED {r.job_id} on {r.worker}: {(r.error or '')[:200]}")
    return wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="jobs per arm")
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1344)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=C.PROJECT / "runs" / "smoke")
    args = ap.parse_args()

    print("\nFarm smoke test\n")
    print("Verifying workers:")
    live = []
    for w in C.WORKERS:
        if verify(w["port"], w["name"]):
            live.append(w)
        print()

    if not live:
        print("No usable workers. Run:  .\\farm\\farm.ps1 start\n")
        return 1
    if args.verify_only:
        print(f"{len(live)}/{len(C.WORKERS)} workers verified.\n")
        return 0
    if len(live) < 2:
        print("Only one worker is live; the scaling comparison needs two.\n")
        single = bench(args.jobs, live, args.out, "single",
                       args.steps, args.width, args.height,
                       seed_base=int(time.time()) % 100000 * 1000)
        print(f"\n  single-card wall: {single:.1f}s\n")
        return 1

    print(f"Benchmark: {args.jobs} jobs per arm, Z-Image {args.width}x{args.height} "
          f"@ {args.steps} steps, cfg 1\n")

    # First job on a cold worker pays the weight-load cost, which would swamp
    # the comparison. Warm both cards with one job each before timing.
    # Disjoint seed ranges per arm, wide enough that --jobs cannot overlap them.
    base = int(time.time()) % 100000 * 1000

    print("Warming both workers (weights off disk)...")
    bench(2, live, args.out, "warmup", args.steps, args.width, args.height,
          seed_base=base)

    single = bench(args.jobs, live[:1], args.out, "single",
                   args.steps, args.width, args.height, seed_base=base + 1000)
    dual = bench(args.jobs, live, args.out, "dual",
                 args.steps, args.width, args.height, seed_base=base + 2000)

    speedup = single / dual if dual > 0 else 0.0
    print("\n" + "=" * 62)
    print(f"  one card : {single:6.1f}s for {args.jobs} jobs")
    print(f"  two cards: {dual:6.1f}s for {args.jobs} jobs")
    print(f"  speedup  : {speedup:6.2f}x")
    if speedup >= 1.7:
        verdict = "both cards working"
    elif speedup >= 1.25:
        verdict = ("partial - GPU0 drives the displays and benchmarks slower "
                   "under load (29.3 vs 46.1 TFLOPS fp16), so some gap is expected")
    else:
        verdict = ("NOT SCALING - check that each worker reports exactly one "
                   "device and that both PIDs are alive")
    print(f"  verdict  : {verdict}")
    print("=" * 62 + "\n")
    print(f"  images + manifests under {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
