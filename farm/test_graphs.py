"""Offline structural checks for the graph builders. No GPU, no ComfyUI.

Catches the class of bug that costs a full round of renders: a link pointing at
a node that does not exist, a node ordering that silently drops a model patch,
or a negative encoder that was handed different images than the positive one.

    python farm/test_graphs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import graphs  # noqa: E402

fails: list[str] = []


def check(name: str, g: dict, expect: list[str] | None = None) -> None:
    ids = set(g)
    for nid, node in g.items():
        if "class_type" not in node or "inputs" not in node:
            fails.append(f"{name}: node {nid} malformed")
            return
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                if v[0] not in ids:
                    fails.append(f"{name}: node {nid}.{k} -> missing node {v[0]}")
    classes = [n["class_type"] for n in g.values()]
    for e in expect or []:
        if e not in classes:
            fails.append(f"{name}: missing expected node {e}")
    print(f"  {name:26s} {len(g):2d} nodes  {' -> '.join(classes[:4])} ...")


def main() -> int:
    check("zimage_t2i", graphs.zimage_t2i("test"),
          ["UNETLoader", "KSampler", "SaveImage"])
    check("zimage_depth_control", graphs.zimage_depth_control("test", "d.png"),
          ["ModelPatchLoader", "ZImageFunControlnet"])
    check("zimage_depth+init",
          graphs.zimage_depth_control("t", "d.png", init_image="p.png", denoise=0.45),
          ["VAEEncode"])
    check("qwen_edit", graphs.qwen_edit("test", ["clay.png"]),
          ["CFGNorm", "TextEncodeQwenImageEditPlus"])
    check("qwen_edit lightning4",
          graphs.qwen_edit("t", ["a.png", "b.png"], steps=4, cfg=1.0, shift=1.0,
                           lightning=config.MODELS["qwen_lightning_4"]),
          ["LoraLoaderModelOnly"])
    check("wan_i2v", graphs.wan_i2v("test", "plate.png"),
          ["Wan22ImageToVideoLatent", "VHS_VideoCombine"])
    check("wan_i2v core video", graphs.wan_i2v("t", "p.png", video_node="core"),
          ["CreateVideo", "SaveVideo"])

    # The patch wraps block forwards, so sampling must already be applied or the
    # wrapper re-clones the model and drops the patch. This has bitten before.
    g = graphs.zimage_depth_control("t", "d.png")
    patch = next(nid for nid, n in g.items() if n["class_type"] == "ZImageFunControlnet")
    upstream = g[patch]["inputs"]["model"][0]
    if g[upstream]["class_type"] != "ModelSamplingAuraFlow":
        fails.append(f"ORDER: ZImageFunControlnet consumes {g[upstream]['class_type']}")
    else:
        print("  order invariant            ZImageFunControlnet <- ModelSamplingAuraFlow")

    # Both Qwen encoders must see the same images, or the two conditioning
    # branches describe different scenes (inert at cfg 1, not at 2.5-4.0).
    g = graphs.qwen_edit("t", ["a.png", "b.png"])
    encs = [n for n in g.values() if n["class_type"] == "TextEncodeQwenImageEditPlus"]
    counts = [sum(1 for k in e["inputs"] if k.startswith("image")) for e in encs]
    if len(encs) != 2 or counts != [2, 2]:
        fails.append(f"QWEN: encoders carry {counts} images, expected [2, 2]")
    else:
        print("  qwen neg encoder           carries the same 2 images")

    for label, fn in [
        ("qwen ref cap", lambda: graphs.qwen_edit("t", ["a.png"] * 4)),
        ("wan 4n+1 guard", lambda: graphs.wan_i2v("t", "p.png", length=50)),
    ]:
        try:
            fn()
            fails.append(f"{label}: did not raise")
        except ValueError:
            print(f"  {label:26s} rejects bad input")

    print()
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("all graph checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
