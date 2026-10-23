"""API-format ComfyUI graphs, built in code.

Never store UI-format workflow JSON and convert it at runtime. There is no
server-side converter, and reimplementing graphToPrompt() means handling
Get/Set nodes, subgraph expansion, muted and bypassed nodes, and widget
inlining. Both other codebases on this machine that drive ComfyUI seriously
(Roko's comfy.ts, CivAI's comfyui_backend.py) build graphs programmatically for
this reason.

Node ordering is load-bearing in two places and both are marked below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

# Names the CG failure explicitly, which is safe in a negative: it is what we
# do NOT want. The boxy terms were added after the clay route kept returning
# furniture as untextured rectangular blocks — a desk and a desk chair that
# stayed cuboids through the whole edit.
#
# IMPORTANT: this is inert at cfg 1.0. Classifier-free guidance is what makes a
# negative prompt do anything, and cfg 1.0 skips the unconditional pass
# entirely. The 8-step Lightning regime runs at cfg 1.0, so for every Lightning
# render this string has been decorative. If the output is too close to the
# clay input, raising cfg is what turns this back on.
NEGATIVE_EDIT = (
    "3d render, clay render, untextured, grey plastic, flat shading, "
    "video game screenshot, boxy, angular, rectangular blocks, cuboid "
    "furniture, cardboard boxes, simplified geometry, low-poly, polygonal, "
    "faceted, flat untextured panels, visible polygon edges, "
    "blurry, soft focus, low detail, washed out, "
    "deformed hands, extra limbs, text, watermark"
)

NEGATIVE_VIDEO = (
    "overexposed, static, blurry details, watermark, text, low quality, "
    "deformed, ugly, extra fingers, bad hands"
)


class G:
    """Tiny graph builder. Nodes are numbered strings; links are [node_id, slot]."""

    def __init__(self) -> None:
        self._n = 0
        self.nodes: dict[str, dict] = {}

    def add(self, class_type: str, inputs: dict, title: str | None = None) -> str:
        self._n += 1
        nid = str(self._n)
        node = {"class_type": class_type, "inputs": inputs}
        if title:
            node["_meta"] = {"title": title}
        self.nodes[nid] = node
        return nid

    def out(self) -> dict:
        return self.nodes


def _link(node_id: str, slot: int = 0) -> list:
    return [node_id, slot]


# --------------------------------------------------------------- Z-Image ---

def zimage_t2i(prompt: str, width: int = 1024, height: int = 1344,
               steps: int = 9, cfg: float = 1.0, shift: float = 6.0,
               seed: int = 31337, sampler: str = "res_multistep",
               scheduler: str = "simple", prefix: str = "zi") -> dict:
    """Z-Image text-to-image. The cheapest graph that exercises the full
    load -> encode -> sample -> decode path, so it doubles as the smoke test.

    cfg 1 means the negative prompt is inert. That is the production setting
    (RENDER.cfg in scenespec.ts) and the reason MultiGPUCFGSplitNode is useless
    here, so the smoke test deliberately runs at the real value.
    """
    g = G()
    unet = g.add("UNETLoader", {
        "unet_name": C.MODELS["zimage_unet"], "weight_dtype": "default"})
    clip = g.add("CLIPLoader", {
        "clip_name": C.MODELS["zimage_clip"], "type": "lumina2"})
    vae = g.add("VAELoader", {"vae_name": C.MODELS["zimage_vae"]})

    model = g.add("ModelSamplingAuraFlow", {"model": _link(unet), "shift": shift})

    pos = g.add("TextEncodeZImageOmni", {
        "clip": _link(clip), "vae": _link(vae),
        "prompt": prompt, "auto_resize_images": True})
    neg = g.add("ConditioningZeroOut", {"conditioning": _link(pos)})

    latent = g.add("EmptySD3LatentImage", {
        "width": width, "height": height, "batch_size": 1})

    k = g.add("KSampler", {
        "model": _link(model), "positive": _link(pos), "negative": _link(neg),
        "latent_image": _link(latent), "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0})

    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae)})
    g.add("SaveImage", {"images": _link(dec), "filename_prefix": prefix})
    return g.out()


def zimage_depth_control(prompt: str, depth_image: str,
                         width: int = 1024, height: int = 1344,
                         strength: float = 0.5, steps: int = 9, cfg: float = 1.0,
                         shift: float = 6.0, seed: int = 31337,
                         init_image: str | None = None, denoise: float = 1.0,
                         sampler: str = "res_multistep", scheduler: str = "simple",
                         prefix: str = "zictl") -> dict:
    """Z-Image held to a depth map from the blockout. The cheap stage-6 tier,
    ~28 s at 1024x1344 on one card.

    control strength is 0.5 and should stay there. Every arm at or above 0.65
    in the settled-findings ledger went dark and mushy and hallucinated
    architecture absent from the depth map, without becoming more faithful.
    Scheduled release and depth blur did not rescue it.
    """
    g = G()
    unet = g.add("UNETLoader", {
        "unet_name": C.MODELS["zimage_unet"], "weight_dtype": "default"})
    clip = g.add("CLIPLoader", {
        "clip_name": C.MODELS["zimage_clip"], "type": "lumina2"})
    vae = g.add("VAELoader", {"vae_name": C.MODELS["zimage_vae"]})

    # ORDER IS LOAD-BEARING: ModelSamplingAuraFlow must wrap the model BEFORE
    # the control patch. The patch wraps block forwards; if sampling is applied
    # afterwards it re-clones the model and silently drops the patch.
    model = g.add("ModelSamplingAuraFlow", {"model": _link(unet), "shift": shift})

    patch = g.add("ModelPatchLoader", {"name": C.MODELS["zimage_control"]})
    depth = g.add("LoadImage", {"image": depth_image})
    # vae is a REQUIRED input on this node, not optional. Omitting it returns
    # HTTP 400 prompt_outputs_failed_validation, which is a clean failure but
    # only at queue time, so a whole batch dies at once.
    model = g.add("ZImageFunControlnet", {
        "model": _link(model), "model_patch": _link(patch), "vae": _link(vae),
        "image": _link(depth), "strength": strength})

    pos = g.add("TextEncodeZImageOmni", {
        "clip": _link(clip), "vae": _link(vae),
        "prompt": prompt, "auto_resize_images": True})
    neg = g.add("ConditioningZeroOut", {"conditioning": _link(pos)})

    if init_image:
        src = g.add("LoadImage", {"image": init_image})
        scaled = g.add("ImageScale", {
            "image": _link(src), "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "disabled"})
        latent = g.add("VAEEncode", {"pixels": _link(scaled), "vae": _link(vae)})
    else:
        latent = g.add("EmptySD3LatentImage", {
            "width": width, "height": height, "batch_size": 1})
        denoise = 1.0

    k = g.add("KSampler", {
        "model": _link(model), "positive": _link(pos), "negative": _link(neg),
        "latent_image": _link(latent), "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise})

    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae)})
    g.add("SaveImage", {"images": _link(dec), "filename_prefix": prefix})
    return g.out()


# --------------------------------------------------- Qwen-Image-2.1 gen/ed ---

# ========================================================= Qwen-Image-2.1 ==
# One 7B checkpoint for text-to-image AND editing, replacing
# Qwen-Image-Edit-2509. cfg MUST stay 1.0 (at cfg 1 the uncond pass is
# skipped, so the negative is inert and left blank), euler/simple, 25 working
# steps, resolution is a TOTAL pixel budget on TextEncodeQwenImage21, not
# width/height. No Lightning LoRA, no AuraFlow shift, no CFGNorm: all the
# 2509 accelerators are a different architecture and do not load. Max 10
# reference images (image_1 is the edit target/canvas). Prompts are written
# per QWEN_IMAGE_2.1_PROMPTING.md (observe-don't-instruct; <imageN> tags for
# N>=2). Width/height must be multiples of 32.
#
# Qwen-Image-2.1-Turbo (official, 2026-10-09) is the same architecture distilled
# to 8 steps at cfg 1; text encoder and VAE are shared with the base. `variant`
# picks the transformer and the sampler:
#   base          qwen21_unet, KSampler euler/simple, `steps` as given
#   turbo         qwen21_turbo_unet, euler on the 8 sample_sigmas saved in the
#                 turbo's model_index.json (diffusers' default for it)
#   turbo_simple  qwen21_turbo_unet, KSampler euler/simple at 8 steps
# Turbo variants ignore `steps`: the card says the saved schedule is the only one
# evaluated. Bench 2026-10-10 (one 5060 Ti, 1024^2, qwen21_turbo_bench/): same
# prompt 26.4 s base vs 10.9 s turbo, new prompt 40.5 s vs 27.7 s.
QWEN21_VARIANTS = ("base", "turbo", "turbo_simple")
QWEN21_TURBO_SIGMAS = ("1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, "
                       "0.704534, 0.414568, 0.0")
QWEN21_TURBO_STEPS = 8


def _qwen21_unet(g: G, variant: str) -> str:
    if variant not in QWEN21_VARIANTS:
        raise ValueError(f"qwen21 variant {variant!r}: use one of {QWEN21_VARIANTS}")
    key = "qwen21_unet" if variant == "base" else "qwen21_turbo_unet"
    return g.add("UNETLoader", {"unet_name": C.MODELS[key], "weight_dtype": "default"})


def _qwen21_sample(g: G, variant: str, model: str, pos: str, latent: list,
                   seed: int, steps: int, cfg: float, denoise: float = 1.0) -> str:
    """The sampler for a Qwen-2.1 graph; returns the node whose output 0 is the latent.
    denoise < 1 is for a VAE-encoded latent only (the turbo path ignores it)."""
    if variant == "turbo":
        noise = g.add("RandomNoise", {"noise_seed": seed})
        guider = g.add("BasicGuider", {"model": _link(model), "conditioning": _link(pos, 0)})
        sampler = g.add("KSamplerSelect", {"sampler_name": "euler"})
        sigmas = g.add("ManualSigmas", {"sigmas": QWEN21_TURBO_SIGMAS})
        return g.add("SamplerCustomAdvanced", {
            "noise": _link(noise), "guider": _link(guider), "sampler": _link(sampler),
            "sigmas": _link(sigmas), "latent_image": latent})
    if variant == "turbo_simple":
        steps, cfg = QWEN21_TURBO_STEPS, 1.0
    return g.add("KSampler", {
        "model": _link(model), "positive": _link(pos, 0), "negative": _link(pos, 1),
        "latent_image": latent, "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "denoise": float(denoise)})


def qwen21_t2i(prompt: str, width: int = 1024, height: int = 1024,
               steps: int = 25, cfg: float = 1.0, resolution: int = 1024,
               seed: int = 31337, negative: str = "",
               prefix: str = "qwen21", variant: str = "base") -> dict:
    """Qwen-Image-2.1 text-to-image. No cache node (the t2i template has none)."""
    width = max(32, (width + 31) // 32 * 32)
    height = max(32, (height + 31) // 32 * 32)

    g = G()
    unet = _qwen21_unet(g, variant)
    clip = g.add("CLIPLoader", {
        "clip_name": C.MODELS["qwen21_clip"], "type": "qwen_image"})
    vae = g.add("VAELoader", {"vae_name": C.MODELS["qwen21_vae"]})
    latent = g.add("EmptyLatentImage", {
        "width": width, "height": height, "batch_size": 1})

    pos = g.add("TextEncodeQwenImage21", {
        "clip": _link(clip), "prompt": prompt,
        "negative_prompt": negative, "vae": _link(vae),
        "resolution": int(resolution)})

    k = _qwen21_sample(g, variant, unet, pos, _link(latent), seed, steps, cfg)

    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae)})
    g.add("SaveImage", {"images": _link(dec), "filename_prefix": prefix})
    return g.out()


def qwen21_img2img(prompt: str, image: str, denoise: float = 0.5,
                   steps: int = 25, cfg: float = 1.0, seed: int = 31337, negative: str = "",
                   prefix: str = "qwen21i", variant: str = "base") -> dict:
    """Qwen-Image-2.1 img2img: VAE-encode `image`, re-noise to `denoise`, sample from text only.

    The encoder sees no image here. Every qwen21_edit of a photograph came back relit and
    over-sharpened (contrast x1.3-1.4, detail x2.3-2.9, all variants and step counts, cache off
    too; gardening_2min 2026-10-10), while t2i from the same encoder is clean. This keeps the
    still's composition by latent instead of by the encoder's image path.
    """
    if variant == "turbo":
        raise ValueError("qwen21_img2img: the turbo ManualSigmas start at 1.0; use turbo_simple or base")
    g = G()
    unet = _qwen21_unet(g, variant)
    clip = g.add("CLIPLoader", {"clip_name": C.MODELS["qwen21_clip"], "type": "qwen_image"})
    vae = g.add("VAELoader", {"vae_name": C.MODELS["qwen21_vae"]})
    img = g.add("LoadImage", {"image": image})
    lat = g.add("VAEEncode", {"pixels": _link(img), "vae": _link(vae)})
    pos = g.add("TextEncodeQwenImage21", {
        "clip": _link(clip), "prompt": prompt, "negative_prompt": negative,
        "vae": _link(vae), "resolution": 1024})
    if variant == "turbo_simple":
        steps, cfg = QWEN21_TURBO_STEPS, 1.0
    k = g.add("KSampler", {
        "model": _link(unet), "positive": _link(pos, 0), "negative": _link(pos, 1),
        "latent_image": _link(lat), "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "denoise": float(denoise)})
    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae)})
    g.add("SaveImage", {"images": _link(dec), "filename_prefix": prefix})
    return g.out()


def qwen21_edit(prompt: str, refs: list[str],
                steps: int = 25, cfg: float = 1.0, resolution: int = 1024,
                seed: int = 31337, negative: str = "",
                cache_device: str = "auto", cache_dtype: str = "default",
                prefix: str = "qwen21e", variant: str = "base") -> dict:
    """Qwen-Image-2.1 instruction edit.

    refs: 1-10 image filenames already uploaded to the worker's input dir.
    refs[0] is image_1 = the edit target/canvas; the rest are references to
    transfer material from. Address slots in the prompt as <image1>..<imageN>
    when N >= 2; with a single image refer to it naturally, no tags.

    The latent comes from the encoder's THIRD output (the first reference's
    size), exactly as the official template has it. Do not substitute an
    empty latent: any size that disagrees with image_1 shifts the edit.
    QwenImage21Cache sits between the UNET and the sampler on the edit path
    only.
    """
    if not 1 <= len(refs) <= 10:
        raise ValueError(f"qwen21_edit takes 1-10 reference images, got {len(refs)}")

    g = G()
    unet = _qwen21_unet(g, variant)
    clip = g.add("CLIPLoader", {
        "clip_name": C.MODELS["qwen21_clip"], "type": "qwen_image"})
    vae = g.add("VAELoader", {"vae_name": C.MODELS["qwen21_vae"]})
    cache = g.add("QwenImage21Cache", {
        "model": _link(unet), "device": cache_device, "dtype": cache_dtype})

    loaded = [g.add("LoadImage", {"image": r}) for r in refs]

    inputs = {
        "clip": _link(clip), "prompt": prompt,
        "negative_prompt": negative, "vae": _link(vae),
        "resolution": int(resolution),
    }
    for i, node in enumerate(loaded, start=1):
        inputs[f"images.image_{i}"] = _link(node)
    pos = g.add("TextEncodeQwenImage21", inputs)

    k = _qwen21_sample(g, variant, cache, pos, _link(pos, 2), seed, steps, cfg)

    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae)})
    g.add("SaveImage", {"images": _link(dec), "filename_prefix": prefix})
    return g.out()


def qwen21_inpaint(prompt: str, plate: str, mask: str, ref: str | None = "",
                   steps: int = 25, cfg: float = 1.0, resolution: int = 1024,
                   seed: int = 31337, negative: str = "", diff_strength: float = 1.0,
                   cache_device: str = "auto", cache_dtype: str = "default",
                   prefix: str = "qwen21m", variant: str = "base",
                   extra_refs: list[str] | None = None, denoise: float = 1.0) -> dict:
    """Qwen-Image-2.1 edit confined to a mask: the two-pass still's second pass.

    extra_refs: further images for the encoder as image_2.. (e.g. an identity portrait for a
    face repair). denoise < 1 keeps the plate's structure inside the mask (head pose, outline).

    plate: the frame to paint into, uploaded; also image_1, so the encoder sees the whole room.
    mask: a greyscale image, white where the new content goes (read from its red channel).
    ref: image_1 for the encoder. "" uses the plate; None sends no image (text-only inpaint, no
    cache node, as on the t2i path); a filename sends that image instead, e.g. the plate with a
    stand-in figure drawn where the person goes. With the bare plate as image_1 the edit model
    copied the empty chair back into the mask (gardening p01 s005, 2026-10-11).

    The encoder's third output is an EMPTY latent (zeros at image_1's size), so a masked edit
    cannot use it: SetLatentNoiseMask on zeros paints the unmasked area black. The plate is
    VAE-encoded instead, and the sampler keeps every pixel outside the mask. DifferentialDiffusion
    lets a feathered mask edge blend instead of cutting. The plate must already be at the
    encoder's size for `resolution` (1344x768 for a 16:9 frame at 1024), or the masked region
    is painted against a shifted context.
    """
    g = G()
    unet = _qwen21_unet(g, variant)
    clip = g.add("CLIPLoader", {"clip_name": C.MODELS["qwen21_clip"], "type": "qwen_image"})
    vae = g.add("VAELoader", {"vae_name": C.MODELS["qwen21_vae"]})
    model = g.add("DifferentialDiffusion", {"model": _link(unet), "strength": float(diff_strength)})
    if ref is not None:
        model = g.add("QwenImage21Cache", {"model": _link(model), "device": cache_device, "dtype": cache_dtype})
    img = g.add("LoadImage", {"image": plate})
    m = g.add("LoadImageMask", {"image": mask, "channel": "red"})
    lat = g.add("VAEEncode", {"pixels": _link(img), "vae": _link(vae)})
    masked = g.add("SetLatentNoiseMask", {"samples": _link(lat), "mask": _link(m)})
    enc = {"clip": _link(clip), "prompt": prompt, "negative_prompt": negative,
           "vae": _link(vae), "resolution": int(resolution)}
    if ref is not None:
        enc["images.image_1"] = _link(g.add("LoadImage", {"image": ref}) if ref else img)
        for i, r in enumerate(extra_refs or [], start=2):
            enc[f"images.image_{i}"] = _link(g.add("LoadImage", {"image": r}))
    pos = g.add("TextEncodeQwenImage21", enc)
    k = _qwen21_sample(g, variant, model, pos, _link(masked), seed, steps, cfg, denoise)
    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae)})
    g.add("SaveImage", {"images": _link(dec), "filename_prefix": prefix})
    return g.out()


# ------------------------------------------------------------ Qwen edit ---

def qwen_edit(prompt: str, refs: list[str], width: int = 1024, height: int = 1344,
              steps: int = 20, cfg: float = 2.5, shift: float = 3.0,
              seed: int = 31337, sampler: str = "euler", scheduler: str = "simple",
              lightning: str | None = None, negative: str = NEGATIVE_EDIT,
              prefix: str = "qwen") -> dict:
    """Qwen-Image-Edit-2509: turns the clay blockout into a photographic plate.

    refs is 1-3 image filenames already uploaded to the worker's input dir.
    TextEncodeQwenImageEditPlus emits the literal tokens
    `Picture 1: <|vision_start|><|image_pad|><|vision_end|> Picture 2: ...`
    before the prompt, so address them in text as "Picture N" and nothing else.
    More than three throws.

    Four things this graph gets right that a hand-built one got wrong for a
    full round of renders:
      - ModelSamplingAuraFlow at shift 3.0 is present
      - CFGNorm is present (the documented lever for the saturation blowout)
      - the latent is a VAEEncode of the reference at denoise 1.0, not an
        empty latent
      - the NEGATIVE encoder gets the SAME images as the positive one. Without
        that the two conditioning branches describe different scenes, which is
        inert at cfg 1 but not at 2.5-4.0.

    Lightning regimes: 8-step is steps=8 cfg=1.0 shift=1.0; 4-step is
    steps=4 cfg=1.0 shift=1.0. Measured 76.9 s and 47.5 s against a 754 s base.
    """
    if not 1 <= len(refs) <= 3:
        raise ValueError(f"qwen_edit takes 1-3 reference images, got {len(refs)}")

    g = G()
    unet = g.add("UnetLoaderGGUF", {"unet_name": C.MODELS["qwen_edit_unet"]})
    clip = g.add("CLIPLoader", {
        "clip_name": C.MODELS["qwen_edit_clip"], "type": "qwen_image"})
    vae = g.add("VAELoader", {"vae_name": C.MODELS["qwen_edit_vae"]})

    loaded = [g.add("LoadImage", {"image": r}) for r in refs]

    model = unet
    # `lightning` accepts a name, or (name, strength), or a list of either, so
    # a style LoRA can stack on top of the step-distillation one. Order is the
    # order given: the distiller first, then anything shaping the output.
    for spec in (lightning if isinstance(lightning, list)
                 else [lightning] if lightning else []):
        name, strength = spec if isinstance(spec, tuple) else (spec, 1.0)
        model = g.add("LoraLoaderModelOnly", {
            "model": _link(model), "lora_name": name,
            "strength_model": strength})
    model = g.add("ModelSamplingAuraFlow", {"model": _link(model), "shift": shift})
    model = g.add("CFGNorm", {"model": _link(model), "strength": 1.0})

    def encode(text: str) -> str:
        inputs = {"clip": _link(clip), "vae": _link(vae), "prompt": text}
        for i, node in enumerate(loaded, start=1):
            inputs[f"image{i}"] = _link(node)
        return g.add("TextEncodeQwenImageEditPlus", inputs)

    pos = encode(prompt)
    neg = encode(negative)

    scaled = g.add("ImageScale", {
        "image": _link(loaded[0]), "upscale_method": "lanczos",
        "width": width, "height": height, "crop": "disabled"})
    latent = g.add("VAEEncode", {"pixels": _link(scaled), "vae": _link(vae)})

    k = g.add("KSampler", {
        "model": _link(model), "positive": _link(pos), "negative": _link(neg),
        "latent_image": _link(latent), "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0})

    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae)})
    g.add("SaveImage", {"images": _link(dec), "filename_prefix": prefix})
    return g.out()


# ----------------------------------------------------------- Wan 2.2 I2V ---

def wan_i2v(prompt: str, start_image: str, width: int = 832, height: int = 480,
            length: int = 49, fps: float = 24.0, steps: int = 12, cfg: float = 5.0,
            shift: float = 8.0, seed: int = 424242, sampler: str = "euler",
            scheduler: str = "simple", negative: str = NEGATIVE_VIDEO,
            prefix: str = "wan", video_node: str = "VHS_VideoCombine") -> dict:
    """Wan2.2-TI2V-5B image-to-video.

    The defaults are the arm that was actually measured on this machine on
    2026-09-10: 832x480, length 49, shift 8.0, cfg 5.0, euler/simple. At 12
    steps it took 60.47 s for 2.04 s of video, which is 30x realtime and the
    fastest quality-acceptable arm on disk. 20 steps was 173.98 s for the same
    output, so the extra 8 steps cost 2.9x for no measured gain.

    length must satisfy 4n+1. 49 frames at 24 fps is 2.04 s.
    """
    if (length - 1) % 4 != 0:
        raise ValueError(f"wan length must be 4n+1, got {length}")

    g = G()
    unet = g.add("UnetLoaderGGUF", {"unet_name": C.MODELS["wan_unet"]})
    clip = g.add("CLIPLoader", {"clip_name": C.MODELS["wan_clip"], "type": "wan"})
    vae = g.add("VAELoader", {"vae_name": C.MODELS["wan_vae"]})

    model = g.add("ModelSamplingSD3", {"model": _link(unet), "shift": shift})

    pos = g.add("CLIPTextEncode", {"clip": _link(clip), "text": prompt})
    neg = g.add("CLIPTextEncode", {"clip": _link(clip), "text": negative})

    img = g.add("LoadImage", {"image": start_image})
    latent = g.add("Wan22ImageToVideoLatent", {
        "vae": _link(vae), "width": width, "height": height,
        "length": length, "batch_size": 1, "start_image": _link(img)})

    k = g.add("KSampler", {
        "model": _link(model), "positive": _link(pos), "negative": _link(neg),
        "latent_image": _link(latent), "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0})

    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae)})

    if video_node == "VHS_VideoCombine":
        g.add("VHS_VideoCombine", {
            "images": _link(dec), "frame_rate": fps, "loop_count": 0,
            "filename_prefix": prefix, "format": "video/h264-mp4",
            "pingpong": False, "save_output": True})
    else:
        vid = g.add("CreateVideo", {"images": _link(dec), "fps": fps})
        g.add("SaveVideo", {"video": _link(vid), "filename_prefix": prefix,
                            "format": "auto", "codec": "auto"})
    return g.out()


# Node classes each graph needs, for a live /object_info check before a batch.
REQUIRED_NODES = {
    "zimage_t2i": ["UNETLoader", "CLIPLoader", "VAELoader", "ModelSamplingAuraFlow",
                   "TextEncodeZImageOmni", "ConditioningZeroOut",
                   "EmptySD3LatentImage", "KSampler", "VAEDecode", "SaveImage"],
    "zimage_depth_control": ["ModelPatchLoader", "ZImageFunControlnet", "LoadImage",
                             "ImageScale", "VAEEncode"],
    "qwen_edit": ["UnetLoaderGGUF", "TextEncodeQwenImageEditPlus", "CFGNorm",
                  "LoraLoaderModelOnly"],
    "qwen21_t2i": ["UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21",
                   "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"],
    "qwen21_edit": ["UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21",
                    "QwenImage21Cache", "LoadImage", "KSampler", "VAEDecode",
                    "SaveImage"],
    "qwen21_inpaint": ["DifferentialDiffusion", "LoadImageMask", "VAEEncode", "SetLatentNoiseMask"],
    "qwen21_turbo":["RandomNoise", "BasicGuider", "KSamplerSelect", "ManualSigmas",
                     "SamplerCustomAdvanced"],
    "wan_i2v":["UnetLoaderGGUF", "ModelSamplingSD3", "CLIPTextEncode",
                "Wan22ImageToVideoLatent"],
}

# config key -> (loader node class, enum input name), for preflight.verify_worker
MODEL_ENUMS = {
    "zimage_unet": ("UNETLoader", "unet_name"),
    "zimage_clip": ("CLIPLoader", "clip_name"),
    "zimage_vae": ("VAELoader", "vae_name"),
    "zimage_control": ("ModelPatchLoader", "name"),
    "qwen_edit_unet": ("UnetLoaderGGUF", "unet_name"),
    "qwen_edit_clip": ("CLIPLoader", "clip_name"),
    "qwen_edit_vae": ("VAELoader", "vae_name"),
    "qwen_lightning_4": ("LoraLoaderModelOnly", "lora_name"),
    "qwen_lightning_8": ("LoraLoaderModelOnly", "lora_name"),
    "qwen21_unet": ("UNETLoader", "unet_name"),
    "qwen21_turbo_unet": ("UNETLoader", "unet_name"),
    "qwen21_clip": ("CLIPLoader", "clip_name"),
    "qwen21_vae": ("VAELoader", "vae_name"),
    "wan_unet": ("UnetLoaderGGUF", "unet_name"),
    "wan_clip": ("CLIPLoader", "clip_name"),
    "wan_vae": ("VAELoader", "vae_name"),
}


# ------------------------------------------------- Wan 2.2 I2V, distilled ---

def wan_i2v_fp8(prompt: str, start_image: str, width: int = 1280,
                height: int = 704, length: int = 121, fps: float = 24.0,
                steps: int = 4, cfg: float = 1.0, shift: float = 8.0,
                seed: int = 424242, sampler: str = "euler",
                scheduler: str = "simple", negative: str = NEGATIVE_VIDEO,
                unet: str = "wan2.2_ti2v_5B_lightx2v_fp8.safetensors",
                weight_dtype: str = "default",
                prefix: str = "wanfp8") -> dict:
    """lightx2v's step-distilled TI2V-5B, as an fp8 safetensors.

    Same architecture as wan_i2v, so the rest of the graph is identical. What
    changes is the arithmetic, and the arithmetic is the whole cost:

        current   12 steps at cfg 5.0  = 24 model evaluations per clip
        this       4 steps at cfg 1.0  =  4 model evaluations per clip

    cfg above 1.0 runs the unconditional pass as well, which doubles the work;
    MEASURED.md already records that for the Qwen route and it is the same
    arithmetic here. A distilled model does not need guidance, so both factors
    come off at once.

    The negative prompt is therefore INERT at cfg 1.0 - classifier-free
    guidance is what makes a negative do anything. It is still accepted so the
    call sites do not change, and so that raising cfg for a comparison arm
    turns it back on.
    """
    if (length - 1) % 4 != 0:
        raise ValueError(f"wan length must be 4n+1, got {length}")

    g = G()
    unet_node = g.add("UNETLoader", {"unet_name": unet,
                                     "weight_dtype": weight_dtype})
    clip = g.add("CLIPLoader", {"clip_name": C.MODELS["wan_clip"], "type": "wan"})
    vae = g.add("VAELoader", {"vae_name": C.MODELS["wan_vae"]})

    model = g.add("ModelSamplingSD3", {"model": _link(unet_node), "shift": shift})
    pos = g.add("CLIPTextEncode", {"clip": _link(clip), "text": prompt})
    neg = g.add("CLIPTextEncode", {"clip": _link(clip), "text": negative})

    img = g.add("LoadImage", {"image": start_image})
    latent = g.add("Wan22ImageToVideoLatent", {
        "vae": _link(vae), "width": width, "height": height,
        "length": length, "batch_size": 1, "start_image": _link(img)})

    k = g.add("KSampler", {
        "model": _link(model), "positive": _link(pos), "negative": _link(neg),
        "latent_image": _link(latent), "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0})

    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae)})
    g.add("VHS_VideoCombine", {
        "images": _link(dec), "frame_rate": fps, "loop_count": 0,
        "filename_prefix": prefix, "format": "video/h264-mp4",
        "pingpong": False, "save_output": True})
    return g.out()


# ------------------------------------------------------- MiniMax H3 I2V ---

H3_FRAME_GRID = 17          # length must be 17k+5; the node snaps up, we don't


def h3_length(seconds: float, fps: float = 24.0, cap: int = 362) -> int:
    """Frames for a duration, on MiniMax H3's 17k+5 grid.

    Wan wanted 4n+1; this wants 17k+5, and the node's own tooltip gives the
    trained range as ~124-362 frames (124 is about 5 s at 24 fps). Asking for
    something off-grid gets silently snapped UP, which would make a clip longer
    than its narration beat, so the arithmetic happens here where it is visible.
    """
    n = int(round(seconds * fps))
    k = max(0, round((n - 5) / H3_FRAME_GRID))
    n = 5 + k * H3_FRAME_GRID
    return max(5, min(cap, n))


def minimax_h3_official(prompt: str, start_image: str, width: int = 1280,
                        height: int = 704, seconds: float = 5.17,
                        fps: float = 24.0, turbo: bool = True,
                        seed: int = 424242,
                        unet: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                        # the template names this qwen3vl_32b_minimax_h3_nvfp4_awq;
                        # it is the same file, renamed on this box
                        clip: str = "qwen3vl32b_h3_nvfp4.safetensors",
                        vae: str = "minimax_h3_video_vae_fp16.safetensors",
                        lora: str = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                        lora_strength: float = 1.0,
                        steps: int | None = None,
                        sampler: str = "res_multistep",
                        scheduler: str = "simple",
                        last_image: str | None = None,
                        guides: list | None = None,
                        distorch: dict | None = None,
                        sparse: dict | None = None,
                        prefix: str = "h3off") -> dict:
    """A faithful copy of ComfyUI's shipped MiniMax H3 image-to-video template.

    Transcribed node for node and link for link from
    `comfyui_workflow_templates_json/templates/video_minimax_h3_i2v.json`,
    subgraph "Image to Video (MiniMax H3)". Nothing here is inferred.

    This exists because `minimax_h3_i2v` above was built from prose
    documentation rather than from the shipped graph, and differs from it in
    almost every part that matters:

        official                              what was built instead
        ----------------------------------    ----------------------------
        SamplerCustomAdvanced + BasicGuider   KSampler
          + RandomNoise + BasicScheduler
        no negative conditioning at all       ConditioningZeroOut at cfg 1.0
        res_multistep                         euler
        turbo_8step LoRA at 6 steps           turbo_4step LoRA at 4 steps
        non-turbo at 20 steps                 n/a
        pruned_int8_convrot                   nvfp4
        video vae fp16                        video vae int8_convrot
        audio vae fp32 + VAEDecodeAudio       absent
        no ModelSamplingMiniMaxH3             one added unconditionally
        0.9 megapixel canvas                  1.03 MP (1344x768) in tests

    The important structural one is the guider. BasicGuider carries POSITIVE
    conditioning only - there is no negative input and no cfg field - whereas
    KSampler always runs its cfg path. At cfg 1.0 comfy/samplers.py sets
    uncond to None so the arithmetic should agree, but "should agree" is what
    the rest of this file kept assuming, so it is not assumed here.

    The turbo switch is a single boolean driving two ComfySwitchNodes in the
    template: on, the LoRA is applied and steps come from a primitive set to
    6; off, the raw UNET runs at 20. Note the LoRA shipped in the template is
    the EIGHT step distill run at SIX steps, not the four step one.

    Width and height default to 1280x704 because the outer workflow feeds the
    H3 node through ImageScaleToTotalPixels at 0.9 megapixels on a multiple of
    32, and 1280x704 is 901,120 pixels - 0.9 MP exactly.

    Output goes through VHS_VideoCombine rather than the template's
    CreateVideo + SaveVideo because the farm collects files by suffix, and the
    audio branch is dropped: this pipeline narrates separately and discards
    H3's generated audio. The video latent is decoded identically either way.
    """
    g = G()
    unet_args = {"unet_name": unet, "weight_dtype": "default"}
    if distorch:
        unet_node = g.add("UNETLoaderDisTorch2MultiGPU", {
            **unet_args,
            "compute_device": distorch.get("compute_device", "cuda:0"),
            "donor_device": distorch.get("donor_device", "cuda:1"),
            "virtual_vram_gb": distorch.get("virtual_vram_gb", 10.0),
            "expert_mode_allocations": distorch.get("expert", ""),
            "eject_models": distorch.get("eject_models", True)})
    else:
        unet_node = g.add("UNETLoader", unet_args)
    clip_node = g.add("CLIPLoader", {"clip_name": clip, "type": "minimax"})
    vae_node = g.add("VAELoader", {"vae_name": vae})

    # the template's two ComfySwitchNodes, resolved here rather than emitted:
    # a switch whose boolean is fixed at graph-build time is just a branch
    model = _link(unet_node)
    if turbo:
        lo = g.add("LoraLoaderModelOnly", {
            "model": model, "lora_name": lora, "strength_model": lora_strength})
        model = _link(lo)
    if steps is None:
        steps = 6 if turbo else 20

    if sparse:
        # Sol-Attn block sparse attention. Not in the template - added here
        # because it is the one speed lever that needs no weights. See the
        # note in minimax_h3_i2v for why the selection input is spelled this
        # way; passing the dict that execute() receives does not work.
        mode = sparse.get("mode", "sol-attn")
        sp_inputs = {
            "model": model,
            "selection": mode,
            "start_percent": sparse.get("start_percent", 0.2),
            "end_percent": sparse.get("end_percent", 1.0),
            "dense_blocks": sparse.get("dense_blocks", ""),
            "min_tokens": sparse.get("min_tokens", 12288),
            "extra_tokens": sparse.get("extra_tokens", 256),
            # exact_kv, not the node default exact_kv_and_rows: the extra mode
            # keeps GENERATED AUDIO intact and this pipeline discards it
            "sink_conditioning": sparse.get("sink_conditioning", "exact_kv"),
            "verbose": sparse.get("verbose", False)}
        if mode == "sol-attn":
            sp_inputs["selection.tau"] = sparse.get("tau", 1.3)
        else:
            sp_inputs["selection.keep_percent"] = sparse.get("keep_percent", 10.0)
        model = _link(g.add("BlockSparseAttention", sp_inputs))

    i2v_inputs = {
        "clip": _link(clip_node), "vae": _link(vae_node), "prompt": prompt,
        "width": width, "height": height,
        "length": h3_length(seconds, fps),
    }
    # start_image=None is pure text-to-video: MiniMaxH3ImageToVideo covers both
    # tasks (t2va and fl2va) on the same node, and the template's own note says
    # so - the keyframe inputs are optional and their absence IS the t2v mode.
    if start_image:
        img = g.add("LoadImage", {"image": start_image})
        i2v_inputs["first_frame"] = _link(img)
    if last_image:
        last = g.add("LoadImage", {"image": last_image})
        i2v_inputs["last_frame"] = _link(last)
    i2v = g.add("MiniMaxH3ImageToVideo", i2v_inputs)

    # Mid-clip keyframes. With first and last pinned, long clips (9-12 s) still wandered in the
    # middle on a painted piece: a tavern became a beach, a galley became a porthole view
    # (oceanus s023/s025, 2026-10-03). Each (image, frame_idx) is anchored with AddGuide.
    # A guide may also be a short clip, {"video": name, "frames": 22, "skip": 0, "idx": 0}: the tail of
    # the previous clip anchored at frame 0 carries its motion into this one (a continuation, not a
    # last-frame target). AddGuide crops a batch to 17k+5 frames (5, 22, 39 ...).
    cond = _link(i2v, 0)
    for gd in (guides or []):
        if isinstance(gd, dict):
            gi = g.add("VHS_LoadVideo", {
                "video": gd["video"], "force_rate": int(fps), "custom_width": 0, "custom_height": 0,
                "frame_load_cap": int(gd.get("frames", 22)), "skip_first_frames": int(gd.get("skip", 0)),
                "select_every_nth": 1})
            name_link, idx = _link(gi, 0), int(gd.get("idx", 0))
        else:
            name, idx = gd
            name_link = _link(g.add("LoadImage", {"image": name}))
        cond = _link(g.add("MiniMaxH3AddGuide", {
            "positive": cond, "vae": _link(vae_node), "latent": _link(i2v, 1),
            "image": name_link, "frame_idx": int(idx)}))

    noise = g.add("RandomNoise", {"noise_seed": seed})
    guider = g.add("BasicGuider", {"model": model,
                                   "conditioning": cond})
    samp = g.add("KSamplerSelect", {"sampler_name": sampler})
    sig = g.add("BasicScheduler", {"model": model, "scheduler": scheduler,
                                   "steps": steps, "denoise": 1.0})
    k = g.add("SamplerCustomAdvanced", {
        "noise": _link(noise), "guider": _link(guider),
        "sampler": _link(samp), "sigmas": _link(sig),
        "latent_image": _link(i2v, 1)})

    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae_node)})
    g.add("VHS_VideoCombine", {
        "images": _link(dec), "frame_rate": fps, "loop_count": 0,
        "filename_prefix": prefix, "format": "video/h264-mp4",
        "pingpong": False, "save_output": True})
    return g.out()


MULTIGPU_CLASSES = ("UNETLoaderDisTorch2MultiGPU",)


def minimax_h3_single_gpu(prompt: str, start_image: str, **kw) -> dict:
    """minimax_h3_official for ONE card of 24 GB or more (a rented pod, 2026-10-11).

    The same weights, nodes and settings; the only difference allowed is the loader: the plain
    UNETLoader, never ComfyUI-MultiGPU's DisTorch split, which needs a donor card the pod does not
    have. Any `distorch` argument is dropped. As of this date clay_film's motion_one and
    animate_shots already pass no distorch, so their graphs are byte-identical to this one; the
    variant exists so a remote worker can never be sent a two-card graph by a later change.
    """
    kw.pop("distorch", None)
    g = minimax_h3_official(prompt, start_image, distorch=None, **kw)
    bad = [n["class_type"] for n in g.values() if n["class_type"] in MULTIGPU_CLASSES]
    if bad:
        raise ValueError(f"single-GPU H3 graph carries multi-GPU nodes: {bad}")
    return g


H3_CLIP = "qwen3vl32b_h3_nvfp4.safetensors"
H3_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_UNET_NVFP4 = "MiniMax_H3_FL2VA_pruned_nvfp4.safetensors"  # banned for renders: no camera motion, crunchy texture (MEASURED.md 18 Sep; Patrick 16 Oct)
H3_UNET_INT8 = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"


def minimax_h3_encode(prompt: str, start_image: str | None, cond_path: str, width: int = 1344,
                      height: int = 768, seconds: float = 5.17, fps: float = 24.0,
                      last_image: str | None = None, guides: list | None = None,
                      clip: str = H3_CLIP, vae: str = H3_VAE) -> dict:
    """Split H3, step 1 of 2 (2026-10-15): the encoder half of minimax_h3_official, saved to cond_path.

    Node for node the same MiniMaxH3ImageToVideo + MiniMaxH3AddGuide chain; instead of sampling, the positive
    conditioning (text embedding plus every keyframe and guide latent) goes to SaveH3Conditioning
    (custom_nodes/h3_cond_cache.py). One worker encodes a whole wave and is then /free'd, so the 15.7 GB
    encoder is never resident beside a sampler. The AV latent is empty and is rebuilt by step 2."""
    g = G()
    clip_node = g.add("CLIPLoader", {"clip_name": clip, "type": "minimax"})
    vae_node = g.add("VAELoader", {"vae_name": vae})
    i2v_inputs = {"clip": _link(clip_node), "vae": _link(vae_node), "prompt": prompt,
                  "width": width, "height": height, "length": h3_length(seconds, fps)}
    if start_image:
        i2v_inputs["first_frame"] = _link(g.add("LoadImage", {"image": start_image}))
    if last_image:
        i2v_inputs["last_frame"] = _link(g.add("LoadImage", {"image": last_image}))
    i2v = g.add("MiniMaxH3ImageToVideo", i2v_inputs)
    cond = _link(i2v, 0)
    for gd in (guides or []):
        if isinstance(gd, dict):
            gi = g.add("VHS_LoadVideo", {
                "video": gd["video"], "force_rate": int(fps), "custom_width": 0, "custom_height": 0,
                "frame_load_cap": int(gd.get("frames", 22)), "skip_first_frames": int(gd.get("skip", 0)),
                "select_every_nth": 1})
            name_link, idx = _link(gi, 0), int(gd.get("idx", 0))
        else:
            name, idx = gd
            name_link = _link(g.add("LoadImage", {"image": name}))
        cond = _link(g.add("MiniMaxH3AddGuide", {
            "positive": cond, "vae": _link(vae_node), "latent": _link(i2v, 1),
            "image": name_link, "frame_idx": int(idx)}))
    g.add("SaveH3Conditioning", {"conditioning": cond, "path": cond_path})
    return g.out()


def minimax_h3_generate(cond_path: str, width: int = 1344, height: int = 768, seconds: float = 5.17,
                        fps: float = 24.0, seed: int = 424242, unet: str = H3_UNET_INT8,
                        vae: str = H3_VAE,
                        lora: str = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                        lora_strength: float = 1.0, steps: int = 6, sampler: str = "res_multistep",
                        scheduler: str = "simple", prefix: str = "h3gen", control: dict | None = None,
                        sparse: dict | None = None) -> dict:
    """Split H3, step 2 of 2: minimax_h3_official's sampler half, reading step 1's conditioning file.
    No text encoder is loaded. The latent comes from EmptyMiniMaxH3LatentAV, which builds exactly what
    MiniMaxH3ImageToVideo returns (both call _empty_av_latent with the same size and length).

    control (optional): the H3 Fun ControlNet Union 2.0 as a model patch, e.g. a clay depth pass,
    {"patch": "MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors", "video": "<uploaded mp4 or png>",
     "strength": 0.6, "start_percent": 0.0, "end_percent": 0.6}. A png is a held control frame.

    sparse (optional): BlockSparseAttention exactly as in minimax_h3_official, e.g. {"mode": "sol-attn"}. On
    15 Oct a step cost 193 s at 243 frames and 465 s at 362 (1.5x frames, 2.4x time), so attention dominates."""
    g = G()
    unet_node = g.add("UNETLoader", {"unet_name": unet, "weight_dtype": "default"})
    vae_node = g.add("VAELoader", {"vae_name": vae})
    model = _link(unet_node)
    if lora:
        model = _link(g.add("LoraLoaderModelOnly", {"model": model, "lora_name": lora,
                                                    "strength_model": lora_strength}))
    if sparse:
        mode = sparse.get("mode", "sol-attn")
        sp = {"model": model, "selection": mode, "start_percent": sparse.get("start_percent", 0.2),
              "end_percent": sparse.get("end_percent", 1.0), "dense_blocks": sparse.get("dense_blocks", ""),
              "min_tokens": sparse.get("min_tokens", 12288), "extra_tokens": sparse.get("extra_tokens", 256),
              "sink_conditioning": sparse.get("sink_conditioning", "exact_kv"),
              "verbose": sparse.get("verbose", False)}
        if mode == "sol-attn":
            sp["selection.tau"] = sparse.get("tau", 1.3)
        else:
            sp["selection.keep_percent"] = sparse.get("keep_percent", 10.0)
        model = _link(g.add("BlockSparseAttention", sp))
    if control:
        n = h3_length(seconds, fps)
        if str(control["video"]).lower().endswith((".png", ".jpg")):
            img = _link(g.add("LoadImage", {"image": control["video"]}))
            cv = _link(g.add("RepeatImageBatch", {"image": img, "amount": n}))
        else:
            cv = _link(g.add("VHS_LoadVideo", {
                "video": control["video"], "force_rate": int(fps), "custom_width": width, "custom_height": height,
                "frame_load_cap": n, "skip_first_frames": 0, "select_every_nth": 1}), 0)
        patch = g.add("ModelPatchLoader", {"name": control.get(
            "patch", "MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors")})
        model = _link(g.add("MiniMaxH3FunControlNetApply", {
            "model": model, "model_patch": _link(patch), "vae": _link(vae_node),
            "strength": float(control.get("strength", 0.6)),
            "start_percent": float(control.get("start_percent", 0.0)),
            "end_percent": float(control.get("end_percent", 0.6)), "control_video": cv}))
    cond = g.add("LoadH3Conditioning", {"path": cond_path})
    lat = g.add("EmptyMiniMaxH3LatentAV", {"width": width, "height": height, "length": h3_length(seconds, fps)})
    noise = g.add("RandomNoise", {"noise_seed": seed})
    guider = g.add("BasicGuider", {"model": model, "conditioning": _link(cond)})
    samp = g.add("KSamplerSelect", {"sampler_name": sampler})
    sig = g.add("BasicScheduler", {"model": model, "scheduler": scheduler, "steps": steps, "denoise": 1.0})
    k = g.add("SamplerCustomAdvanced", {"noise": _link(noise), "guider": _link(guider), "sampler": _link(samp),
                                        "sigmas": _link(sig), "latent_image": _link(lat)})
    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae_node)})
    g.add("VHS_VideoCombine", {"images": _link(dec), "frame_rate": fps, "loop_count": 0,
                               "filename_prefix": prefix, "format": "video/h264-mp4",
                               "pingpong": False, "save_output": True})
    return g.out()


def minimax_h3_i2v(prompt: str, start_image: str, width: int = 1280,
                   height: int = 704, length: int = 124, fps: float = 24.0,
                   steps: int = 4, cfg: float = 1.0,
                   shift_video: float = 12.0, shift_audio: float = 3.0,
                   seed: int = 424242, sampler: str = "euler",
                   scheduler: str = "simple",
                   # SUPERSEDED by minimax_h3_official, which is transcribed
                   # from ComfyUI's shipped template. Do not use this for new
                   # work. The three files it used to default to are deleted:
                   # nvfp4 does not execute camera moves at all, and the
                   # 4-step LoRA and int8 VAE are both superseded by what the
                   # template loads. Defaults now point at the surviving set
                   # so this still runs, but it will not run WELL.
                   unet: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                   clip: str = "qwen3vl32b_h3_nvfp4.safetensors",
                   clip_recovered: str | None = None,
                   vae: str = "minimax_h3_video_vae_fp16.safetensors",
                   lora: str | None = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                   lora_strength: float = 1.0,
                   last_image: str | None = None,
                   distorch: dict | None = None,
                   sparse: dict | None = None,
                   compile_backend: str | None = None,
                   prefix: str = "h3") -> dict:
    """MiniMax H3 (Hailuo 3.0) image-to-video, local weights.

    Open-weighted 2026-08-03 with day-zero ComfyUI support, which makes it
    thirteen months newer than the Wan 2.2 this pipeline was built on. Wan 2.7
    is not an option: there are no 2.7 weights on HuggingFace at all and
    ComfyUI's 2.7 nodes are cloud Partner Nodes.

    Why these particular files, on a 16 GB card:

      unet    the official repack's smallest is 21 GB, over the card. The
              PRUNED NVFP4 is 12.5 GB and fits. GGUF was tried first and is a
              dead end: ComfyUI-GGUF does not know this architecture. "Pruned"
              means layers were removed; that cost is unmeasured and is the
              main quality risk here.
      clip    Qwen3-VL-32B at nvfp4_awq, 15.7 GB. Note this needs ~16 GB of
              HOST ram to read, so two workers cannot both do it on a 33.5 GB
              box - that is what `hostbuf_file_reader_read failed` means. One
              card at a time until a memory-mapped encoder is in place.
      vae     int8 at 2.8 GB rather than fp16 at 5.2 GB, for headroom.

    Size is 1280x704, not the node's 1344x768 default. Z-Image renders its
    stills sharpest at 1280x720 (laplacian 500.8 against 182.3 at 1344x768),
    and 720 crops to 704 with no resampling because 704 is on the 32-pixel
    grid. Feeding H3 its "native" size meant upscaling every still and losing
    30% of the detail before sampling started.
      lora    the 4-step turbo distill, same idea as Wan's Lightning LoRA.

    Peak VRAM is sequential: ComfyUI loads the encoder, encodes, frees it, then
    loads unet+VAE to sample. ~15.7 GB peak, not the 31.7 GB that gets quoted
    for an everything-resident setup. So each card runs whole jobs and the farm
    layout does not change.

    `MiniMaxH3ImageToVideo` emits only `positive` and a latent - there is no
    negative output - so the negative is a zeroed copy, which is the cfg 1.0
    convention this model expects.
    """
    g = G()
    # UNETLoader, not UnetLoaderGGUF. ComfyUI-GGUF's IMG_ARCH_LIST is
    # {flux, sd1, sdxl, sd3, aura, hidream, cosmos, ltxv, hyvid, wan, lumina2,
    # qwen_image} - MiniMax H3 is not in it, so a GGUF unet fails outright with
    # "Unknown model architecture!". NVFP4 safetensors sidesteps that and is
    # natively supported on Blackwell, which is what these cards are.
    unet_args = {"unet_name": unet, "weight_dtype": "default"}
    if distorch:
        # ComfyUI-MultiGPU's DisTorch2 wrapper. It subclasses UNETLoader and
        # adds four optional inputs, so unet_name/weight_dtype are unchanged
        # and only the placement is new:
        #
        #   compute_device           where sampling runs
        #   donor_device             where the spare weights live
        #   virtual_vram_gb          how much to borrow from the donor
        #   expert_mode_allocations  explicit split, e.g. "cuda:0,8gb;cuda:1,13gb"
        #
        # This is what makes the 21 GB int8_convrot build runnable at all: it
        # does not fit on one 16 GB card, and picking quants that DO fit one
        # card is how this pipeline ended up on a pruned 4-bit repack whose own
        # LoRA author says to use the non-pruned INT8 instead.
        unet_node = g.add("UNETLoaderDisTorch2MultiGPU", {
            **unet_args,
            "compute_device": distorch.get("compute_device", "cuda:0"),
            "donor_device": distorch.get("donor_device", "cuda:1"),
            "virtual_vram_gb": distorch.get("virtual_vram_gb", 10.0),
            "expert_mode_allocations": distorch.get("expert", ""),
            "eject_models": distorch.get("eject_models", True)})
    else:
        unet_node = g.add("UNETLoader", unet_args)
    if clip_recovered:
        # SearchingMan's recovered 8B, via kgonia/ComfyUI-MiniMaxH3TextEncoders.
        # Their measured text-encoder peak VRAM: 6.18 GiB for the INT8 ConvRot
        # 8B against 16.04 GiB for the official 32B NVFP4/AWQ. That difference
        # is the whole reason both cards cannot currently run H3 at once - the
        # 32B needs ~16 GB of HOST ram to load and this box has 33.5 GB total.
        # Their own README says conditioning parity is argued numerically and
        # visual parity is NOT established, so this is a test, not a freebie.
        clip_node = g.add("MiniMaxH3TextEncoderLoader",
                          {"text_encoder": clip_recovered})
    else:
        clip_node = g.add("CLIPLoader", {"clip_name": clip, "type": "minimax"})
    vae_node = g.add("VAELoader", {"vae_name": vae})

    model = _link(unet_node)
    if lora:
        lo = g.add("LoraLoaderModelOnly", {
            "model": model, "lora_name": lora, "strength_model": lora_strength})
        model = _link(lo)
    # ALWAYS, and with both clocks. MiniMax H3 denoises video and audio on two
    # different flow schedules - video shift 12, audio shift 3 - and a plain
    # KSampler steps both on one. ComfyUI PR #15243 ("Fix sampler issues for
    # audio with minimax") exists for this, and the issue threads say almost
    # every "static noise" report traces back to that schedule mismatch.
    #
    # The previous version of this took a single `shift` argument, only applied
    # it when explicitly passed, and passed it under the key "shift" - which is
    # not a field on this node. So every clip generated before this line was a
    # dual-schedule model running on a single default schedule. Measured
    # consequence: noise in stationary blocks of 6.92 against a synthetic
    # film-grain reference of 1.76 and a still-frame floor of 0.00.
    ms = g.add("MiniMaxH3SigmaShift", {"model": model,
                                       "shift_video": shift_video,
                                       "shift_audio": shift_audio})
    model = _link(ms)
    # Block sparse attention, merged into ComfyUI core 2026-09-06 and reported
    # to cut H3 render time by up to 4x. `sol-attn` is threshold-based and
    # needs no special checkpoint; `sla` and `vsa` expect a matching distilled
    # LoRA or a FastH3-VSA checkpoint. start_percent 0.2 leaves the early
    # structure-setting steps dense, which is why it can be close to free.
    #
    # Deliberately NOT SageAttention: ComfyUI issue #15263 reports its FP8 PV
    # kernels producing noise above ~160k tokens on sm_120, and sm_120 is
    # exactly what these cards are.
    if sparse:
        # io.DynamicCombo in API format: the chosen mode is a plain STRING
        # under the parent id, and that mode's own field is a SIBLING key
        # prefixed with the parent id - "selection" and "selection.tau", per
        # _io.py: "nested inputs are prefixed with parent ID to match frontend
        # naming convention". Passing the assembled dict that execute() ends up
        # receiving does NOT work: the expansion never runs, no kwarg is built,
        # and it dies at execute time with "missing 1 required positional
        # argument: 'selection'". Probed all three forms against a live worker:
        #   selection + selection.tau   executed
        #   selection + tau             rejected at validation
        #   nested dict                 missing positional argument
        mode = sparse.get("mode", "sol-attn")
        sp_inputs = {
            "model": model,
            "selection": mode,
            "start_percent": sparse.get("start_percent", 0.2),
            "end_percent": sparse.get("end_percent", 1.0),
            "dense_blocks": sparse.get("dense_blocks", ""),
            "min_tokens": sparse.get("min_tokens", 12288),
            "extra_tokens": sparse.get("extra_tokens", 256),
            "sink_conditioning": sparse.get("sink_conditioning", "exact_kv"),
            "verbose": sparse.get("verbose", False)}
        if mode == "sol-attn":
            # threshold in score-distribution sigmas; higher is sparser.
            # 1.0 keeps ~16% of key blocks exact, 1.3 ~10%, 1.5 ~7%. 1.3 is
            # the H3-tuned default, above the paper's 1.0.
            sp_inputs["selection.tau"] = sparse.get("tau", 1.3)
        else:
            # sla and vsa both want weights distilled against their pattern
            sp_inputs["selection.keep_percent"] = sparse.get("keep_percent", 10.0)
        sp = g.add("BlockSparseAttention", sp_inputs)
        model = _link(sp)
    if compile_backend:
        tc = g.add("TorchCompileModel", {"model": model,
                                         "backend": compile_backend})
        model = _link(tc)

    img = g.add("LoadImage", {"image": start_image})
    i2v_inputs = {
        "clip": _link(clip_node), "vae": _link(vae_node), "prompt": prompt,
        "width": width, "height": height, "length": length,
        "first_frame": _link(img),
    }
    if last_image:
        last = g.add("LoadImage", {"image": last_image})
        i2v_inputs["last_frame"] = _link(last)
    i2v = g.add("MiniMaxH3ImageToVideo", i2v_inputs)

    neg = g.add("ConditioningZeroOut", {"conditioning": _link(i2v, 0)})
    k = g.add("KSampler", {
        "model": model, "positive": _link(i2v, 0), "negative": _link(neg),
        "latent_image": _link(i2v, 1), "seed": seed, "steps": steps,
        "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
        "denoise": 1.0})

    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae_node)})
    g.add("VHS_VideoCombine", {
        "images": _link(dec), "frame_rate": fps, "loop_count": 0,
        "filename_prefix": prefix, "format": "video/h264-mp4",
        "pingpong": False, "save_output": True})
    return g.out()


# ------------------------------------------------------------- Flux.2 t2i ---

def flux2_t2i(prompt: str, width: int = 1344, height: int = 768,
              steps: int = 20, cfg: float = 1.0, seed: int = 424242,
              sampler: str = "euler", scheduler: str = "simple",
              unet: str = "flux-2-klein-4b-fp8.safetensors",
              clip: str = "qwen_3_4b.safetensors",
              vae: str = "flux2-vae.safetensors",
              prefix: str = "flux2") -> dict:
    """Flux.2 Klein 4B. Every file for this was already on the disk.

    Z-Image Turbo is a distilled speed model from November 2025, and a
    side-by-side of its still against H3's output showed the waxy skin and
    plastic highlights are present BEFORE the video stage touches anything.
    The video model was faithfully animating an artefacty image, so every fix
    applied downstream of the still was polishing the wrong thing.

    Klein is Apache 2.0, January 2026, and shares Z-Image's `qwen_3_4b` text
    encoder, so this costs one graph and no download.
    """
    g = G()
    unet_node = g.add("UNETLoader", {"unet_name": unet,
                                     "weight_dtype": "default"})
    clip_node = g.add("CLIPLoader", {"clip_name": clip, "type": "flux2"})
    vae_node = g.add("VAELoader", {"vae_name": vae})

    pos = g.add("CLIPTextEncode", {"clip": _link(clip_node), "text": prompt})
    neg = g.add("ConditioningZeroOut", {"conditioning": _link(pos)})
    latent = g.add("EmptySD3LatentImage", {"width": width, "height": height,
                                           "batch_size": 1})
    k = g.add("KSampler", {
        "model": _link(unet_node), "positive": _link(pos),
        "negative": _link(neg), "latent_image": _link(latent), "seed": seed,
        "steps": steps, "cfg": cfg, "sampler_name": sampler,
        "scheduler": scheduler, "denoise": 1.0})
    dec = g.add("VAEDecode", {"samples": _link(k), "vae": _link(vae_node)})
    g.add("SaveImage", {"images": _link(dec), "filename_prefix": prefix})
    return g.out()
