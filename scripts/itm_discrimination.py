"""Can an image-text model tell "the target is in view" from "it is not"?

A value map is only worth its compute if its per-frame score separates those two
cases. Measured for CLIP ViT-B/32 on the 500-episode run, it barely does: mean
per-episode max score 0.2571 on successes vs 0.2530 on failures, a 0.004 gap,
and the value map produced no SR change (44.6% -> 44.8%, 44 gained / 43 lost).

This tests whether a stronger encoder (BLIP-2 ITM, what VLFM and ASCENT actually
use) separates them, before paying 4 GB of VRAM to find out in a full eval.

Labels come from the runs themselves, no annotation needed:
  positive = LAST keyframe of a SUCCESSFUL episode. Success means the agent
             stopped within the goal view-point set, so the target is in view.
  negative = FIRST keyframe of the same episode -- a random start pose, where
             the target is almost never visible.

Same episodes for both classes, so scene, lighting and category are controlled;
the only variable is whether the target is there.

Usage:
    python scripts/itm_discrimination.py outputs/v1_valuemap500 --limit 120
"""
import argparse
import glob
import json
import os
import statistics as st
import sys


def load_pairs(run_dir, limit):
    eps = [json.loads(l) for l in open(os.path.join(run_dir, "episodes.jsonl"))]
    out = []
    for r in eps:
        if not r["success"]:
            continue
        tag = f"{r['scene'].split('.')[0]}_ep{r['episode_id']}"
        kfs = sorted(glob.glob(os.path.join(run_dir, "keyframes", tag, "kf_*.jpg")))
        if len(kfs) < 5:
            continue
        out.append((r["target"].replace("_", " "), kfs[-1], kfs[0]))
        if len(out) >= limit:
            break
    return out


def auc(pos, neg):
    """Probability a random positive outscores a random negative (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def report(name, pos, neg):
    gap = st.mean(pos) - st.mean(neg)
    sd = st.pstdev(pos + neg) or 1e-9
    print(f"\n{name}")
    print(f"  target in view : mean {st.mean(pos):.4f}  sd {st.pstdev(pos):.4f}")
    print(f"  target absent  : mean {st.mean(neg):.4f}  sd {st.pstdev(neg):.4f}")
    print(f"  gap {gap:+.4f}   effect size {gap / sd:+.2f} sd   AUC {auc(pos, neg):.3f}")
    print(f"  (AUC 0.5 = no discrimination, 1.0 = perfect)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--skip-blip", action="store_true")
    args = ap.parse_args()

    pairs = load_pairs(args.run_dir, args.limit)
    if not pairs:
        sys.exit("no successful episodes with keyframes found")
    print(f"{len(pairs)} successful episodes; scoring last (target in view) vs first (absent)")

    from PIL import Image

    # ---- CLIP ViT-B/32, the encoder actually used in the value map run ----
    import clip
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=dev)
    cpos, cneg = [], []
    with torch.no_grad():
        for target, p_img, n_img in pairs:
            tok = clip.tokenize([f"a photo of a {target}"]).to(dev)
            tf = model.encode_text(tok)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            for path, dst in ((p_img, cpos), (n_img, cneg)):
                im = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
                f = model.encode_image(im)
                f = f / f.norm(dim=-1, keepdim=True)
                dst.append(float((f @ tf.T).item()))
    report("CLIP ViT-B/32  'a photo of a {target}'", cpos, cneg)
    del model
    torch.cuda.empty_cache()

    if args.skip_blip:
        return

    # ---- BLIP-2 ITM, what VLFM/ASCENT use ----
    from transformers import AutoProcessor, Blip2ForImageTextRetrieval

    name = "Salesforce/blip2-itm-vit-g"
    proc = AutoProcessor.from_pretrained(name)
    blip = Blip2ForImageTextRetrieval.from_pretrained(
        name, torch_dtype=torch.float16
    ).to(dev).eval()

    # Both heads: VLFM/ASCENT call the method `cosine`, which is the CONTRASTIVE
    # (ITC) head, not the matching (ITM) head. Test both so the comparison is
    # fair rather than resting on a guess about their setup.
    heads = {"ITM head": True, "contrastive (ITC) head": False}
    for label, use_itm in heads.items():
        bpos, bneg = [], []
        with torch.no_grad():
            for target, p_img, n_img in pairs:
                text = f"Seems like there is a {target} ahead."
                for path, dst in ((p_img, bpos), (n_img, bneg)):
                    inp = proc(
                        images=Image.open(path).convert("RGB"), text=text,
                        return_tensors="pt",
                    ).to(dev, torch.float16)
                    out = blip(**inp, use_image_text_matching_head=use_itm)
                    lg = out.logits_per_image.float()
                    dst.append(
                        float(torch.softmax(lg, dim=-1)[0, 1]) if use_itm
                        else float(lg.max())
                    )
        report(f"BLIP-2 {label}  'Seems like there is a {{target}} ahead.'", bpos, bneg)


if __name__ == "__main__":
    main()
