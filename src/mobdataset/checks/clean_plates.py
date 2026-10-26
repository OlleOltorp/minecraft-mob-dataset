"""Check a capture's labels against its clean plates (the same shot without our mobs).

The mobs we placed are the only thing that differs between a picture and its clean plate,
so their label pixels must have changed, and no sizeable change may be left unlabelled
(shadows under mobs excepted: they only darken). Mobs that came with the world are in
both shots and must not change at all.

Per picture it measures:
    recall      share of placed mobs' lit label pixels that changed (low: label, no mob);
    unlabelled  biggest changed blob away from every label that isn't only darker (high: a
                mob drawn but not labelled, or the shots differ in something else);
    natives     changed share of the pixels of mobs that are in both shots.

Prints a summary, writes the worst pictures side by side (picture, clean plate, change in
red over label outlines) to data/checks/clean-plates/<capture>/, and exits 1 if any picture
fails the limits.

Usage:
    mobdataset check clean-plates <capture>
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from mobdataset.game.label_client import OTHER, Shot, load_shot
from mobdataset.paths import DATA

CHANGE = 12  # per-channel difference that counts as changed (JPEG-free PNGs, so small)
MIN_MOB_PIXELS = 50  # mobs smaller than this are too noisy to judge on their own
LIT = 40  # brighter of the two shots (0-255) above which a mob must show as a change


def changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pixels whose colour differs by more than CHANGE in any channel."""
    return (np.abs(a.astype(np.int16) - b.astype(np.int16)) > CHANGE).any(axis=2)


def check(folder: Path, stem: str) -> dict:
    """Measure one picture against its clean plate.

    Args:
        folder: The capture's pictures folder.
        stem: The picture's file stem.

    Returns:
        The measures (see the module docstring) and the worst mobs.
    """
    shot = load_shot(folder, stem, timeout=0)
    clean = load_shot(folder, f"{stem}_clean", timeout=0)
    in_clean = {m["uuid"] for m in clean.info["mobs"]}
    placed = [m for m in shot.info["mobs"] if m["uuid"] not in in_clean]
    natives = [m for m in shot.info["mobs"] if m["uuid"] in in_clean]
    diff = changed(shot.picture, clean.picture)
    labels = shot.labels
    # In the dark a mob and what is behind it can look the same; only judge lit pixels.
    lit = np.maximum(shot.picture.max(2), clean.picture.max(2)) > LIT
    placed_mask = np.isin(labels, [m["colour"] for m in placed]) & lit
    native_mask = np.isin(labels, [m["colour"] for m in natives])

    worst_mobs = []
    for m in placed:
        mask = (labels == m["colour"]) & lit
        n = int(mask.sum())
        if n >= MIN_MOB_PIXELS:
            worst_mobs.append((float(diff[mask].mean()), n, m["type"][10:]))
    worst_mobs.sort()

    # Other entities (armour stands we placed) change too but are unlabelled on purpose.
    near_label = ndimage.binary_dilation(labels > 0, iterations=4)
    loose = diff & ~near_label
    darker = shot.picture.astype(np.int16).sum(2) < clean.picture.astype(np.int16).sum(2)
    blobs, count = ndimage.label(ndimage.binary_closing(loose, iterations=2))
    biggest = 0
    if count:
        sizes = ndimage.sum(loose, blobs, range(1, count + 1))
        brighter = ndimage.mean(~darker, blobs, range(1, count + 1))
        # A shadow only darkens; anything that brightens a sizeable part is something else.
        biggest = int(max((s for s, b in zip(sizes, brighter, strict=True) if b > 0.2), default=0))
    return {
        "stem": stem,
        "placed": len(placed),
        "recall": float(diff[placed_mask].mean()) if placed_mask.sum() >= MIN_MOB_PIXELS else None,
        "worst_mob": worst_mobs[0] if worst_mobs else None,
        "unlabelled": biggest,
        "natives": float(diff[native_mask].mean()) if native_mask.any() else None,
        "picture_changed": float(diff.mean()),
    }


def draw(folder: Path, stem: str, out: Path, note: str) -> None:
    """Save the picture, clean plate and change (red) over label outlines (yellow) side by side.

    Args:
        folder: The capture's pictures folder.
        stem: The picture's file stem.
        out: Folder for the sheet.
        note: Text written above the pictures.
    """
    shot: Shot = load_shot(folder, stem, timeout=0)
    clean = load_shot(folder, f"{stem}_clean", timeout=0)
    diff = changed(shot.picture, clean.picture)
    lab = (shot.labels > 0) & (shot.labels != OTHER)
    edge = lab & ~ndimage.binary_erosion(lab)
    view = (shot.picture * 0.35).astype(np.uint8)
    view[diff] = (255, 40, 40)
    view[ndimage.binary_dilation(edge)] = (255, 230, 0)
    h, w = shot.picture.shape[:2]
    sheet = Image.new("RGB", (w * 3, h + 20), (0, 0, 0))
    for i, img in enumerate((shot.picture, clean.picture, view)):
        sheet.paste(Image.fromarray(img), (i * w, 20))
    ImageDraw.Draw(sheet).text((6, 4), f"{stem}  {note}", fill=(255, 255, 255))
    sheet.resize((w * 3 // 2, (h + 20) // 2)).save(out / f"{stem}.jpg", quality=85)


def main(argv: Sequence[str] | None = None) -> None:
    """Check every picture of a capture that has a clean plate.

    Args:
        argv: Command-line arguments; default sys.argv.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="capture name under data/captures/")
    p.add_argument("--min-recall", type=float, default=0.8, help="per picture, placed mobs")
    p.add_argument("--max-unlabelled", type=int, default=150, help="pixels in one blob")
    p.add_argument("--max-natives", type=float, default=0.05, help="changed share")
    p.add_argument("--show", type=int, default=12, help="worst pictures to draw")
    args = p.parse_args(argv)
    folder = DATA / "captures" / args.capture / "pictures"
    stems = sorted(f.name[: -len("_clean.json")] for f in folder.glob("*_clean.json"))
    if not stems:
        raise SystemExit(f"no clean plates in {folder}")
    rows = [check(folder, s) for s in stems]

    def fails(r: dict) -> list[str]:
        why = []
        if r["recall"] is not None and r["recall"] < args.min_recall:
            why.append(f"recall {r['recall']:.2f}")
        if r["unlabelled"] > args.max_unlabelled:
            why.append(f"unlabelled blob {r['unlabelled']} px")
        if r["natives"] is not None and r["natives"] > args.max_natives:
            why.append(f"natives changed {r['natives']:.2f}")
        return why

    bad = [(r, fails(r)) for r in rows if fails(r)]
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    mob_recalls = sorted(r["worst_mob"] for r in rows if r["worst_mob"])
    print(f"{args.capture}: {len(rows)} pictures with clean plates, {len(bad)} fail")
    if recalls:
        print(
            f"  recall of placed mobs' labels: median {np.median(recalls):.3f}, "
            f"5th percentile {np.percentile(recalls, 5):.3f}"
        )
    print(f"  worst single mobs (changed share, pixels, kind): {mob_recalls[:8]}")
    unl = [r["unlabelled"] for r in rows]
    print(f"  biggest unlabelled blob: median {np.median(unl):.0f} px, max {max(unl)} px")
    nat = [r["natives"] for r in rows if r["natives"] is not None]
    if nat:
        print(f"  natives changed share: max {max(nat):.3f} over {len(nat)} pictures")
    out = DATA / "checks" / "clean-plates" / args.capture
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    worst = sorted(rows, key=lambda r: (r["recall"] or 1.0) - r["unlabelled"] / 1e4)
    shown = bad or [(r, []) for r in worst]
    for r, why in shown[: args.show]:
        draw(folder, r["stem"], out, ", ".join(why) or "worst of the passing")
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r, why in bad[:20]:
        print(f"  FAIL {r['stem']}: {', '.join(why)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
