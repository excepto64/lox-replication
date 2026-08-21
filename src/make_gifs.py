#!/usr/bin/env python3
"""Stitch per-step graph.py PDF outputs into per-run GIFs.

Groups PDFs by (plot type, run identifier, weights-vs-activations), where the
run identifier is whatever graph.py used as model_local (fine-tune name,
method, optimiser, seed, etc. all folded in) -- so different runs are picked
up automatically without hardcoding any run-specific naming.
"""
import re, subprocess, glob, os, argparse
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# When run from src/, default to the sibling results/ dir; when run from a
# copy placed directly in results/ (e.g. to sweep up leftover runs), default
# to the script's own directory.
if os.path.basename(SCRIPT_DIR) == "src":
    DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
else:
    DEFAULT_RESULTS_DIR = SCRIPT_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
parser.add_argument("--duration-ms", type=int, default=1000, help="Equal time per step.")
parser.add_argument("--run", type=str, default=None, help="Only stitch GIFs for this run identifier (matches graph.py's model_local, e.g. measure_update.sh's local_name). Default: all runs found in results-dir.")
args = parser.parse_args()

RESULTS_DIR = args.results_dir
OUT_DIR = os.path.join(RESULTS_DIR, "gifs")
TMP_DIR = os.path.join(RESULTS_DIR, "gif_frames")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

FONT_SIZE = 36
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", FONT_SIZE)
except OSError:
    FONT = ImageFont.load_default()


def label_step(img, step):
    """Return a copy of img with a "Step {step}" label drawn in the top-left corner."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    text = f"Step {step}"
    pad = 10
    bbox = draw.textbbox((0, 0), text, font=FONT)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = pad, pad
    draw.rectangle([x - 6, y - 4, x + w + 6, y + h + 12], fill="white")
    draw.text((x, y), text, fill="black", font=FONT)
    return img

# Matches graph.py's savefig naming: f"{plot}_{model_local}{tag}.pdf" where
# model_local ends in "..._step-{n}" (revision) and tag is "" or "_dWX".
# `run` captures everything identifying the run -- fine-tune name, method,
# optimiser, seed, etc -- up to the trailing step suffix.
pattern = re.compile(
    r"^(?P<plot>average_singular_value|lorenz_curve|cumulative_proportion)_"
    r"(?P<run>.+)_step-(?P<step>\d+)(?P<dwx>_dWX)?\.pdf$"
)

series = defaultdict(list)  # key -> [(step, path)]

for path in glob.glob(os.path.join(RESULTS_DIR, "*.pdf")):
    fname = os.path.basename(path)
    m = pattern.match(fname)
    if not m:
        continue
    if args.run is not None and m.group("run") != args.run:
        continue
    key = (m.group("plot"), m.group("run"), bool(m.group("dwx")))
    series[key].append((int(m.group("step")), path))

print(f"Found {len(series)} series")

for (plot, run, dwx), items in sorted(series.items()):
    items.sort(key=lambda x: x[0])
    frames = []
    for step, pdf_path in items:
        png_prefix = os.path.join(TMP_DIR, f"{plot}_{run}{'_dWX' if dwx else ''}_step-{step}")
        subprocess.run(
            ["pdftoppm", "-png", "-r", "150", "-singlefile", pdf_path, png_prefix],
            check=True,
        )
        frame = Image.open(png_prefix + ".png").convert("RGB")
        frames.append(label_step(frame, step))

    tag = "_dWX" if dwx else ""
    out_name = f"{plot}_{run}{tag}.gif"
    out_path = os.path.join(OUT_DIR, out_name)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
    )
    print(f"Wrote {out_path} ({len(frames)} frames)")

print("Done.")
