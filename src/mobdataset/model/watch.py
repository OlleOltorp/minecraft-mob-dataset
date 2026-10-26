"""Live page for a training run: loss, learning rate, validation mAP, progress and GPU.

Serves a page that refreshes itself every 15 s from the newest run folder under runs/
(metrics.csv and run_config.json), the job queue's log and nvidia-smi. Progress is
the logged step against the run's total steps; speed is measured here between refreshes
(the training log has no progress bar when it is not a terminal). Standard library only,
so it runs next to training without touching its environment.

    mobdataset watch --port 8766    (then open http://localhost:8766)
"""

import argparse
import csv
import json
import math
import subprocess
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mobdataset.paths import DATA, LOGS, ROOT

SPEED_WINDOW = 600.0  # seconds of step readings the speed is measured over
_seen: list[tuple[float, float]] = []  # (time, step) readings of the current run
_images: dict[str, int] = {}  # train images per dataset folder


def newest(pattern: str) -> Path | None:
    """The most recently changed path matching a glob under the data folder, or None."""
    found = sorted(DATA.glob(pattern), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def number(text: str | None) -> float | None:
    """A CSV cell as a finite number, or None."""
    try:
        value = float(text or "")
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def metrics(csv_path: Path) -> dict[str, list[list[float]]]:
    """Series from metrics.csv: [x, y] points per column we plot.

    Train columns are plotted against the step, validation ones against the epoch.
    """
    series: dict[str, list[list[float]]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step, epoch = number(row.get("step")), number(row.get("epoch"))
            for key, cell in row.items():
                value = number(cell)
                if value is None or not key:
                    continue
                if key.startswith("train/loss") or key == "train/lr":
                    x = step
                elif key.startswith("val/") and ("mAP" in key or key == "val/loss"):
                    x = epoch
                else:
                    continue
                if x is not None:
                    series.setdefault(key, []).append([x, value])
    return series


def last_step(series: dict[str, list[list[float]]]) -> float | None:
    """The newest logged training step, or None before the first one."""
    steps = [pts[-1][0] for key, pts in series.items() if key.startswith("train/") and pts]
    return max(steps) if steps else None


def train_images(dataset: Path) -> int:
    """Pictures in a dataset's train split (counted once per folder)."""
    key = str(dataset)
    if key not in _images:
        coco = json.loads((dataset / "train" / "_annotations.coco.json").read_text("utf-8"))
        _images[key] = len(coco["images"])
    return _images[key]


def progress(run: Path, step: float | None) -> dict[str, Any] | None:
    """Where the run is: epoch, steps, share done, speed and time left."""
    config_path = run / "run_config.json"
    if step is None or not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = Path(config["dataset_dir"])
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    batches = math.ceil(train_images(dataset) / config["batch_size"])
    per_epoch = math.ceil(batches / config["grad_accum_steps"])  # optimiser steps
    total = per_epoch * config["epochs"]
    now = time.time()
    if _seen and step < _seen[-1][1]:
        _seen.clear()  # a new run started
    if not _seen or step != _seen[-1][1]:
        _seen.append((now, step))
    while len(_seen) > 2 and now - _seen[0][0] > SPEED_WINDOW:
        _seen.pop(0)
    rate = None
    if len(_seen) >= 2 and _seen[-1][0] > _seen[0][0]:
        rate = (_seen[-1][1] - _seen[0][1]) / (_seen[-1][0] - _seen[0][0])
    left = (total - step) / rate if rate else None
    return {
        "epoch": int(step // per_epoch) + 1,
        "epochs": config["epochs"],
        "step": int(step),
        "steps": total,
        "percent": round(100 * step / total, 1),
        "rate": f"{rate * 60:.0f} steps/min" if rate else "measuring",
        "left": f"{int(left // 3600)} h {int(left % 3600 // 60):02d} min" if left else "?",
        "batch": f"{config['batch_size']} x {config['grad_accum_steps']}",
    }


def gpu() -> dict[str, str] | None:
    """GPU use from nvidia-smi, or None without one."""
    query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    util, used, total, temp, power = [v.strip() for v in out.splitlines()[0].split(",")]
    return {"util": util, "memory": f"{used} / {total} MiB", "temp": temp, "power": power}


def status() -> dict[str, Any]:
    """Everything the page shows, as JSON."""
    run = newest("runs/*/metrics.csv")
    series = metrics(run) if run else {}
    queue = LOGS / "tower-job.log"  # the job queue's log on the GPU machine
    queue_lines = []
    if queue.exists():
        lines = queue.read_text(encoding="utf-8", errors="replace").splitlines()
        queue_lines = [ln[:200] for ln in lines if ln[:1].isalnum()][-8:]
    return {
        "run": run.parent.name if run else None,
        "series": series,
        "progress": progress(run.parent, last_step(series)) if run else None,
        "gpu": gpu(),
        "queue": queue_lines,
    }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Training</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
body { background: #111; color: #ddd; font: 14px system-ui, sans-serif; margin: 16px; }
h1 { font-size: 18px; margin: 0 0 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
.card { background: #1b1b1b; border-radius: 8px; padding: 12px; }
.bar { height: 10px; background: #333; border-radius: 5px; overflow: hidden; margin: 6px 0; }
.bar div { height: 100%; background: #4c9be8; }
pre { white-space: pre-wrap; margin: 0; font-size: 12px; color: #aaa; }
.k { color: #888; }
</style></head><body>
<h1 id="title">Training</h1>
<div class="grid">
  <div class="card"><div id="prog"></div><div class="bar"><div id="fill"></div></div>
    <div id="gpu"></div></div>
  <div class="card"><pre id="queue"></pre></div>
  <div class="card"><canvas id="loss"></canvas></div>
  <div class="card"><canvas id="map"></canvas></div>
  <div class="card"><canvas id="lr"></canvas></div>
</div>
<script>
const colours = ["#4c9be8", "#e8a33c", "#5cc48a", "#d9605a", "#a77de0", "#c9c95a"];
function chart(id, title, xLabel, log) {
  return new Chart(document.getElementById(id), {
    type: "line", data: { datasets: [] },
    options: { animation: false, parsing: false, normalized: true,
      plugins: { title: { display: true, text: title, color: "#ddd" },
                 legend: { labels: { color: "#bbb", boxWidth: 12 } } },
      scales: { x: { type: "linear", title: { display: true, text: xLabel, color: "#888" },
                     ticks: { color: "#888" }, grid: { color: "#2a2a2a" } },
                y: { type: log ? "logarithmic" : "linear",
                     ticks: { color: "#888" }, grid: { color: "#2a2a2a" } } } } });
}
const charts = { loss: chart("loss", "Loss", "step", true),
                 map: chart("map", "Validation mAP (per epoch)", "epoch", false),
                 lr: chart("lr", "Learning rate", "step", false) };
function fill(c, series, keys) {
  c.data.datasets = keys.filter(k => series[k]).map((k, i) => ({
    label: k, data: series[k].map(([x, y]) => ({ x, y })), borderColor: colours[i % 6],
    backgroundColor: colours[i % 6], pointRadius: series[k].length < 30 ? 3 : 0,
    borderWidth: 1.5 }));
  c.update();
}
async function refresh() {
  const s = await (await fetch("/data")).json();
  document.getElementById("title").textContent =
    "Training: " + (s.run || "no run yet");
  const keys = Object.keys(s.series);
  fill(charts.loss, s.series, keys.filter(k => k.startsWith("train/loss") &&
       !k.includes("aux") && !k.includes("_0") && !k.includes("_1") && !k.includes("_2")
       && !k.includes("enc")).concat(["val/loss"]));
  fill(charts.map, s.series, keys.filter(k => k.startsWith("val/") && k.includes("mAP")));
  fill(charts.lr, s.series, ["train/lr"]);
  const p = s.progress;
  document.getElementById("prog").innerHTML = p
    ? `<b>Epoch ${p.epoch} of ${p.epochs}</b> &nbsp; step ${p.step} / ${p.steps}
       (${p.percent}%) &nbsp; ${p.rate} &nbsp; <span class="k">left</span> ${p.left}
       &nbsp; <span class="k">batch</span> ${p.batch}` : "No steps logged yet";
  document.getElementById("fill").style.width = (p ? p.percent : 0) + "%";
  const g = s.gpu;
  document.getElementById("gpu").innerHTML = g
    ? `<span class="k">GPU</span> ${g.util}% &nbsp; <span class="k">memory</span> ${g.memory}
       &nbsp; ${g.temp} °C &nbsp; ${g.power} W` : "No GPU reading";
  document.getElementById("queue").textContent = s.queue.join("\\n");
}
refresh(); setInterval(refresh, 15000);
</script></body></html>
"""


def main(argv: Sequence[str] | None = None) -> None:
    """Serve the page until stopped."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args(argv)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            if self.path == "/data":
                body, kind = json.dumps(status()).encode("utf-8"), "application/json"
            else:
                body, kind = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass  # one line per refresh is noise

    print(f"serving on http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
