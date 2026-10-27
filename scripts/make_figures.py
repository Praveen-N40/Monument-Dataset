# -*- coding: utf-8 -*-
"""Generate the paper figures (300 dpi) from the yolov5s_full run on dataset_v2.
Outputs to D:/Code/paper_experiments/figures and writes figures/metrics_extra.json
(F1-optimal confidence threshold computed from test-set predictions)."""
import os, sys, json, glob, subprocess, random, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

EXP = Path("D:/Code/paper_experiments"); Y5 = Path("D:/Code/yolov5")
DS = EXP / "dataset_v2"; RUN = EXP / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "yolov5s_full")
FIG = EXP / "figures"; FIG.mkdir(exist_ok=True)
NAMES = [r[1] for r in json.load(open(DS / "stats.json"))["classes"]]
random.seed(0)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

def font(sz):
    for f in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()

PALETTE = [(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),(140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207)]
def color(c): return PALETTE[c % len(PALETTE)]

def draw_boxes(im, boxes, labels, confs=None):
    d = ImageDraw.Draw(im); W, H = im.size; f = font(max(14, W // 28)); lw = max(2, W // 220)
    for k, (b, l) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = b; col = color(l)
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        txt = NAMES[l] + (f" {confs[k]:.2f}" if confs is not None else "")
        tw, th = d.textbbox((0, 0), txt, font=f)[2:]
        ty = max(0, y1 - th - 2)
        d.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=col)
        d.text((x1 + 2, ty), txt, fill="white", font=f)
    return im

def yolo_to_xyxy(lines, W, H):
    boxes, labels, confs = [], [], []
    for line in lines:
        q = line.split()
        if len(q) < 5: continue
        c, x, y, w, h = int(q[0]), *map(float, q[1:5])
        boxes.append([(x - w/2)*W, (y - h/2)*H, (x + w/2)*W, (y + h/2)*H]); labels.append(c)
        confs.append(float(q[5]) if len(q) > 5 else None)
    return boxes, labels, confs

def grid(images, cols, cell=520, pad=6, out=None):
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell + (cols + 1) * pad, rows * cell + (rows + 1) * pad), "white")
    for i, im in enumerate(images):
        im = im.copy(); im.thumbnail((cell, cell))
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad) + (cell - im.width) // 2; y = pad + r * (cell + pad) + (cell - im.height) // 2
        canvas.paste(im, (x, y))
    canvas.save(out, dpi=(300, 300)); return canvas

# ---------------------------------------------------------------- Fig 2: annotated training samples
def fig_samples():
    picks = []
    for pref, n in (("charminar", 3), ("gatewayofindia", 2), ("mysore_palace", 2), ("Tagmahal", 2)):
        files = sorted((DS / "images" / "train").glob(pref + "*"))
        # prefer images with many distinct classes
        def nclasses(p):
            lp = DS / "labels" / "train" / (p.stem + ".txt")
            return len({l.split()[0] for l in open(lp) if l.strip()})
        files.sort(key=nclasses, reverse=True)
        picks += files[:n]
    ims = []
    for p in picks:
        im = Image.open(p).convert("RGB"); W, H = im.size
        b, l, _ = yolo_to_xyxy(open(DS / "labels" / "train" / (p.stem + ".txt")).read().splitlines(), W, H)
        ims.append(draw_boxes(im, b, l))
    grid(ims, 3, out=FIG / "fig2_samples.png")

# ---------------------------------------------------------------- Fig 3: training curves
def fig_curves():
    import csv
    rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(open(RUN / "results.csv"))]
    ep = [r["epoch"] for r in rows]
    keys = [("train/box_loss", "Train box loss"), ("train/obj_loss", "Train objectness loss"), ("train/cls_loss", "Train classification loss"),
            ("metrics/precision", "Validation precision"), ("metrics/recall", "Validation recall"),
            ("val/box_loss", "Validation box loss"), ("val/obj_loss", "Validation objectness loss"), ("val/cls_loss", "Validation classification loss"),
            ("metrics/mAP_0.5", "Validation mAP@0.5"), ("metrics/mAP_0.5:0.95", "Validation mAP@0.5:0.95")]
    fig, axs = plt.subplots(2, 5, figsize=(12, 4.6), dpi=300)
    for ax, (k, t) in zip(axs.flat, keys):
        y = [r[k] for r in rows]
        ax.plot(ep, y, lw=1.0, color="#1f77b4")
        ax.set_title(t, fontsize=8); ax.set_xlabel("Epoch", fontsize=7); ax.tick_params(labelsize=6); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "fig3_curves.png"); plt.close()
    best = max(rows, key=lambda r: 0.1 * r["metrics/mAP_0.5"] + 0.9 * r["metrics/mAP_0.5:0.95"])
    vo = [r["val/obj_loss"] for r in rows]
    return dict(first=rows[0], last=rows[-1], best=best, best_epoch=int(best["epoch"]) + 1,
                val_obj_min_epoch=int(np.argmin(vo)) + 1, val_obj_min=min(vo),
                peak_map50=max(r["metrics/mAP_0.5"] for r in rows), peak_map=max(r["metrics/mAP_0.5:0.95"] for r in rows))

# ---------------------------------------------------------------- Fig 4-6: YOLOv5's own test-set plots, with the F1 optimum captured
def fig_eval_plots():
    """Re-run YOLOv5 val.run on the test split (CPU) with plots, capturing the F1-confidence optimum exactly as the
    F1_curve.png legend reports it (smoothed mean over classes)."""
    cwd = os.getcwd(); sys.path.insert(0, str(Y5)); os.chdir(Y5)
    import utils.metrics as um, val as v5
    captured = {}
    orig = um.plot_mc_curve
    def patched(px, py, save_dir="mc_curve.png", names=(), xlabel="Confidence", ylabel="Metric"):
        if ylabel == "F1":
            y = um.smooth(py.mean(0), 0.05); i = int(y.argmax())
            captured.update(f1_max=float(y[i]), f1_conf=float(px[i]), n_classes_f1=int(len(py)))
        return orig(px, py, save_dir, names, xlabel, ylabel)
    um.plot_mc_curve = patched
    out = EXP / "runs" / RUN.name / "eval_test_plots"
    if out.exists(): shutil.rmtree(out)
    v5.run(data=str(DS / "data.yaml"), weights=str(RUN / "weights" / "best.pt"), batch_size=16, imgsz=640, task="test",
           device="cpu", workers=0, plots=True, verbose=False, project=str(out.parent), name=out.name, exist_ok=True)
    os.chdir(cwd)
    for a, b in (("confusion_matrix.png", "fig4_confusion.png"), ("PR_curve.png", "fig5_pr.png"), ("F1_curve.png", "fig6_f1.png")):
        if (out / a).exists(): shutil.copy2(out / a, FIG / b)
    if (RUN / "labels.jpg").exists(): shutil.copy2(RUN / "labels.jpg", FIG / "fig7_labels.jpg")
    return captured

# ---------------------------------------------------------------- Fig 8: predictions + F1 operating point
def predictions_and_f1(conf_grid=0.25):
    pred_dir = EXP / "runs" / RUN.name / "pred_test"
    if pred_dir.exists(): shutil.rmtree(pred_dir)
    subprocess.run([PY, "detect.py", "--weights", str(RUN / "weights" / "best.pt"), "--source", str(DS / "images" / "test"),
                    "--img", "640", "--conf-thres", "0.001", "--iou-thres", "0.6", "--save-txt", "--save-conf", "--nosave", "--device", "cpu",
                    "--project", str(EXP / "runs" / RUN.name), "--name", "pred_test", "--exist-ok"], cwd=str(Y5), env=ENV,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # F1 sweep (per class, averaged over classes with ground truth), greedy IoU>=0.5 matching
    def iou(a, b):
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)
    per_class = {c: {"tp_conf": [], "fp_conf": [], "n_gt": 0} for c in range(len(NAMES))}
    test_imgs = sorted((DS / "images" / "test").glob("*.*"))
    for p in test_imgs:
        W, H = Image.open(p).size
        gb, gl, _ = yolo_to_xyxy(open(DS / "labels" / "test" / (p.stem + ".txt")).read().splitlines(), W, H)
        pf = pred_dir / "labels" / (p.stem + ".txt")
        pb, pl, pc = yolo_to_xyxy(open(pf).read().splitlines(), W, H) if pf.exists() else ([], [], [])
        for c in set(gl) | set(pl):
            g = [b for b, l in zip(gb, gl) if l == c]; per_class[c]["n_gt"] += len(g)
            preds = sorted([(s, b) for b, l, s in zip(pb, pl, pc) if l == c], key=lambda t: -t[0])
            used = [False] * len(g)
            for s, b in preds:
                best, bi = 0.5, -1
                for j, gg in enumerate(g):
                    if not used[j]:
                        v = iou(b, gg)
                        if v >= best: best, bi = v, j
                if bi >= 0: used[bi] = True; per_class[c]["tp_conf"].append(s)
                else: per_class[c]["fp_conf"].append(s)
    ths = np.linspace(0, 1, 1001); f1s = []
    for c, d in per_class.items():
        if d["n_gt"] == 0: continue
        tp = np.array(d["tp_conf"]); fp = np.array(d["fp_conf"]); f1c = []
        for t in ths:
            ntp = (tp >= t).sum(); nfp = (fp >= t).sum()
            P = ntp / (ntp + nfp) if ntp + nfp else 0.0; R = ntp / d["n_gt"]
            f1c.append(2 * P * R / (P + R) if P + R else 0.0)
        f1s.append(f1c)
    f1 = np.mean(np.array(f1s), 0); k = int(np.argmax(f1))
    extra = dict(f1_max_greedy=float(f1[k]), f1_conf_greedy=float(ths[k]))   # cross-check only; the paper uses YOLOv5's values
    # qualitative grid at the deployment threshold
    picks = []
    for pref, n in (("gatewayofindia", 2), ("charminar", 3), ("mysore_palace", 2), ("Tagmahal", 2)):
        files = [p for p in test_imgs if p.name.startswith(pref)]
        random.shuffle(files); picks += files[:n]
    ims = []
    for p in picks:
        im = Image.open(p).convert("RGB"); W, H = im.size
        pf = pred_dir / "labels" / (p.stem + ".txt")
        pb, pl, pc = yolo_to_xyxy(open(pf).read().splitlines(), W, H) if pf.exists() else ([], [], [])
        keep = [i for i, s in enumerate(pc) if s >= conf_grid]
        ims.append(draw_boxes(im, [pb[i] for i in keep], [pl[i] for i in keep], [pc[i] for i in keep]))
    grid(ims, 3, cell=640, out=FIG / "fig8_predictions.png")
    return extra

if __name__ == "__main__":
    fig_samples()
    curves = fig_curves()
    extra = fig_eval_plots()
    extra.update(predictions_and_f1(conf_grid=round(extra.get("f1_conf", 0.25), 2)))
    extra.update({k: v for k, v in curves.items() if k not in ("first", "last", "best")})
    extra["first"] = curves["first"]; extra["last"] = curves["last"]; extra["best"] = curves["best"]
    json.dump(extra, open(FIG / "metrics_extra.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in extra.items() if k not in ("first", "last", "best")}, indent=1))
    print("figures:", sorted(os.listdir(FIG)))
