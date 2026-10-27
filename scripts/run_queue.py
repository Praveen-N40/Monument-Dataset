# -*- coding: utf-8 -*-
"""Sequential experiment queue for the IEEE Access paper.
Trains each configuration on dataset_v2 (if not already trained), then evaluates
best.pt on the TEST split (per-class P/R/AP50/AP50-95), and measures batch-1 speed.
Resumable: runs whose best.pt exists are not retrained; evaluations whose json exists are skipped.
"""
import os, re, sys, json, subprocess, time, shutil
from pathlib import Path

EXP = Path("D:/Code/paper_experiments")
Y5 = Path("D:/Code/yolov5")
DATA = EXP / "dataset_v2" / "data.yaml"
RUNS = EXP / "runs"
LOGS = EXP / "logs"
RES = EXP / "results"
for d in (RUNS, LOGS, RES):
    d.mkdir(exist_ok=True, parents=True)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

QUEUE = [
    # name,            framework, weights,                       extra train args
    ("yolov5s_full",     "y5", EXP/"weights"/"yolov5s.pt", ["--hyp", str(EXP/"hyps"/"hyp_full.yaml")]),
    ("yolov5s_noaug",    "y5", EXP/"weights"/"yolov5s.pt", ["--hyp", str(EXP/"hyps"/"hyp_noaug.yaml")]),
    ("yolov5s_basicaug", "y5", EXP/"weights"/"yolov5s.pt", ["--hyp", str(EXP/"hyps"/"hyp_basicaug.yaml")]),
    ("yolov5s_full_imgw","y5", EXP/"weights"/"yolov5s.pt", ["--hyp", str(EXP/"hyps"/"hyp_full.yaml"), "--image-weights"]),
    ("yolov5n_full",     "y5", EXP/"weights"/"yolov5n.pt", ["--hyp", str(EXP/"hyps"/"hyp_full.yaml")]),
    ("yolov5m_full",     "y5", EXP/"weights"/"yolov5m.pt", ["--hyp", str(EXP/"hyps"/"hyp_full.yaml"), "--batch-size", "8"]),
    ("yolov8s_full",     "y8", EXP/"weights"/"yolov8s.pt", []),
    # larger/smaller YOLOv5 with the full recipe including class-aware sampling (fair comparison with the main model)
    ("yolov5n_full_imgw","y5", EXP/"weights"/"yolov5n.pt", ["--hyp", str(EXP/"hyps"/"hyp_full.yaml"), "--image-weights"]),
    ("yolov5m_full_imgw","y5", EXP/"weights"/"yolov5m.pt", ["--hyp", str(EXP/"hyps"/"hyp_full.yaml"), "--image-weights", "--batch-size", "8"]),
]

def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOGS / "queue.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run(cmd, logfile, cwd):
    with open(logfile, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        p = subprocess.run(list(map(str, cmd)), cwd=str(cwd), env=ENV, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode

CLS_RE = re.compile(r"^\s*(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
SPEED_RE = re.compile(r"Speed: ([\d.]+)ms pre-process, ([\d.]+)ms inference, ([\d.]+)ms NMS per image")
SUMM_RE = re.compile(r"Model summary: (\d+) layers, (\d+) parameters, \d+ gradients, ([\d.]+) GFLOPs")

def parse_val_log(text):
    rows = {}
    for line in text.splitlines():
        m = CLS_RE.match(line)
        if m:
            rows[m.group(1)] = dict(images=int(m.group(2)), instances=int(m.group(3)), P=float(m.group(4)),
                                    R=float(m.group(5)), AP50=float(m.group(6)), AP5095=float(m.group(7)))
    sp = SPEED_RE.findall(text)
    summ = SUMM_RE.findall(text)
    return rows, (sp[-1] if sp else None), (summ[-1] if summ else None)

def train_y5(name, weights, extra):
    out = RUNS / name
    if (out / "weights" / "best.pt").exists() and (out / "results.csv").exists() and \
            sum(1 for _ in open(out / "results.csv")) >= 101:   # header + 100 epochs = finished run
        log(f"[skip train] {name}")
        return
    if out.exists():
        shutil.rmtree(out)
    log(f"[train] {name}")
    cmd = [PY, "train.py", "--img", "640", "--batch-size", "16", "--epochs", "100", "--data", DATA,
           "--weights", weights, "--project", RUNS, "--name", name, "--exist-ok", "--seed", "0", "--workers", "2",
           "--cache", "ram"] + extra
    # --batch-size given in extra overrides the earlier one (argparse keeps last)
    rc = run(cmd, LOGS / f"{name}_train.log", Y5)
    log(f"[train done rc={rc}] {name}")
    if rc != 0:  # do not evaluate a partial checkpoint
        shutil.rmtree(out, ignore_errors=True)
        raise RuntimeError(f"training failed rc={rc}")

def eval_y5(name):
    jf = RES / f"{name}.json"
    if jf.exists():
        log(f"[skip eval] {name}")
        return
    w = RUNS / name / "weights" / "best.pt"
    if not w.exists():
        log(f"[no weights] {name}")
        return
    res = {"name": name}
    for split in ("val", "test"):
        lf = LOGS / f"{name}_eval_{split}.log"
        if lf.exists(): lf.unlink()
        rc = run([PY, "val.py", "--weights", w, "--data", DATA, "--img", "640", "--batch-size", "16", "--task", split,
                  "--verbose", "--project", RUNS / name, "--name", f"eval_{split}", "--exist-ok"], lf, Y5)
        rows, speed, summ = parse_val_log(open(lf, encoding="utf-8", errors="ignore").read())
        res[split] = rows
        if summ: res["summary"] = dict(layers=int(summ[0]), params=int(summ[1]), gflops=float(summ[2]))
    for dev in ("0", "cpu"):
        lf = LOGS / f"{name}_speed_{dev}.log"
        if lf.exists(): lf.unlink()
        run([PY, "val.py", "--weights", w, "--data", DATA, "--img", "640", "--batch-size", "1", "--task", "test",
             "--device", dev, "--project", RUNS / name, "--name", f"speed_{dev}", "--exist-ok"], lf, Y5)
        _, speed, _ = parse_val_log(open(lf, encoding="utf-8", errors="ignore").read())
        if speed:
            pre, inf, nms = map(float, speed)
            res[f"speed_{'gpu' if dev == '0' else 'cpu'}"] = dict(pre=pre, inf=inf, nms=nms, total=pre + inf + nms, fps=1000.0 / (pre + inf + nms))
    json.dump(res, open(jf, "w"), indent=1)
    log(f"[eval done] {name}")

def train_eval_y8(name, weights):
    out = RUNS / name
    jf = RES / f"{name}.json"
    from ultralytics import YOLO
    if not (out / "weights" / "best.pt").exists():
        if out.exists(): shutil.rmtree(out)
        log(f"[train] {name}")
        m = YOLO(str(weights))
        m.train(data=str(DATA), imgsz=640, epochs=100, batch=16, seed=0, workers=2, cache="ram", project=str(RUNS), name=name,
                exist_ok=True, plots=True, verbose=False)
        log(f"[train done] {name}")
    if jf.exists():
        log(f"[skip eval] {name}"); return
    names = json.load(open(EXP / "dataset_v2" / "stats.json"))["classes"]
    cls_names = [r[1] for r in names]
    res = {"name": name}
    m = YOLO(str(out / "weights" / "best.pt"))
    for split in ("val", "test"):
        r = m.val(data=str(DATA), split=split, imgsz=640, batch=16, plots=True, project=str(out), name=f"eval_{split}", exist_ok=True, verbose=False)
        rows = {"all": dict(P=float(r.box.mp), R=float(r.box.mr), AP50=float(r.box.map50), AP5095=float(r.box.map))}
        for k, ci in enumerate(r.box.ap_class_index):
            rows[cls_names[int(ci)]] = dict(P=float(r.box.p[k]), R=float(r.box.r[k]), AP50=float(r.box.ap50[k]), AP5095=float(r.box.ap[k]))
        res[split] = rows
    info = m.info(verbose=False)  # (layers, params, gradients, gflops)
    if info:
        res["summary"] = dict(layers=int(info[0]), params=int(info[1]), gflops=float(info[3]))
    for dev in ("0", "cpu"):
        r = m.val(data=str(DATA), split="test", imgsz=640, batch=1, device=dev, plots=False, project=str(out), name=f"speed_{dev}", exist_ok=True, verbose=False)
        s = r.speed
        tot = s["preprocess"] + s["inference"] + s["postprocess"]
        res[f"speed_{'gpu' if dev == '0' else 'cpu'}"] = dict(pre=s["preprocess"], inf=s["inference"], nms=s["postprocess"], total=tot, fps=1000.0 / tot)
    json.dump(res, open(jf, "w"), indent=1)
    log(f"[eval done] {name}")

if __name__ == "__main__":
    only = sys.argv[1:]  # optional subset of names
    log("==== queue start ====")
    for name, fw, weights, extra in QUEUE:
        if only and name not in only:
            continue
        try:
            if fw == "y5":
                train_y5(name, weights, extra)
                eval_y5(name)
            else:
                train_eval_y8(name, weights)
        except Exception as e:
            log(f"[ERROR] {name}: {e!r}")
    log("==== queue finished ====")
