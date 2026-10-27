# -*- coding: utf-8 -*-
"""Batch-1 latency benchmark for every trained model: warmed-up forward pass + NMS on a preloaded
letterboxed 640x640 test image, median of N runs, GPU then CPU.  Writes speed_gpu/speed_cpu into
results/*.json (the earlier val.py pipeline timings are kept under speed_val_*)."""
import sys, json, time, os
from pathlib import Path
import numpy as np, torch, cv2
EXP = Path("D:/Code/paper_experiments"); Y5 = Path("D:/Code/yolov5"); RES = EXP / "results"
sys.path.insert(0, str(Y5)); sys.path.insert(0, str(EXP))
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

img = cv2.imread(str(sorted((EXP / "dataset_v2" / "images" / "test").glob("charminar*"))[0]))
lb = letterbox(img, 640, auto=False)[0]
t_y5 = torch.from_numpy(lb[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0)   # 1x3x640x640
t_tv = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255)                # 3xHxW

def bench(fn, dev, n):
    with torch.no_grad():
        for _ in range(10): fn()
        if dev == "cuda": torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn()
            if dev == "cuda": torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ms = 1000 * float(np.median(ts)); return dict(total=ms, fps=1000.0 / ms, method="forward+NMS, batch 1, median of %d after warm-up" % n)

def make_fn(name, dev):
    d = torch.device(dev)
    if name.startswith("yolov5"):
        m = DetectMultiBackend(str(EXP / "runs" / name / "weights" / "best.pt"), device=d, fp16=False); m.eval()
        x = t_y5.to(d); m(x)
        return lambda: non_max_suppression(m(x), 0.25, 0.45)
    if name.startswith("yolov8"):
        from ultralytics import YOLO
        m = YOLO(str(EXP / "runs" / name / "weights" / "best.pt")); x = t_y5.to(d)
        return lambda: m.predict(x, verbose=False, device=dev, conf=0.25, iou=0.45)
    import run_torchvision_baselines as tb
    kind = name.split("_")[0]; m = tb.build(kind)
    m.load_state_dict(torch.load(EXP / "runs" / name / "best.pt", map_location="cpu")); m.eval().to(d); x = t_tv.to(d)
    return lambda: m([x])

if __name__ == "__main__":
    names = [n[:-5] for n in sorted(os.listdir(RES)) if n.endswith(".json")]
    for dev, key, n_y, n_tv in (("cuda", "speed_gpu", 100, 50), ("cpu", "speed_cpu", 30, 10)):
        for name in names:
            jf = RES / f"{name}.json"; r = json.load(open(jf))
            try:
                fn = make_fn(name, dev)
                res = bench(fn, dev, n_tv if name.split("_")[0] in ("fasterrcnn", "ssd") else n_y)
                if key in r and r[key].get("method") is None: r["speed_val_" + key[6:]] = r[key]
                r[key] = res; json.dump(r, open(jf, "w"), indent=1)
                print("%-20s %-4s %6.1f ms  %5.1f fps" % (name, dev, res["total"], res["fps"]), flush=True)
            except Exception as e:
                print(name, dev, "FAILED", repr(e), flush=True)
            if dev == "cuda": torch.cuda.empty_cache()
