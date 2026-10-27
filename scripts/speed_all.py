# -*- coding: utf-8 -*-
"""Re-measure batch-1 inference speed of every trained model back-to-back in one session (GPU warmed up,
same power state), and fill missing params/GFLOPs for the YOLOv8 run.  Updates results/*.json in place.
Run only when no other job is using the GPU."""
import json, re, subprocess, sys, os, time
from pathlib import Path
import torch

EXP = Path("D:/Code/paper_experiments"); Y5 = Path("D:/Code/yolov5"); DATA = EXP / "dataset_v2" / "data.yaml"
RES = EXP / "results"; LOGS = EXP / "logs"; PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
SPEED_RE = re.compile(r"Speed: ([\d.]+)ms pre-process, ([\d.]+)ms inference, ([\d.]+)ms NMS per image")

def y5_speed(name, dev):
    w = EXP / "runs" / name / "weights" / "best.pt"; lf = LOGS / f"{name}_speed2_{dev}.log"
    with open(lf, "w", encoding="utf-8") as f:
        subprocess.run([PY, "val.py", "--weights", str(w), "--data", str(DATA), "--img", "640", "--batch-size", "1", "--task", "test",
                        "--device", dev, "--project", str(EXP / "runs" / name), "--name", f"speed2_{dev}", "--exist-ok"],
                       cwd=str(Y5), env=ENV, stdout=f, stderr=subprocess.STDOUT)
    m = SPEED_RE.findall(open(lf, encoding="utf-8", errors="ignore").read())
    pre, inf, nms = map(float, m[-1]); tot = pre + inf + nms
    return dict(pre=pre, inf=inf, nms=nms, total=tot, fps=1000.0 / tot)

def y8_speed(name, dev):
    from ultralytics import YOLO
    m = YOLO(str(EXP / "runs" / name / "weights" / "best.pt"))
    r = m.val(data=str(DATA), split="test", imgsz=640, batch=1, device=dev, plots=False, project=str(EXP / "runs" / name), name=f"speed2_{dev}", exist_ok=True, verbose=False)
    s = r.speed; tot = s["preprocess"] + s["inference"] + s["postprocess"]
    return dict(pre=s["preprocess"], inf=s["inference"], nms=s["postprocess"], total=tot, fps=1000.0 / tot)

def y8_summary(name):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    m = YOLO(str(EXP / "runs" / name / "weights" / "best.pt"))
    return dict(params=int(get_num_params(m.model)), gflops=float(get_flops(m.model, 640)))

def tv_speed(kind, dev):
    sys.path.insert(0, str(EXP)); import run_torchvision_baselines as tb
    m = tb.build(kind); m.load_state_dict(torch.load(EXP / "runs" / f"{kind}_full" / "best.pt", map_location="cpu"))
    return tb.speed(m, tb.YoloDS("test"), "cuda" if dev == "0" else "cpu", n=50 if dev == "0" else 20)

if __name__ == "__main__":
    order = [n[:-5] for n in sorted(os.listdir(RES)) if n.endswith(".json") and not n.startswith("smoke")]
    # warm-up
    if (EXP / "runs" / "yolov5s_full" / "weights" / "best.pt").exists(): y5_speed("yolov5s_full", "0")
    for name in order:
        jf = RES / f"{name}.json"; r = json.load(open(jf))
        try:
            for dev, key in (("0", "speed_gpu"), ("cpu", "speed_cpu")):
                if name.startswith("yolov8"): r[key] = y8_speed(name, dev)
                elif name.startswith("yolov5"): r[key] = y5_speed(name, dev)
                else: r[key] = tv_speed(name.split("_")[0], dev)
            if name.startswith("yolov8") and not (r.get("summary") or {}).get("params"): r["summary"] = y8_summary(name)
            json.dump(r, open(jf, "w"), indent=1)
            print(name, "gpu %.1f ms (%.0f fps)  cpu %.1f ms (%.0f fps)" % (r["speed_gpu"]["total"], r["speed_gpu"]["fps"], r["speed_cpu"]["total"], r["speed_cpu"]["fps"]), flush=True)
        except Exception as e:
            print(name, "FAILED", repr(e), flush=True)
