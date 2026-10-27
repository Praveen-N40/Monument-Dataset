# -*- coding: utf-8 -*-
"""Faster R-CNN (ResNet-50-FPN) and SSD300 (VGG-16) baselines on dataset_v2 with torchvision.
Same splits, 100 epochs, COCO-pretrained backbones, evaluated with pycocotools (mAP@0.5, mAP@0.5:0.95,
per-class AP@0.5) on the TEST split; batch-1 speed on GPU and CPU.  Resumable per model.
Usage: python run_torchvision_baselines.py [fasterrcnn|ssd] [--epochs N]
"""
import os, sys, json, time, math, random, argparse
from pathlib import Path
import numpy as np, torch, torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils as det_utils
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

EXP = Path("D:/Code/paper_experiments")
DS = EXP / "dataset_v2"
RUNS = EXP / "runs"; LOGS = EXP / "logs"; RES = EXP / "results"
for d in (RUNS, LOGS, RES): d.mkdir(exist_ok=True)
NAMES = [r[1] for r in json.load(open(DS / "stats.json"))["classes"]]
NC = len(NAMES)
torch.manual_seed(0); random.seed(0); np.random.seed(0)

class YoloDS(torch.utils.data.Dataset):
    def __init__(self, split, train=False, size=None):
        self.imgs = sorted((DS / "images" / split).glob("*.*"))
        self.split, self.train, self.size = split, train, size
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        p = self.imgs[i]
        im = Image.open(p).convert("RGB"); W, H = im.size
        boxes, labels = [], []
        lp = DS / "labels" / self.split / (p.stem + ".txt")
        if lp.exists():
            for line in open(lp):
                q = line.split()
                if len(q) < 5: continue
                c, x, y, w, h = int(q[0]), *map(float, q[1:5])
                x1, y1, x2, y2 = (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H
                if x2 - x1 < 1 or y2 - y1 < 1: continue
                boxes.append([x1, y1, x2, y2]); labels.append(c + 1)  # 0 = background
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        if self.train and random.random() < 0.5:  # horizontal flip
            im = TF.hflip(im)
            if len(boxes): boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        if self.train:  # photometric jitter comparable to YOLOv5 HSV augmentation
            im = TF.adjust_brightness(im, random.uniform(0.6, 1.4))
            im = TF.adjust_saturation(im, random.uniform(0.3, 1.7))
        t = TF.to_tensor(im)
        return t, dict(boxes=boxes, labels=labels, image_id=torch.tensor(i), orig_size=torch.tensor([H, W]))

def collate(b): return tuple(zip(*b))

def build(kind):
    if kind == "fasterrcnn":
        m = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=640, max_size=640)
        inf = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(inf, NC + 1)
    else:
        m = ssd300_vgg16(weights="DEFAULT")
        in_ch = det_utils.retrieve_out_channels(m.backbone, (300, 300))
        na = m.anchor_generator.num_anchors_per_location()
        m.head.classification_head = SSDClassificationHead(in_ch, na, NC + 1)
    return m

def coco_gt(ds):
    images, anns = [], []
    aid = 1
    for i, p in enumerate(ds.imgs):
        _, t = ds[i]
        H, W = t["orig_size"].tolist()
        images.append(dict(id=i, width=W, height=H, file_name=p.name))
        for b, l in zip(t["boxes"].tolist(), t["labels"].tolist()):
            x1, y1, x2, y2 = b
            anns.append(dict(id=aid, image_id=i, category_id=int(l), bbox=[x1, y1, x2 - x1, y2 - y1],
                             area=(x2 - x1) * (y2 - y1), iscrowd=0)); aid += 1
    cats = [dict(id=i + 1, name=n) for i, n in enumerate(NAMES)]
    gt = COCO(); gt.dataset = dict(images=images, annotations=anns, categories=cats); gt.createIndex()
    return gt

@torch.no_grad()
def evaluate(m, ds, device, bs=8):
    m.eval(); dets = []
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)
    for ims, ts in dl:
        out = m([i.to(device) for i in ims])
        for o, t in zip(out, ts):
            iid = int(t["image_id"])
            for b, s, l in zip(o["boxes"].cpu().tolist(), o["scores"].cpu().tolist(), o["labels"].cpu().tolist()):
                x1, y1, x2, y2 = b
                dets.append(dict(image_id=iid, category_id=int(l), bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s)))
    gt = coco_gt(ds)
    res = {"all": dict(AP50=0.0, AP5095=0.0)}
    if not dets: return res
    dt = gt.loadRes(dets)
    E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    res["all"] = dict(AP5095=float(E.stats[0]), AP50=float(E.stats[1]))
    # per-class AP@0.5 and AP@0.5:0.95 (classes with ground truth only)
    prec = E.eval["precision"]  # [T, R, K, A, M]
    t50 = int(np.where(np.isclose(E.params.iouThrs, 0.5))[0][0])
    present = {a["category_id"] for a in gt.dataset["annotations"]}
    for k, cid in enumerate(E.params.catIds):
        if cid not in present: continue
        p50 = prec[t50, :, k, 0, -1]; pall = prec[:, :, k, 0, -1]
        ap50 = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else 0.0
        ap = float(np.mean(pall[pall > -1])) if (pall > -1).any() else 0.0
        res[NAMES[cid - 1]] = dict(AP50=ap50, AP5095=ap)
    return res

@torch.no_grad()
def speed(m, ds, device, n=50):
    m.eval(); m.to(device)
    ts = []
    for i in range(min(n, len(ds))):
        im, _ = ds[i]
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); m([im.to(device)])
        if device == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ms = 1000 * float(np.mean(ts[5:] if len(ts) > 8 else ts[1:]))
    return dict(total=ms, fps=1000.0 / ms)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=["fasterrcnn", "ssd"]); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, results under runs/smoke_*")
    a = ap.parse_args()
    name = f"{a.kind}_full" if not a.smoke else f"smoke_{a.kind}"; out = RUNS / name; out.mkdir(exist_ok=True, parents=True)
    jf = RES / f"{name}.json"; wbest = out / "best.pt"
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(LOGS / f"{name}_train.log", "a", encoding="utf-8")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    m = build(a.kind).to(device)
    nparams = sum(p.numel() for p in m.parameters())
    train_ds, val_ds, test_ds = YoloDS("train", train=True), YoloDS("val"), YoloDS("test")
    if a.smoke:
        a.epochs = 1; train_ds.imgs = train_ds.imgs[:8]; val_ds.imgs = val_ds.imgs[:8]; test_ds.imgs = test_ds.imgs[:8]
        if wbest.exists(): wbest.unlink()
    if not wbest.exists():
        dl = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, shuffle=True, collate_fn=collate, num_workers=0)
        params = [p for p in m.parameters() if p.requires_grad]
        # torchvision reference recipes scaled linearly to the batch size: Faster R-CNN 0.02/16 img, SSD300 0.002/32 img
        lr = (0.02 / 16 if a.kind == "fasterrcnn" else 0.002 / 32) * a.batch
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
        iters = a.epochs * len(dl); warm = 3 * len(dl)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(it, iters) / iters)) * 0.99 + 0.01))
        best, t_start = -1.0, time.time()
        for ep in range(a.epochs):
            m.train(); tot = 0.0
            for ims, ts in dl:
                ims = [i.to(device) for i in ims]
                ts = [{k: v.to(device) for k, v in t.items() if k in ("boxes", "labels")} for t in ts]
                loss = sum(m(ims, ts).values())
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step(); sched.step()
                tot += float(loss)
            r = evaluate(m, val_ds, device)
            fit = 0.1 * r["all"]["AP50"] + 0.9 * r["all"]["AP5095"]
            log(f"epoch {ep+1}/{a.epochs} loss {tot/len(dl):.4f} val mAP50 {r['all']['AP50']:.4f} mAP50-95 {r['all']['AP5095']:.4f} fitness {fit:.4f}")
            if fit > best:
                best = fit; torch.save(m.state_dict(), wbest)
        log(f"training time {(time.time()-t_start)/3600:.2f} h")
    m.load_state_dict(torch.load(wbest, map_location=device)); m.to(device)
    res = {"name": name, "summary": dict(params=nparams)}
    try:  # FLOPs at the model's working resolution (640 for Faster R-CNN, 300 for SSD)
        from thop import profile
        m.eval(); side = 640 if a.kind == "fasterrcnn" else 300
        flops, _ = profile(m, inputs=([torch.zeros(3, side, side, device=device)],), verbose=False)
        res["summary"]["gflops"] = float(flops / 1e9); res["summary"]["flops_input"] = side
    except Exception as e:
        log(f"thop failed: {e!r}")
    res["val"] = evaluate(m, val_ds, device); res["test"] = evaluate(m, test_ds, device)
    res["speed_gpu"] = speed(m, test_ds, "cuda") if device == "cuda" else None
    res["speed_cpu"] = speed(m, test_ds, "cpu", n=(5 if a.smoke else 20))
    json.dump(res, open(jf, "w"), indent=1)
    log(f"done: test mAP50 {res['test']['all']['AP50']:.4f} mAP50-95 {res['test']['all']['AP5095']:.4f}")

if __name__ == "__main__":
    main()
