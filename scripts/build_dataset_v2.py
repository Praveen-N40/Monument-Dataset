# -*- coding: utf-8 -*-
"""Build dataset_v2 from the annotated pool: drop exact duplicates (MD5), group near-duplicates (64-bit dHash,
Hamming <= 4) so they fall in one subset, and split 70/15/15 stratified by monument with seed 0.
Usage: python build_dataset_v2.py <pool_dir> <out_dir>
  pool_dir has <monument>/images/*.jpg and <monument>/annotation/*.txt (YOLO format)."""
import os, random, shutil, collections, hashlib, json, sys
from PIL import Image
POOL, OUT = sys.argv[1], sys.argv[2]
random.seed(0)
names = ["Auto","Bag","Balloon","Bangle","Banners","Birds","Building","Camel","Camera","Charminar","CottonCandy","Cow","Cycle",
         "Eatable","Flowers","Fruits","Garments","GatewayOfIndia","Goat","GreenPlants","Idol","Light","MetalDetector","Mobile","Moon",
         "MysorePalace","People","PoliceBoard","Scooters","Shops","Ship","SittingTable","Stick","TajMahal","Vessels","Water"]

def dhash(p, s=8):
    im = Image.open(p).convert("L").resize((s + 1, s)); px = list(im.getdata()); bits = []
    for r in range(s):
        for c in range(s): bits.append(px[r * (s + 1) + c] > px[r * (s + 1) + c + 1])
    return sum(b << i for i, b in enumerate(bits))

pairs = []
for mon in sorted(os.listdir(POOL)):
    idir, ldir = f"{POOL}/{mon}/images", f"{POOL}/{mon}/annotation"
    if not os.path.isdir(idir): continue
    for f in sorted(os.listdir(idir)):
        lp = f"{ldir}/{os.path.splitext(f)[0]}.txt"
        if os.path.exists(lp): pairs.append((mon, f"{idir}/{f}", lp))
seen, kept, dropped = set(), [], 0
for p in pairs:
    h = hashlib.md5(open(p[1], "rb").read()).hexdigest()
    if h in seen: dropped += 1; continue
    seen.add(h); kept.append(p)
hs = [(dhash(p[1]), p) for p in kept]; dups = []
for i in range(len(hs)):
    for j in range(i + 1, len(hs)):
        if bin(hs[i][0] ^ hs[j][0]).count("1") <= 4: dups.append((hs[i][1][1], hs[j][1][1]))
parent = {p[1]: p[1] for p in kept}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b in dups: parent[find(a)] = find(b)
groups = collections.defaultdict(list)
for mon, ip, lp in kept: groups[find(ip)].append((mon, ip, lp))
bymon = collections.defaultdict(list)
for g in groups.values(): bymon[g[0][0]].append(g)
split = {"train": [], "val": [], "test": []}
for mon, gs in sorted(bymon.items()):
    random.shuffle(gs); n = len(gs); nv = round(0.15 * n); nt = round(0.15 * n)
    for k, g in enumerate(gs): split["val" if k < nv else "test" if k < nv + nt else "train"].extend(g)
for s in split:
    for sub in ("images", "labels"): os.makedirs(f"{OUT}/{sub}/{s}", exist_ok=True)
    for mon, ip, lp in split[s]:
        shutil.copy2(ip, f"{OUT}/images/{s}/{os.path.basename(ip)}"); shutil.copy2(lp, f"{OUT}/labels/{s}/{os.path.basename(lp)}")
cnt = {s: collections.Counter() for s in split}
for s in split:
    for mon, ip, lp in split[s]:
        for line in open(lp):
            q = line.split()
            if q: cnt[s][int(q[0])] += 1
rows = [(i, nm, cnt["train"][i], cnt["val"][i], cnt["test"][i]) for i, nm in enumerate(names)]
sizes = [Image.open(p[1]).size for p in kept]
json.dump({"images": {s: len(split[s]) for s in split}, "monuments": {m: [sum(1 for p in split[s] if p[0] == m) for s in ("train", "val", "test")] for m in bymon},
           "classes": rows, "instances": {s: sum(cnt[s].values()) for s in split}, "near_dup_pairs": len(dups), "exact_dropped": dropped,
           "size_min": [min(s[0] for s in sizes), min(s[1] for s in sizes)], "size_max": [max(s[0] for s in sizes), max(s[1] for s in sizes)]},
          open(f"{OUT}/stats.json", "w"), indent=1)
open(f"{OUT}/data.yaml", "w").write(f"train: {OUT}/images/train\nval: {OUT}/images/val\ntest: {OUT}/images/test\nnc: 36\nnames: {json.dumps(names)}\n")
print("pool", len(pairs), "exact duplicates dropped", dropped, "kept", len(kept), "near-dup pairs", len(dups), {s: len(split[s]) for s in split})
