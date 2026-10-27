# Monument-Dataset

Annotations, split lists, image sources, configurations, trained weights, and scripts for the paper
*YOLOv5-Based Joint Detection of Indian Monuments and Contextual Objects in Unconstrained Images*.

## Images and licence
This repository does **not** redistribute the images: they were collected from publicly accessible web sources and
copyright in each image remains with its original owner. The images are available from the corresponding author on
request for non-commercial research use; `splits/*.txt` and `labels/` let the exact training, validation and test
subsets be reconstructed. The annotation files, split lists, class mapping, result files, and scripts in this
repository are released under CC BY 4.0 (see LICENSE).

## Layout
```
labels/            one YOLO-format label file per image (class x y w h, normalised)
splits/            train.txt, val.txt, test.txt  (image file names; 364 / 76 / 76)
classes.txt        class index -> name (36 classes)
configs/           data.yaml, hyp_full.yaml, hyp_noaug.yaml, hyp_basicaug.yaml, dataset_stats.json
results/           per-run test/validation metrics (JSON) used for Tables 3-5
weights/           best.pt of every YOLOv5 / YOLOv8 run (main model: yolov5s_full_imgw_best.pt)
release_assets/    Faster R-CNN and SSD weights (>100 MB) - attached to the GitHub Release
scripts/           dataset build, training queue, torchvision baselines, figures, latency benchmark
```

## Rebuilding the dataset
Place the images obtained from the authors in `images/{train,val,test}/` according to `splits/*.txt`, and copy the
matching files from `labels/` into `labels/{train,val,test}/`. `scripts/build_dataset_v2.py` documents how the
de-duplicated, monument-stratified split was produced from the raw annotated pool.

## Training the main model (YOLOv5 v7.0, https://github.com/ultralytics/yolov5)
```
python train.py --img 640 --batch-size 16 --epochs 100 --data configs/data.yaml --weights yolov5s.pt \
       --hyp configs/hyp_full.yaml --image-weights --seed 0 --workers 2 --cache ram --name yolov5s_full_imgw
```
Ablations: drop `--image-weights`; use `configs/hyp_basicaug.yaml` (no mosaic) or `configs/hyp_noaug.yaml`.
Baselines: `yolov5n.pt`, `yolov5m.pt` (`--batch-size 8`), YOLOv8s via `scripts/run_queue.py`,
Faster R-CNN and SSD via `scripts/run_torchvision_baselines.py fasterrcnn|ssd`.

## Evaluation
```
python val.py --weights weights/yolov5s_full_imgw_best.pt --data configs/data.yaml --img 640 --task test --verbose
python scripts/bench_latency.py        # batch-1 latency (forward + NMS), GPU and CPU
python scripts/make_figures.py yolov5s_full_imgw
```

## Hardware used in the paper
NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), Intel Core i5-12500H, 16 GB RAM, Windows 11, PyTorch 2.6.0 + CUDA 12.4.
