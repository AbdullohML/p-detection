"""Batch-one wall-clock prediction, including preprocessing and NMS, excluding disk reads."""
import argparse
import json
import time
import numpy as np
from PIL import Image
from common import ROOT, CONDITIONS, checkpoint, hardware, torch, update_results
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=CONDITIONS+['both'], default='both')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    assert args.warmup > 0 and args.repeats > 0
    torch.set_num_threads(6)
    names = (ROOT / 'data/splits/test.txt').read_text().splitlines()
    images = [Image.open(ROOT / 'data/processed/images/test' / n).convert('RGB') for n in names]
    device = 0 if torch.cuda.is_available() else 'cpu'
    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    for condition in CONDITIONS if args.condition == 'both' else [args.condition]:
        model = YOLO(str(checkpoint(condition)))
        def predict(image):
            return model.predict(image, imgsz=640, rect=False, batch=1, device=device,
                                 conf=0.25, iou=0.7, half=False, verbose=False)
        for i in range(args.warmup):
            predict(images[i % len(images)])
        times = []
        for _ in range(args.repeats):
            for image in images:
                synchronize()
                start = time.perf_counter()
                predict(image)
                synchronize()
                times.append((time.perf_counter()-start)*1000)
        values = dict(latency_ms=float(np.mean(times)), fps=float(1000/np.mean(times)))
        record = dict(metrics=values, hardware=hardware(), warmup=args.warmup,
                      repeats=args.repeats, images=names, timings_ms=times,
                      methodology='PIL images preloaded; batch 1; FP32; fixed 640x640 padding (rect=False); preprocessing + forward + NMS; synchronized wall time')
        (ROOT / f'results/metrics/{condition}_benchmark.json').write_text(json.dumps(record, indent=2)+'\n')
        update_results(condition, values)
        print(condition, values)


if __name__ == '__main__':
    main()
