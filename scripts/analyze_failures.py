"""Confidence-ordered, one-to-one matching at IoU >= 0.5."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from common import ROOT, CONDITIONS, checkpoint, torch
from ultralytics import YOLO


def match(predictions, ground_truth, threshold=0.5):
    used, tp, fp = set(), [], []
    for index in sorted(range(len(predictions)), key=lambda i: predictions[i][4], reverse=True):
        box = np.asarray(predictions[index][:4])
        best, best_iou = None, -1
        for j, gt in enumerate(ground_truth):
            if j in used:
                continue
            gt = np.asarray(gt)
            extent = np.maximum(0, np.minimum(box[2:], gt[2:])-np.maximum(box[:2], gt[:2]))
            intersection = float(np.prod(extent))
            union = float(np.prod(box[2:]-box[:2])+np.prod(gt[2:]-gt[:2])-intersection)
            iou = intersection/union if union > 0 else 0
            if iou > best_iou:
                best, best_iou = j, iou
        if best is not None and best_iou >= threshold:
            used.add(best)
            tp.append({'prediction': index, 'ground_truth': best, 'iou': best_iou})
        else:
            fp.append(index)
    return {'tp': tp, 'fp': fp, 'fn': [j for j in range(len(ground_truth)) if j not in used]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=CONDITIONS+['both'], default='both')
    args = parser.parse_args()
    torch.set_num_threads(6)
    for condition in CONDITIONS if args.condition == 'both' else [args.condition]:
        model = YOLO(str(checkpoint(condition)))
        records = []
        folder = ROOT / f'results/figures/failures/{condition}'
        folder.mkdir(parents=True, exist_ok=True)
        for name in (ROOT / 'data/splits/test.txt').read_text().splitlines():
            image = Image.open(ROOT / 'data/processed/images/test' / name).convert('RGB')
            w, h = image.size
            gt = []
            for row in (ROOT / 'data/processed/labels/test' / (Path(name).stem+'.txt')).read_text().splitlines():
                cls, x, y, bw, bh = map(float, row.split())
                gt.append([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
            result = model.predict(image, imgsz=640, conf=0.25, iou=0.7, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)[0]
            predictions = [box.tolist()+[float(conf)] for box,conf in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu())]
            matched = match(predictions, gt)
            record = dict(image=name, ground_truth=gt, predictions=predictions, **matched)
            records.append(record)
            if matched['fp'] or matched['fn']:
                draw = ImageDraw.Draw(image)
                for j, box in enumerate(gt):
                    color = 'yellow' if j in matched['fn'] else 'lime'
                    draw.rectangle(box, outline=color, width=3)
                    draw.text((box[0], max(0,box[1]-12)), f'GT {j}'+(' FN' if j in matched['fn'] else ''), fill=color)
                for j, box in enumerate(predictions):
                    color = 'red' if j in matched['fp'] else 'cyan'
                    draw.rectangle(box[:4], outline=color, width=2)
                    draw.text((box[0], box[1]+3), f'pred {j} {box[4]:.2f}'+(' FP' if j in matched['fp'] else ''), fill=color)
                image.save(folder / name)
        summary = {key: sum(len(row[key]) for row in records) for key in ['tp','fp','fn']}
        output = {'confidence': 0.25, 'matching_iou': 0.5, 'nms_iou': 0.7, 'summary': summary, 'images': records}
        (ROOT / f'results/predictions/{condition}.json').write_text(json.dumps(output, indent=2)+'\n')
        print(condition, summary)


if __name__ == '__main__':
    main()
