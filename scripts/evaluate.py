"""Evaluate validation-selected checkpoints on the held-out test set."""
import argparse
import json
from common import ROOT, CONDITIONS, checkpoint, dataset, hardware, torch, update_results
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=CONDITIONS+['both'], default='both')
    args = parser.parse_args()
    torch.set_num_threads(6)
    for condition in CONDITIONS if args.condition == 'both' else [args.condition]:
        model = YOLO(str(checkpoint(condition)))
        metrics = model.val(data=dataset(), split='test', imgsz=640, batch=1,
                            device=0 if torch.cuda.is_available() else 'cpu',
                            conf=0.001, iou=0.7, plots=True,
                            project=str(ROOT / 'runs'), name=condition+'_test')
        values = dict(mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                      precision=float(metrics.box.mp), recall=float(metrics.box.mr))
        (ROOT / f'results/metrics/{condition}_test.json').write_text(json.dumps(
            {'metrics': values, 'hardware': hardware(), 'split': 'test',
             'precision_recall': 'Ultralytics confidence at maximum smoothed F1; not fixed 0.25'}, indent=2)+'\n')
        update_results(condition, values)
        import shutil
        folder = ROOT / f'results/figures/{condition}_test'
        folder.mkdir(exist_ok=True)
        for image in metrics.save_dir.glob('*.png'):
            shutil.copy2(image, folder / image.name)


if __name__ == '__main__':
    main()
