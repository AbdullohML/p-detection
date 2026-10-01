"""The two conditions differ only in the explicit augmentation overrides."""
import argparse
import json
import time
import shutil
import pandas as pd
from common import ROOT, dataset, hardware, torch
from ultralytics import YOLO

STRONG = dict(hsv_h=0.03, hsv_s=0.8, hsv_v=0.5, degrees=10.0,
              translate=0.2, scale=0.7, shear=2.0)


def train(condition):
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='1 epoch in a separate run; never used as final results')
    parser.add_argument('--batch', type=int, default=16, help='Reduce only after a recorded CUDA OOM')
    args = parser.parse_args()
    name = condition + ('_smoke' if args.smoke else '')
    if (ROOT / f'runs/{name}').exists():
        raise RuntimeError(f'runs/{name} already exists; preserve or move it before a new run')
    torch.set_num_threads(6)
    config = dict(data=dataset(), imgsz=640, epochs=1 if args.smoke else 50,
                  batch=args.batch, optimizer='AdamW', lr0=0.001, weight_decay=0.0005,
                  seed=42, amp=True, pretrained=True, device=0 if torch.cuda.is_available() else 'cpu',
                  workers=2, project=str(ROOT / 'runs'), name=name, exist_ok=False,
                  deterministic=True, plots=True)
    if condition == 'augmented':
        config.update(STRONG)
    metadata = {'condition': condition, 'smoke': args.smoke, 'configuration': config,
                'hardware': hardware(), 'completed': False,
                'batch_change': None if args.batch == 16 else 'Explicit reduction after CUDA OOM; see execution log'}
    destination = ROOT / f'results/metrics/{name}_training.json'
    start = time.perf_counter()
    try:
        model = YOLO(str(ROOT / 'yolo11n.pt'))
        model.train(**config)
        metadata['training_seconds'] = time.perf_counter()-start
        metadata['effective_amp'] = bool(model.trainer.amp)
        metadata['effective_batch'] = int(model.trainer.batch_size)
        if not args.smoke and (metadata['effective_batch'] != args.batch or (torch.cuda.is_available() and not metadata['effective_amp'])):
            raise RuntimeError('Effective AMP/batch differs from requested configuration; inspect the execution log before continuing')
        run = ROOT / f'runs/{name}'
        frame = pd.read_csv(run / 'results.csv')
        frame.columns = frame.columns.str.strip()
        metadata.update(completed=not args.smoke and len(frame) == 50,
                        best_checkpoint=f'runs/{name}/weights/best.pt',
                        final_epoch_metrics=frame.iloc[-1].to_dict())
        shutil.copy2(run / 'results.csv', ROOT / f'results/metrics/{name}_epochs.csv')
        figures = ROOT / f'results/figures/{name}'
        figures.mkdir(exist_ok=True)
        for image in run.glob('*.png'):
            shutil.copy2(image, figures / image.name)
        for image in run.glob('train_batch*.jpg'):
            shutil.copy2(image, figures / image.name)
        # Revalidate the checkpoint selected by validation fitness, using validation only.
        best = YOLO(str(run / 'weights/best.pt'))
        val = best.val(data=dataset(), split='val', imgsz=640, batch=args.batch,
                       device=config['device'], project=str(ROOT / 'runs'), name=name+'_best_val')
        metadata['best_validation_metrics'] = {k: float(v) for k,v in val.results_dict.items()}
    except Exception as error:
        metadata['error'] = repr(error)
        raise
    finally:
        metadata['training_and_validation_seconds'] = time.perf_counter()-start
        destination.write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    train('baseline')
