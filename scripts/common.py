"""Small shared path and result-writing helpers."""
from pathlib import Path
import json
import platform
import os
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parents[1] / '.cache/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import torch
import ultralytics
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ['baseline', 'augmented']


def dataset():
    # Ultralytics otherwise resolves relative `path` against its global datasets directory.
    config = yaml.safe_load((ROOT / 'configs/penn_fudan.yaml').read_text())
    config['path'] = str(ROOT / config['path'])
    target = ROOT / 'data/processed/penn_fudan.yaml'
    target.write_text(yaml.safe_dump(config))
    return str(target)


def checkpoint(condition):
    path = ROOT / f'runs/{condition}/weights/best.pt'
    record = ROOT / f'results/metrics/{condition}_training.json'
    if not record.exists() or not json.loads(record.read_text())['completed']:
        raise RuntimeError(f'{condition}: complete the full 50-epoch training before test evaluation')
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def hardware():
    return {'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cpu': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.machine()), 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'ultralytics': ultralytics.__version__, 'python': platform.python_version(),
            'torch_threads': torch.get_num_threads()}


def update_results(condition, values):
    path = ROOT / 'results/metrics/results.csv'
    columns = ['model','experiment','checkpoint','img_size','mAP50','mAP50_95','precision','recall','latency_ms','fps']
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    if condition not in frame.experiment.values:
        frame.loc[len(frame)] = {'model': 'YOLO11n', 'experiment': condition,
                                'checkpoint': f'runs/{condition}/weights/best.pt', 'img_size': 640}
    for key, value in values.items():
        frame.loc[frame.experiment == condition, key] = value
    frame.to_csv(path, index=False)
    lines = ['| Model | Condition | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Latency ms | FPS |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for _, row in frame.iterrows():
        vals = ['YOLO11n', row.experiment] + [f'{row[k]:.4f}' if pd.notna(row[k]) else 'pending' for k in columns[4:]]
        lines.append('| ' + ' | '.join(vals) + ' |')
    (ROOT / 'results/metrics/comparison.md').write_text('\n'.join(lines)+'\n')
