"""Build the final README results directly from measured artifacts."""
import json
from common import ROOT, CONDITIONS
import pandas as pd


def main():
    frame = pd.read_csv(ROOT / 'results/metrics/results.csv')
    assert set(frame.experiment) == set(CONDITIONS), 'Measure both conditions first'
    assert not frame.isna().any().any(), 'Evaluation or benchmark is incomplete'
    lines = ['| Model | Condition | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Latency ms | FPS |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for _, row in frame.iterrows():
        name = 'Baseline' if row.experiment == 'baseline' else 'Strong augmentation'
        lines.append(f'| {row.model} | {name} | ' + ' | '.join(f'{row[k]:.4f}' for k in
                     ['mAP50','mAP50_95','precision','recall','latency_ms','fps']) + ' |')
    table = '\n'.join(lines)+'\n'
    (ROOT / 'results/metrics/comparison.md').write_text(table)
    baseline_ap = float(frame.loc[frame.experiment == 'baseline', 'mAP50_95'].iloc[0])
    augmented_ap = float(frame.loc[frame.experiment == 'augmented', 'mAP50_95'].iloc[0])
    stats = json.loads((ROOT / 'data/splits/statistics.json').read_text())
    body = table + '\nMeasured on the same 26 held-out test images; metrics are fractions.\n\n'
    body += f'Stronger augmentation changed mAP@0.5:0.95 by {augmented_ap-baseline_ap:+.4f}. '
    body += ('It did not improve overall AP on this split.\n\n' if augmented_ap < baseline_ap else 'Interpret this single split and seed with care.\n\n')
    body += f"Dataset: {stats['total_images']} images; {stats['train_images']} train, {stats['val_images']} validation, {stats['test_images']} test. "
    body += f"{stats['instances']} pedestrian instances; mean {stats['average_pedestrians']:.3f}/image, minimum {stats['min_pedestrians']}, maximum {stats['max_pedestrians']}.\n\n"
    for condition in CONDITIONS:
        training = json.loads((ROOT / f'results/metrics/{condition}_training.json').read_text())
        benchmark = json.loads((ROOT / f'results/metrics/{condition}_benchmark.json').read_text())
        assert training['completed'] and training['effective_batch'] == training['configuration']['batch']
        body += f"{condition.capitalize()}: 50 epochs, batch {training['effective_batch']}, AMP {'enabled' if training['effective_amp'] else 'disabled'}; training command {training['training_seconds']:.2f} s "
        body += f"(including startup, plotting, checkpoint saving and Ultralytics' final validation); "
        body += f"{training['training_and_validation_seconds']:.2f} s including the additional best-checkpoint validation. "
        body += f"Checkpoint: `{training['best_checkpoint']}` (local, ignored by Git).\n\n"
    hw = benchmark['hardware']
    body += f"Hardware: {hw['gpu']}; {hw['cpu']}. Python {hw['python']}, PyTorch {hw['torch']}, CUDA runtime {hw['cuda']}, Ultralytics {hw['ultralytics']}.\n\n"
    body += "The table is generated from `results/metrics/results.csv`. Training/validation records, per-epoch curves, test metrics, and all benchmark timings are saved in `results/`. "
    body += "Precision and recall use Ultralytics' maximum-F1 operating point. The separate FP/FN analysis uses confidence 0.25.\n"
    body += '\nRegenerate the table and README results with `python scripts/report_results.py`.\n'
    readme = ROOT / 'README.md'
    text = readme.read_text()
    begin, end = text.index('## Results\n'), text.index('## Repository structure\n')
    readme.write_text(text[:begin]+'## Results\n\n'+body+'\n'+text[end:])
    print(table)


if __name__ == '__main__':
    main()
