"""Convert instance masks to boxes, split deterministically, and draw checks."""
from pathlib import Path
import json
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / 'data/raw/PennFudanPed'
    names = sorted(p.name for p in (raw / 'PNGImages').glob('*.png'))
    assert names, 'Run download_dataset.py first'
    random.Random(42).shuffle(names)
    ntrain, nval = len(names) * 70 // 100, len(names) * 15 // 100
    splits = {'train': names[:ntrain], 'val': names[ntrain:ntrain+nval], 'test': names[ntrain+nval:]}
    sets = [set(v) for v in splits.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set.union(*sets)) == len(names)
    counts = []
    for split, filenames in splits.items():
        (ROOT / f'data/splits/{split}.txt').write_text('\n'.join(filenames) + '\n')
        images = ROOT / f'data/processed/images/{split}'
        labels = ROOT / f'data/processed/labels/{split}'
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        assert not (set(p.name for p in images.glob('*.png')) - set(filenames)), 'Stale images: remove processed data and rerun'
        for name in filenames:
            source = raw / 'PNGImages' / name
            image = Image.open(source).convert('RGB')
            w, h = image.size
            mask = np.asarray(Image.open(raw / 'PedMasks' / (Path(name).stem + '_mask.png')))
            assert mask.shape == (h, w)
            rows, boxes = [], []
            for instance in np.unique(mask):
                if instance == 0:
                    continue
                ys, xs = np.where(mask == instance)
                # Exclusive upper edges include every foreground pixel.
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
                assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
                values = [(x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h]
                assert all(0 <= v <= 1 for v in values)
                rows.append('0 ' + ' '.join(f'{v:.8f}' for v in values))
                boxes.append((x1, y1, x2, y2))
            counts.append(len(rows))
            shutil.copy2(source, images / name)
            (labels / (Path(name).stem + '.txt')).write_text('\n'.join(rows) + '\n')
            if split == 'train' and name in splits['train'][:6]:
                draw = ImageDraw.Draw(image)
                for box in boxes:
                    draw.rectangle(box, outline='lime', width=3)
                image.save(ROOT / 'results/figures/dataset_examples' / name)
    stats = {'seed': 42, 'total_images': len(names), **{f'{k}_images': len(v) for k,v in splits.items()},
             'instances': sum(counts), 'average_pedestrians': float(np.mean(counts)),
             'min_pedestrians': min(counts), 'max_pedestrians': max(counts)}
    (ROOT / 'data/splits/statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps(stats, indent=2))
    print('Verified: zero filename overlap; all mask/image pairs and normalized boxes valid.')


if __name__ == '__main__':
    main()
