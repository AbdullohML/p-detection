"""Download and safely extract the official Penn-Fudan archive."""
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'


def main():
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / 'PennFudanPed'
    if all((target / part).is_dir() for part in ['PNGImages', 'PedMasks', 'Annotation']):
        print('Official dataset already available:', target)
        return
    archive = raw / 'PennFudanPed.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.tmp')
        urlretrieve(URL, temporary)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            destination = (raw / member.filename).resolve()
            if not destination.is_relative_to(raw.resolve()):
                raise ValueError('Unsafe archive path')
        z.extractall(raw)
    assert all((target / part).is_dir() for part in ['PNGImages', 'PedMasks', 'Annotation'])
    metadata = {'url': URL, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (ROOT / 'data/splits/download.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(metadata)


if __name__ == '__main__':
    main()
