"""Verify the files in this release without network access or third-party libraries."""
from pathlib import Path
from hashlib import sha256
import json

ROOT=Path(__file__).resolve().parent

def verify():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf8'))
    for name,expected in manifest['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT):raise ValueError('Manifest path escapes repository')
        if not p.is_file() or sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Release file missing or changed: '+name)
    print('Release integrity passed: '+str(len(manifest['files']))+' files.')
    return manifest

if __name__=='__main__':verify()
