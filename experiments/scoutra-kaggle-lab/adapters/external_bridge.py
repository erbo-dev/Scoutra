"""Validate exports from isolated upstream environments without inventing implementations."""
import json
from pathlib import Path
from scoutra_lab.common import sha256,read_table

def load_export(path,expected_kind,cache_sha256,method):
    path=Path(path); manifest=json.loads((path/'export.json').read_text())
    for key,value in [('status','EXECUTED'),('kind',expected_kind),('input_cache_sha256',cache_sha256),('method',method)]:
        if manifest.get(key)!=value:raise ValueError(f'External export contract mismatch: {key}')
    for required in ('upstream_commit','checkpoint_sha256','environment_file','config_file','runtime_file','license_review'):
        if not manifest.get(required):raise ValueError(f'External export lacks {required}')
    files=manifest.get('files',{})
    if not files:raise ValueError('No integrity-checked export files')
    for name,digest in files.items():
        p=(path/name).resolve()
        if not p.is_relative_to(path.resolve()) or sha256(p)!=digest:raise ValueError('Export path/integrity failure')
    return manifest
