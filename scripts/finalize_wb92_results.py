#!/usr/bin/env python3
"""Index immutable WB92 attempt evidence without changing scientific gates."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, read_public, write_new, digest

p = argparse.ArgumentParser()
p.add_argument('--output-root', type=Path, required=True)
a = p.parse_args(); out = a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb92_'):
    p.error('WB92 output required')
summary = read_public(out/'summary.json')
attempts = []
for directory in sorted((ROOT/'outputs').glob('mc24_four_station_wb92_common_seed_acts_v*')):
    files = [f for f in directory.iterdir() if f.is_file() and
             (f.suffix in ('.json', '.log', '.sub', '.out', '.err'))]
    for event in sorted((directory/'events').glob('*')):
        files.extend(f for f in event.iterdir() if f.is_file() and f.suffix in ('.json', '.log'))
    attempts.append({'path': str(directory.relative_to(ROOT)),
                     'hashes': {str(f.relative_to(ROOT)): digest(f) for f in files}})
manifest = {'output_root': str(out.relative_to(ROOT)), 'gate': summary['gate'],
            'baseline_count': summary['baseline_count'], 'geometry_count': summary['geometry_count'],
            'attempts': attempts, 'index_script_sha256': digest(Path(__file__)),
            'qualification': 'NOT_EVALUATED', 'association': 'NOT_EVALUATED', 'covariance': 'NOT_EVALUATED',
            'all_mode_conditions_Htheta': 'UNKNOWN', 'held_out_access': False,
            'population_independence': 'seen_development_only'}
write_new(out/'result_integrity.json', manifest)
write_new(ROOT/'docs/wb92_common_seed_acts_result_manifest.json', {
    **manifest, 'result_integrity_sha256': digest(out/'result_integrity.json')})
print({k: manifest[k] for k in ('output_root','gate','baseline_count','geometry_count')})
