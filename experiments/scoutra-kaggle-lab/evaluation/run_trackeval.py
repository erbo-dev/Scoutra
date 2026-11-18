"""CPU official evaluator launcher after explicit upstream/environment provisioning."""
import argparse,subprocess,sys
from pathlib import Path
from scoutra_lab.common import verify_repo

def evaluate(repo,export_root,tracker):
    verify_repo({'path':repo,'commit':'12c8791b303e0a0b50f753af204249e622d0281a'})
    import numpy as np
    if tuple(map(int,np.__version__.split('.')[:2]))>=(1,24):raise RuntimeError('Use separate original-TrackEval profile with NumPy1.23.5; no alias monkeypatch')
    root=Path(export_root).resolve()
    subprocess.check_call([sys.executable,str(Path(repo)/'scripts/run_mot_challenge.py'),'--GT_FOLDER',str(root/'gt'),'--TRACKERS_FOLDER',str(root/'trackers'),'--TRACKERS_TO_EVAL',tracker,'--SEQMAP_FILE',str(root/'seqmap.txt'),'--SKIP_SPLIT_FOL','True','--DO_PREPROC','False','--BENCHMARK','MOT17','--SPLIT_TO_EVAL','test','--METRICS','HOTA','CLEAR','Identity','--USE_PARALLEL','False','--PLOT_CURVES','False'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--export-root',required=True);p.add_argument('--tracker',default='method');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('CPU evaluation not executed; supply --execute only after reviewing dense export and compatible profile')
    evaluate(a.repo,a.export_root,a.tracker)
