"""Explicit source-only acquisition, no checkpoints and no install/import of external code."""
import argparse,json,subprocess
from pathlib import Path

def fetch(lock,name,destination):
    spec=json.loads(Path(lock).read_text())[name];out=Path(destination)/name
    if out.exists():raise FileExistsError(out)
    subprocess.check_call(['git','clone','--filter=blob:none','--no-checkout',spec['url']+'.git',str(out)])
    subprocess.check_call(['git','-C',str(out),'checkout','--detach',spec['commit']])
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--name',required=True);p.add_argument('--destination',required=True);p.add_argument('--allow-network',action='store_true');a=p.parse_args()
    if not a.allow_network:p.error('Source-only network acquisition needs --allow-network; review licenses first')
    print(fetch(a.lock,a.name,a.destination))
