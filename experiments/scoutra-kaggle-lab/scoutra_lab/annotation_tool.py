"""Generate a small standalone human box-labeling page; no auto-labeling or interpolation."""
import argparse,base64,json
from pathlib import Path
from .dataset import load_frames

HTML=r'''<!doctype html><meta charset="utf-8"><title>Scoutra research annotation</title>
<style>body{font:16px sans-serif;margin:20px}canvas{max-width:95vw;border:2px solid #555;cursor:crosshair}label{margin-right:10px}button{margin:8px}input{width:110px}</style>
<h1>Scoutra: reviewed frame annotations</h1><p>Drag boxes in native coordinates. Every frame starts unreviewed. Physical player IDs must stay consistent within a source match. Hidden players are not boxes. Do not guess identity or ball position.</p>
<button onclick="step(-1)">Previous</button><button onclick="step(1)">Next</button><span id="info"></span><br>
<label>Class <select id="cls"><option value="1">On-pitch person</option><option value="2">Ball</option></select></label>
<label>Role <select id="role"><option>unknown</option><option>player</option><option>goalkeeper</option><option>referee</option></select></label>
<label>Physical ID <input id="pid" placeholder="unknown = blank"></label><label>Team <input id="team"></label>
<button onclick="undo()">Undo last box</button><br>
<label><input id="reviewed" type="checkbox" onchange="flags()">Independently reviewed</label>
<label><input id="boxes" type="checkbox" onchange="flags()">ALL person/ball boxes labeled</label>
<label><input id="identities" type="checkbox" onchange="flags()">ALL person IDs verified</label>
<button onclick="save()">Export annotations JSON</button><br><canvas id="canvas"></canvas><pre id="list"></pre>
<script>
const frames=__DATA__;let n=0,start=null;const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),img=new Image();
const labels=frames.map(f=>({clip_id:f.clip_id,scene_id:f.scene_id,frame_index:f.frame_index,timestamp_s:f.timestamp_s,width:f.width,height:f.height,camera_type:'unknown',reviewed:false,boxes_exhaustive:false,identities_exhaustive:false,objects:[]}));
function el(id){return document.getElementById(id)}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0);for(const b of labels[n].objects){ctx.strokeStyle=b.class_id===2?'yellow':'lime';ctx.lineWidth=3;ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);ctx.fillStyle='red';ctx.font='22px sans-serif';ctx.fillText(b.player_id||'unknown',b.x1,b.y1+24)}el('list').textContent=JSON.stringify(labels[n],null,2)}
function load(){const f=frames[n];el('info').textContent=`${n+1}/${frames.length} ${f.clip_id} frame ${f.frame_index}`;canvas.width=f.width;canvas.height=f.height;img.onload=draw;img.src=f.data;el('reviewed').checked=labels[n].reviewed;el('boxes').checked=labels[n].boxes_exhaustive;el('identities').checked=labels[n].identities_exhaustive}
function flags(){Object.assign(labels[n],{reviewed:el('reviewed').checked,boxes_exhaustive:el('boxes').checked,identities_exhaustive:el('identities').checked});draw()}
function step(d){n=Math.max(0,Math.min(frames.length-1,n+d));load()}
function point(e){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]}
canvas.onpointerdown=e=>{start=point(e);canvas.setPointerCapture(e.pointerId)};
canvas.onpointerup=e=>{if(!start)return;let end=point(e);let b={x1:Math.max(0,Math.min(start[0],end[0])),y1:Math.max(0,Math.min(start[1],end[1])),x2:Math.min(canvas.width,Math.max(start[0],end[0])),y2:Math.min(canvas.height,Math.max(start[1],end[1])),class_id:Number(el('cls').value),role:el('role').value,player_id:el('pid').value.trim()||null,team_id:el('team').value.trim()||null,ignore:false};start=null;if(b.x2-b.x1<1||b.y2-b.y1<1)return;labels[n].objects.push(b);labels[n].reviewed=false;el('reviewed').checked=false;draw()};
function undo(){labels[n].objects.pop();labels[n].reviewed=false;el('reviewed').checked=false;draw()}
function save(){const blob=new Blob([JSON.stringify({schema_version:'scoutra-ground-truth/1.0',coordinate_system:'native_image_xyxy_pixels',frames:labels},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='detections.json';a.click();URL.revokeObjectURL(a.href)}
load();</script>'''

def generate(frames_path,output,every=30,limit=100,start_index=0):
    frames=load_frames(frames_path)[start_index::every][:limit]
    if not frames:raise ValueError('No frames to annotate')
    data=[{**f,'data':'data:image/png;base64,'+base64.b64encode(Path(f['path']).read_bytes()).decode()} for f in frames]
    text=HTML.replace('__DATA__',json.dumps(data).replace('</','<\\/'))
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(text)
    return len(frames)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frames',required=True);p.add_argument('--output',required=True);p.add_argument('--every',type=int,default=30);p.add_argument('--limit',type=int,default=100);p.add_argument('--start-index',type=int,default=0);a=p.parse_args()
    if a.every<1 or a.limit<1 or a.start_index<0:raise ValueError('Positive sampling parameters required')
    print(generate(a.frames,a.output,a.every,a.limit,a.start_index),'frames; annotations must be manually reviewed')
