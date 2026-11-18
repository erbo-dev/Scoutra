import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

BOX=('x1','y1','x2','y2')
def iou(a,b):
    ax1,ay1,ax2,ay2=[a[k] for k in BOX]; bx1,by1,bx2,by2=[b[k] for k in BOX]
    intersection=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
    union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-intersection
    return intersection/union if union>0 else 0.

def match(gt,pred,threshold=.5):
    if not gt or not pred:return []
    # Forbidden pairs receive a large cost; optimize cardinality before IoU.
    score=np.asarray([[iou(a,b) if a['class_id']==b['class_id'] else 0 for b in pred] for a in gt])
    costs=np.where(score>=threshold,1-score,1e6)
    gi,pi=linear_sum_assignment(costs)
    return [(int(g),int(p)) for g,p in zip(gi,pi) if score[g,p]>=threshold]

def load_gt(path):
    if not path or not Path(path).is_file():return None
    result=json.loads(Path(path).read_text())
    if result.get('schema_version')!='scoutra-ground-truth/1.0':raise ValueError('Unexpected annotation schema')
    keys=[(r['clip_id'],r['frame_index']) for r in result['frames']]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate annotated frame')
    return result

def grouped(rows):
    result=defaultdict(list)
    for r in rows:result[(r['clip_id'],r['frame_index'])].append(r)
    return result

def detection_metrics(pred,gt,threshold=.5,score_threshold=.25):
    if gt is None:return {'status':'NOT EVALUATED','metrics':None,'reason':'No reviewed ground truth'}
    frames=[r for r in gt['frames'] if r.get('boxes_exhaustive') is True and r.get('reviewed') is True]
    if not frames:return {'status':'NOT EVALUATED','metrics':None,'reason':'No exhaustively reviewed detection frames'}
    by=grouped([r for r in pred if r['confidence']>=score_threshold]); counts=defaultdict(Counter)
    for frame in frames:
        truth=[o for o in frame['objects'] if not o.get('ignore',False)]; guesses=by[frame['clip_id'],frame['frame_index']]
        # Ignore-region matching is not implemented; disallow such frames rather than count misleading FPs.
        if any(o.get('ignore',False) for o in frame['objects']):raise ValueError('Ignore regions require official benchmark handling; remove frame from quantitative set or implement reviewed ignore policy')
        pairs=match(truth,guesses,threshold); matched_g={g for g,p in pairs}; matched_p={p for g,p in pairs}
        for i,obj in enumerate(truth):
            h=obj['y2']-obj['y1']; size='small_h<32' if h<32 else 'medium_h32-96' if h<96 else 'large_h>=96'
            bins=[f'class:{obj["class_id"]}',f'class:{obj["class_id"]}/{size}',f'camera:{frame.get("camera_type","unknown")}/class:{obj["class_id"]}']
            if obj.get('role') in ('player','goalkeeper'):bins.append('player_or_goalkeeper')
            for b in bins:counts[b]['gt']+=1; counts[b]['tp']+=i in matched_g
        for i,obj in enumerate(guesses):counts[f'class:{obj["class_id"]}']['pred']+=1; counts[f'class:{obj["class_id"]}']['fp']+=i not in matched_p
    summary={k:{**dict(v),'recall':v['tp']/v['gt'] if v['gt'] else None} for k,v in counts.items()}
    result={'status':'EVALUATED','metrics':{'recall_and_fp':summary,'frames':len(frames),'iou_threshold':threshold,'score_threshold':score_threshold},'reason':'Only exhaustively reviewed frames; unreviewed predictions excluded'}
    # COCO AP uses all saved scores, not only the operational threshold above.
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco=COCO(); images=[]; anns=[]; results=[]; full=grouped(pred)
        classes=sorted({o['class_id'] for f in frames for o in f['objects']})
        for image_id,f in enumerate(frames,1):
            images.append({'id':image_id,'width':f['width'],'height':f['height']})
            for o in f['objects']:
                w=o['x2']-o['x1']; h=o['y2']-o['y1']; anns.append({'id':len(anns)+1,'image_id':image_id,'category_id':o['class_id'],'bbox':[o['x1'],o['y1'],w,h],'area':w*h,'iscrowd':0})
            for o in full[f['clip_id'],f['frame_index']]:results.append({'image_id':image_id,'category_id':o['class_id'],'bbox':[o['x1'],o['y1'],o['x2']-o['x1'],o['y2']-o['y1']],'score':o['confidence']})
        if not results:
            result['metrics']['coco_map']=None;result['metrics']['coco_map_reason']='No detections: use recall and FP; COCO empty-result path not evaluated'
        else:
            coco.dataset={'images':images,'annotations':anns,'categories':[{'id':c,'name':str(c)} for c in classes],'info':{}};coco.createIndex()
            evaluation=COCOeval(coco,coco.loadRes(results),'bbox');evaluation.evaluate();evaluation.accumulate();evaluation.summarize()
            result['metrics']['coco_map']=float(evaluation.stats[0]);result['metrics']['coco_ap50']=float(evaluation.stats[1])
    except ImportError:result['metrics']['coco_map']=None;result['metrics']['coco_map_reason']='pycocotools not installed'
    return result

def identity_metrics(pred,gt,anchors=None,iou_threshold=.5):
    if gt is None:return {'status':'NOT EVALUATED','metrics':None,'reason':'No identity ground truth'}
    frames=[f for f in gt['frames'] if f.get('reviewed') is True and f.get('identities_exhaustive') is True and f.get('boxes_exhaustive') is True]
    if not frames:return {'status':'NOT EVALUATED','metrics':None,'reason':'No exhaustively reviewed physical-ID frames'}
    by=grouped(pred); hist=defaultdict(Counter); mapping_perframe={}; unmatch=0; last={}; missed=set(); switches=fragments=0
    for f in sorted(frames,key=lambda r:(r['clip_id'],r['frame_index'])):
        truth=[o for o in f['objects'] if o.get('player_id') is not None and not o.get('ignore',False)]; guesses=by[f['clip_id'],f['frame_index']]
        pairs=match(truth,guesses,iou_threshold); matched={g for g,p in pairs}
        for gi,pi in pairs:
            physical=(f['clip_id'],str(truth[gi]['player_id'])); proposed=guesses[pi].get('player_id',guesses[pi].get('tracklet_id'))
            if proposed is None:continue
            proposed=str(proposed); hist[(f['clip_id'],proposed)][physical]+=1
            mapping_perframe[(f['clip_id'],f['frame_index'],physical[1])]=proposed
            if physical in last and last[physical]!=proposed:switches+=1
            if physical in missed:fragments+=1;missed.discard(physical)
            last[physical]=proposed
        for i,t in enumerate(truth):
            if i not in matched and (f['clip_id'],str(t['player_id'])) in last:missed.add((f['clip_id'],str(t['player_id'])))
        unmatch+=len(guesses)-len(pairs)
    total=sum(sum(c.values()) for c in hist.values()); pure=sum(max(c.values()) for c in hist.values())
    metrics={'matched_identity_observations':total,'matched_tracklet_purity':pure/total if total else None,'identity_switches_on_reviewed_frames':switches,'fragments_on_reviewed_frames':fragments,'unmatched_predictions':unmatch,'contaminated_predicted_identities':sum(len(c)>1 for c in hist.values()),'false_merge_pairs':sum(len(c)*(len(c)-1)//2 for c in hist.values()),'HOTA':None,'IDF1':None,'official_metric_reason':'Run TrackEval separately on dense complete sequences; these supplemental counts are not substitutes'}
    selected=[]
    for anchor in anchors or []:
        clip=anchor['clip_id']; physical=str(anchor['player_id']); seed=mapping_perframe.get((clip,anchor['frame_index'],physical))
        visible=correct=claimed=0
        for f in frames:
            if f['clip_id']!=clip:continue
            truth=[o for o in f['objects'] if not o.get('ignore',False)]
            visible+=sum(str(o.get('player_id'))==physical for o in truth)
            guesses=by[clip,f['frame_index']]
            pairs=match(truth,guesses,iou_threshold); good={p for g,p in pairs if str(truth[g].get('player_id'))==physical}
            for i,p in enumerate(guesses):
                if seed is not None and str(p.get('player_id',p.get('tracklet_id')))==seed:claimed+=1;correct+=i in good
        selected.append({'clip_id':clip,'physical_id':physical,'anchor_recovered':seed is not None,'identity_precision':correct/claimed if claimed else None,'identity_coverage':correct/visible if visible else None,'correct_observations':correct,'assigned_observations':claimed,'visible_gt_observations':visible,'false_assignments':claimed-correct})
    metrics['selected_players']=selected
    return {'status':'EVALUATED','metrics':metrics,'reason':'Anchor-only identity mapping; no best-over-video identity selection. Sparse intervals are exploratory, not MOT benchmark results.'}

def retrieval_metrics(distance,queries,gallery):
    distance=np.asarray(distance); ranks=[]; aps=[]
    if distance.shape!=(len(queries),len(gallery)):raise ValueError('Distance shape mismatch')
    for i,q in enumerate(queries):
        eligible=[j for j,g in enumerate(gallery) if g['crop_id']!=q['crop_id'] and g['source_match']==q['source_match'] and g['tracklet_id']!=q['tracklet_id']]
        if not eligible:continue
        ordered=sorted(eligible,key=lambda j:distance[i,j]); relevant=np.array([gallery[j]['player_id']==q['player_id'] for j in ordered],dtype=bool)
        if not relevant.any():continue
        ranks.append(float(relevant[0]));precision=np.cumsum(relevant)/(np.arange(len(ordered))+1);aps.append(float(precision[relevant].mean()))
    return {'status':'EVALUATED' if ranks else 'NOT EVALUATED','metrics':{'rank1':float(np.mean(ranks)) if ranks else None,'mAP':float(np.mean(aps)) if aps else None,'eligible_queries':len(ranks),'total_queries':len(queries),'excluded_queries_no_eligible_positive':len(queries)-len(ranks)},'reason':'Cross-tracklet within-match retrieval; unmatched/no-positive queries excluded and counted; identities namespaced by source match; not cross-match player recognition'}

def paired_match_bootstrap(differences,seed=1729,repeats=2000):
    values=np.asarray(differences,dtype=float)
    if len(values)<5:return {'ci95':None,'reason':'Fewer than five independent matches; descriptive only','n_matches':len(values)}
    rng=np.random.default_rng(seed);means=np.mean(rng.choice(values,size=(repeats,len(values)),replace=True),axis=1)
    return {'mean_difference':float(values.mean()),'ci95':list(map(float,np.quantile(means,[.025,.975]))),'n_matches':len(values),'unit':'source match','repeats':repeats}

def retrieval_operating_point(distance,queries,gallery,threshold):
    """Fixed validation-chosen distance threshold; no test-set threshold optimization."""
    d=np.asarray(distance);tp=fn=fp=tn=0
    if d.shape!=(len(queries),len(gallery)):raise ValueError('Distance shape mismatch')
    for i,q in enumerate(queries):
        for j,g in enumerate(gallery):
            if q['source_match']!=g['source_match'] or q['crop_id']==g['crop_id'] or q['tracklet_id']==g['tracklet_id']:continue
            same=q['player_id']==g['player_id'];accept=np.isfinite(d[i,j]) and d[i,j]<=threshold
            tp+=int(same and accept);fn+=int(same and not accept);fp+=int(not same and accept);tn+=int(not same and not accept)
    return {'distance_threshold':float(threshold),'true_match_rate':tp/(tp+fn) if tp+fn else None,'false_match_rate':fp/(fp+tn) if fp+tn else None,'positive_pairs':tp+fn,'negative_pairs':fp+tn,'false_matches':fp,'unit_warning':'Pairs are correlated; uncertainty must resample independent matches, not these pairs'}


def restrict_gt(gt,frames):
    if gt is None:return None
    valid={(f['clip_id'],f['frame_index']):f for f in frames}
    selected=[]
    for ann in gt['frames']:
        key=(ann['clip_id'],ann['frame_index'])
        if key not in valid:continue
        native=valid[key]
        if (ann['width'],ann['height'])!=(native['width'],native['height']):raise ValueError('GT/native frame resolution mismatch')
        selected.append(ann)
    return {**gt,'frames':selected}
