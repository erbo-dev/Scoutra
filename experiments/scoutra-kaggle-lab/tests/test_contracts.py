import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from evaluation.metrics import iou,match,identity_metrics,retrieval_metrics,paired_match_bootstrap,restrict_gt,retrieval_operating_point
from scoutra_lab.common import read_table,write_table,sha256,gpu_guard
from scoutra_lab.pipelines import detect,track,reid,recover
from scoutra_lab.dataset import validate_split

def obj(identity=None,**kw):
 return {'class_id':1,'x1':0.,'y1':0.,'x2':10.,'y2':20.,'player_id':identity,**kw}
def frame(n,objects):
 return {'clip_id':'fixture','frame_index':n,'scene_id':'fixture:scene0','reviewed':True,'boxes_exhaustive':True,'identities_exhaustive':True,'objects':objects}
def pred(n,identity,**kw):
 result=obj(None,clip_id='fixture',frame_index=n,tracklet_id=identity,**kw);result.pop('player_id');return result

class Contracts(unittest.TestCase):
 def test_iou_and_class_gating(self):
  self.assertEqual(iou(obj(),obj()),1);self.assertEqual(match([obj()],[obj(class_id=2)]),[])
 def test_match_cardinality(self):
  self.assertEqual(len(match([obj(),obj(x1=20,x2=30)],[obj(x1=20,x2=30),obj()])),2)
 def test_anchor_not_best_video_mapping(self):
  gt={'frames':[frame(0,[obj('A')]),frame(1,[obj('A')]),frame(2,[obj('B')])]}
  output=identity_metrics([pred(0,'t1'),pred(1,'t2'),pred(2,'t1')],gt,[{'clip_id':'fixture','player_id':'A','frame_index':0}])['metrics']
  selected=output['selected_players'][0]
  self.assertEqual(selected['identity_precision'],.5);self.assertEqual(selected['identity_coverage'],.5);self.assertEqual(output['false_merge_pairs'],1)
 def test_unlabeled_is_not_zero(self):
  self.assertIsNone(identity_metrics([],None)['metrics'])
  self.assertIsNone(identity_metrics([],{'frames':[{'reviewed':False}]})['metrics'])
 def test_retrieval_no_same_track_self_leak(self):
  q=[{'crop_id':'q','source_match':'m','tracklet_id':'t1','player_id':'A'}]
  gallery=[{'crop_id':'q','source_match':'m','tracklet_id':'t1','player_id':'A'},{'crop_id':'g','source_match':'m','tracklet_id':'t2','player_id':'B'},{'crop_id':'h','source_match':'m','tracklet_id':'t3','player_id':'A'}]
  metrics=retrieval_metrics([[0,.1,.2]],q,gallery)['metrics']
  self.assertEqual(metrics['rank1'],0);self.assertEqual(metrics['mAP'],.5)
 def test_no_significance_with_tiny_sample(self):self.assertIsNone(paired_match_bootstrap([.1,.2])['ci95'])
 def test_group_leakage_detected(self):
  rows=[{'clip_id':'a','source_match':'m','split':'train'},{'clip_id':'b','source_match':'m','split':'test'}]
  self.assertTrue(any('leakage' in r for r in validate_split(rows)))
 def test_parquet_empty_schema(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'p.parquet';write_table(p,[],'detections');self.assertEqual(read_table(p),[])
 def test_dryrun_never_loads_models(self):
  with tempfile.TemporaryDirectory() as t:
   for i,(function,kind) in enumerate([(detect,'detections'),(track,'tracks'),(reid,'embeddings'),(recover,'identities')]):
    path=function({'execute':False,'experiment_id':f'dry-{i}','output_root':t})
    self.assertEqual(json.loads((path/'runtime.json').read_text())['status'],'NOT EXECUTED')
    self.assertIsNone(json.loads((path/'metrics.json').read_text())['metrics'])
    self.assertEqual(read_table(path/'predictions.parquet'),[])
 def test_annotation_scope_matches_cached_inputs(self):
  annotations={'frames':[{'clip_id':'a','frame_index':0,'width':100,'height':50},{'clip_id':'a','frame_index':1,'width':100,'height':50}]}
  cache=[{'clip_id':'a','frame_index':0,'width':100,'height':50}]
  self.assertEqual(len(restrict_gt(annotations,cache)['frames']),1)
  with self.assertRaisesRegex(ValueError,'resolution'):restrict_gt(annotations,[{'clip_id':'a','frame_index':0,'width':200,'height':50}])
 def test_fixed_fmr_threshold_is_not_test_tuned(self):
  q=[{'crop_id':'q','source_match':'m','tracklet_id':'a','player_id':'A'}]
  g=[{'crop_id':'p','source_match':'m','tracklet_id':'b','player_id':'A'},{'crop_id':'n','source_match':'m','tracklet_id':'c','player_id':'B'}]
  m=retrieval_operating_point([[.1,.3]],q,g,.2)
  self.assertEqual(m['true_match_rate'],1);self.assertEqual(m['false_match_rate'],0);self.assertIsInstance(m['false_matches'],int)
 def test_gpu_guard_refuses_disabled(self):
  with self.assertRaisesRegex(RuntimeError,'execute=false'):gpu_guard({'execute':False})

if __name__=='__main__':unittest.main()
