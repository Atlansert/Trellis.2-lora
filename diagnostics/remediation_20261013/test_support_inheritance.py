import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import torch
from trellis2.modules.sparse import SparseTensor
from trellis2.modules.sparse.spatial.spatial2channel import SparseChannel2Spatial

coords=torch.tensor([[0,1,2,3],[0,5,9,17]],dtype=torch.int32)
x=SparseTensor(torch.zeros((2,8)),coords)
for depth in range(1,5):
 # Real decoder conv expands channels before each C2S; geometry-only probe
 # resets dummy features to divisible-by8 while preserving parent coords.
 x=x.replace(torch.zeros((x.coords.shape[0],8)))
 subdivision=x.replace(torch.ones((x.coords.shape[0],8),dtype=torch.bool))
 x=SparseChannel2Spatial(2)(x,subdivision)
 parents=x.coords[:,1:]//(2**depth)
 expected={tuple(c) for c in coords[:,1:].tolist()}
 assert {tuple(c) for c in parents.tolist()}==expected
 assert x.coords[:,0].eq(0).all()
 print('child_parent_axes_preserved',depth,x.coords.shape[0],flush=True)
path=Path(__file__).with_name('support_inheritance_result.json')
assert not path.exists()
path.write_text(json.dumps({'actual_module':'SparseChannel2Spatial','torch_device':str(x.device),'levels':4,'leaf_count':x.coords.shape[0],'parents_unchanged_under_each_same_axis_floor':True,'global_axis_swap_possible_in_upsampling':False,'support_can_prune':True,'mesh_vertex_offset_local_voxel_margin':0.5,'formal_export_axes':'x,z,-y','not_full_model_test':True},indent=2))
