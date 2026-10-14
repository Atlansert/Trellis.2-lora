import bpy,json,base64,struct,math,os
from pathlib import Path
from mathutils import Vector
import numpy as np
out=Path('/data/lzx/TRELLIS.2/diagnostics/remediation_20261013');asset=out/'axis_markers_native.gltf'
if asset.exists():raise FileExistsError(asset)
blob=struct.pack('<9f',0,0,0,.001,0,0,0,.001,0)
markers={'DEPTH_NATIVE_X':[.125,0,0],'UP_NATIVE_Y':[0,.25,0],'LENGTH_NATIVE_Z':[0,0,.5]}
gltf={'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0,1,2]}],'nodes':[{'name':name,'mesh':0,'translation':pos} for name,pos in markers.items()],
'meshes':[{'primitives':[{'attributes':{'POSITION':0}}]}],'buffers':[{'byteLength':len(blob),'uri':'data:application/octet-stream;base64,'+base64.b64encode(blob).decode()}],
'bufferViews':[{'buffer':0,'byteLength':len(blob),'target':34962}],'accessors':[{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3','min':[0,0,0],'max':[.001,.001,0]}]}
asset.write_text(json.dumps(gltf))
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.import_scene.gltf(filepath=str(asset))
records=[]
for name,native in markers.items():
 actual=np.array(bpy.data.objects[name].matrix_world.translation);native=np.array(native);expected=np.array([native[0],-native[2],native[1]])
 assert np.allclose(actual,expected,atol=1e-7),(name,actual,expected)
 # Existing generator defaults at yaw0: axis_map(0,2,1), signs(-1,+1,-1), then row Rz(-90).
 vv=native[[0,2,1]]*np.array([-1,1,-1]);t=-math.pi/2;rot=np.array([[math.cos(t),-math.sin(t),0],[math.sin(t),math.cos(t),0],[0,0,1]])
 q=vv@rot;current=q[[2,0,1]];diagnostic=np.array([-current[2],current[1],-current[0]])
 assert np.allclose(diagnostic,actual,atol=1e-7)
 formal_current=np.array([current[0],current[2],-current[1]]);formal_expected=np.array([actual[0],actual[2],-actual[1]])
 records.append({'label':name,'native':native.tolist(),'actual_blender_import':actual.tolist(),'current_generator_ss':current.tolist(),'common_internal_diagnostic':diagnostic.tolist(),'formal_export_current_ss_support':formal_current.tolist(),'formal_export_blender_expected':formal_expected.tolist()})
result={'method':'Actual installed Blender native glTF axis-marker import, CPU no render/model; current generator formula at yaw0; no production changes',
'blender_version':bpy.app.version_string,'native_to_blender_det':1,'native_to_current_ss_det':-1,'current_ss_to_blender_diagnostic':'(-ss2,ss1,-ss0)','current_ss_to_blender_diagnostic_det':-1,
'physical_up_native_y_current_ss':'negative index0; after formalGLB remains negative x, not positive y',
'passed_marker_math':True,'training_coordinate_gate_passed':False,'records':records}
p=out/'axis_markers_result.json';assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
