#!/usr/bin/env python3
"""TRELLIS.1-compatible per-view GT generation from original mesh dumps.

Each view rotates the original triangle mesh into the display frame and then
voxelizes that mesh at 64^3.  No previously quantized latent/voxel coordinates
are rotated.
"""
import argparse, json, os, pickle, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import open3d as o3d

def rz(deg):
    a=np.deg2rad(deg); c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float64)

def one(job):
    root,obj,out_name,res,num_views,overwrite=job
    dump_path=os.path.join(root,'mesh_dumps',obj+'.pickle')
    render_dir=os.path.join(root,'renders_cond',obj)
    out_dir=os.path.join(root,out_name,obj); os.makedirs(out_dir,exist_ok=True)
    try:
        frames=json.load(open(os.path.join(render_dir,'transforms.json')))['frames'][:num_views]
        dump=pickle.load(open(dump_path,'rb'))
    except Exception as e:
        return obj,0,str(e)
    vertices=[]; faces=[]; offset=0
    for part in dump['objects']:
        v=np.asarray(part['vertices'])
        f=np.asarray(part['faces'])
        if len(v):
            vertices.append(v); faces.append(f.astype(np.int64)+offset); offset+=len(v)
    if not vertices: return obj,0,'empty mesh'
    vertices=np.concatenate(vertices).astype(np.float64); faces=np.concatenate(faces) if faces else None
    done=0
    for i,fr in enumerate(frames):
        path=os.path.join(out_dir,f'{i:03d}.npz')
        if os.path.exists(path) and not overwrite:
            done+=1; continue
        yaw=float(fr.get('yaw_deg',i*15.0))
        # Same display reference used by TRELLIS.1: R=R_cam@R_display.T,
        # with R_display yaw=270 degrees; row-vector application is v @ R.
        vv=vertices @ rz(yaw-270.0)
        if faces is not None and len(faces):
            mesh=o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vv),
                o3d.utility.Vector3iVector(faces))
            grid=o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                mesh, voxel_size=1/res, min_bound=(-.5,-.5,-.5), max_bound=(.5,.5,.5))
            coords=np.array([v.grid_index for v in grid.get_voxels()],dtype=np.uint8)
        else:
            coords=np.unique(np.clip(np.floor((vv+.5)*res),0,res-1),axis=0).astype(np.uint8)
        np.savez_compressed(path,coords=coords); done+=1
    return obj,done,''

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output_name',default='ss_view_voxels_mesh_t1'); ap.add_argument('--num_views',type=int,default=12); ap.add_argument('--resolution',type=int,default=64); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args()
    root=os.path.abspath(a.root); os.makedirs(os.path.join(root,a.output_name),exist_ok=True)
    objs=[os.path.basename(p)[:-7] for p in glob.glob(os.path.join(root,'mesh_dumps','*.pickle')) if os.path.isdir(os.path.join(root,'renders_cond',os.path.basename(p)[:-7]))]
    print(f'objects={len(objs)} views={len(objs)*a.num_views} output={os.path.join(root,a.output_name)}',flush=True)
    jobs=[(root,o,a.output_name,a.resolution,a.num_views,a.overwrite) for o in sorted(objs)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n,(o,d,e) in enumerate(ex.map(one,jobs),1):
            if e: print(f'{o}: {e}',flush=True)
            if n%25==0 or n==len(jobs): print(f'{n}/{len(jobs)} ({d}/{a.num_views})',flush=True)
if __name__=='__main__': main()
