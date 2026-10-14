#!/usr/bin/env python3
"""Generate TRELLIS.1-style per-view 64^3 occupancy voxels from normalized mesh dumps.

For each rendered yaw, rotate the mesh into the view-aligned canonical frame,
then voxelize the rotated mesh.  Outputs are stored as compact npz coordinate
lists under ss_view_voxels/<sha256>/<view>.npz.
"""
import argparse, json, os, pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import open3d as o3d

def rot_z(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

def process(job):
    root, obj, res, overwrite, output_name = job
    dump_path = os.path.join(root, 'mesh_dumps', obj + '.pickle')
    cond_dir = os.path.join(root, 'renders_cond', obj)
    out_dir = os.path.join(root, output_name, obj)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(cond_dir, 'transforms.json')) as f: frames = json.load(f)['frames']
    with open(dump_path, 'rb') as f: dump = pickle.load(f)
    vs=[]
    for part in dump['objects']:
        if part['vertices'].size: vs.append(part['vertices'])
    vertices=np.concatenate(vs,axis=0).astype(np.float32)
    # Mesh dumps are already normalized to [-.5,.5]. Build the face index
    # array once, then apply TRELLIS.1's display-frame transform per view.
    fs=[]; start=0
    for part in dump['objects']:
        if part['vertices'].size and part['faces'].size:
            fs.append(part['faces'].astype(np.int64)+start)
            start += len(part['vertices'])
        elif part['vertices'].size:
            start += len(part['vertices'])
    faces=np.concatenate(fs,axis=0) if fs else None
    done=0
    for i,fr in enumerate(frames[:24]):
        out=os.path.join(out_dir,f'{i:03d}.npz')
        if os.path.exists(out) and not overwrite: done+=1; continue
        yaw=float(fr.get('yaw_deg', i*15.0))
        # TRELLIS.1 uses R = R_cam @ R_display.T.  Our renderer stores the
        # camera azimuth directly; the reference display camera is yaw=270°.
        # Thus the row-vector equivalent is a Z rotation by (yaw - 270°),
        # making the 270° view the identity/reference orientation.
        vv=vertices @ rot_z(yaw - 270.0)
        # Open3D voxelization on a point cloud is insufficient for interiors;
        # use triangle mesh if faces are available.
        if fs:
            mesh=o3d.geometry.TriangleMesh()
            mesh.vertices=o3d.utility.Vector3dVector(vv)
            mesh.triangles=o3d.utility.Vector3iVector(faces)
            grid=o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                mesh, voxel_size=1/res, min_bound=(-.5,-.5,-.5), max_bound=(.5,.5,.5))
            coords=np.array([v.grid_index for v in grid.get_voxels()],dtype=np.uint8)
        else:
            coords=np.floor((vv+.5)*res).astype(np.int32)
            coords=np.unique(np.clip(coords,0,res-1),axis=0).astype(np.uint8)
        np.savez_compressed(out, coords=coords)
        done+=1
    return obj,done

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--resolution',type=int,default=64); ap.add_argument('--workers',type=int,default=16); ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--output_name',default='ss_view_voxels_v2'); args=ap.parse_args()
    root=os.path.abspath(args.root); os.makedirs(os.path.join(root,args.output_name),exist_ok=True)
    objs=[]
    for p in sorted(os.listdir(os.path.join(root,'mesh_dumps'))):
        if p.endswith('.pickle') and os.path.isdir(os.path.join(root,'renders_cond',p[:-7])): objs.append(p[:-7])
    print(f'objects={len(objs)} views={len(objs)*24} output={os.path.join(root,args.output_name)}',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n,(obj,done) in enumerate(ex.map(process,[(root,o,args.resolution,args.overwrite,args.output_name) for o in objs]),1):
            if n%25==0 or n==len(objs): print(f'{n}/{len(objs)} {obj} ({done}/24)',flush=True)
if __name__=='__main__': main()
