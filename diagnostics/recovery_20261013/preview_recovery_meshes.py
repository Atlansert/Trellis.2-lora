"""Fixed-axis CPU orthographic geometry diagnostic, not a photorealistic renderer."""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw
ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
root=Path(args.root)
items=[('base SS','ctrl_base_ss_datatest/001'),('view-side 10k','viewside_10k_001'),('view-side 20k','viewside_20k_001'),('canonical axis10 20k','axis10_20k_001_recovery_20261013')]
W,H=1400,260
canvas=Image.new('RGB',(W,H*len(items)+70),(22,24,28)); draw=ImageDraw.Draw(canvas)
draw.text((12,10),'001 / seed 42 / fixed shape20k. Raw GLB coordinates, no axis correction. Fixed scale across models.',fill='white')
draw.text((12,30),'CPU orthographic triangle projection (approximate depth sorting); geometry only, not final visual acceptance.',fill='white')
# Horizontal axis z and vertical y. Oblique view changes camera only, never the mesh.
for row,(label,tag) in enumerate(items):
    p=root/'results_infer'/tag/'001.glb'
    scene=trimesh.load(p,force='scene')
    mesh=scene.dump(concatenate=True)
    vertices=np.asarray(mesh.vertices); faces=np.asarray(mesh.faces); centers=vertices[faces].mean(axis=1); normals=np.asarray(mesh.face_normals)
    e=np.asarray(scene.extents); center=np.asarray(scene.bounds).mean(axis=0)
    draw.text((12,row*H+75),f'{label}: extents {np.round(e,4).tolist()} z/y={e[2]/e[1]:.3f}, x/y={e[0]/e[1]:.3f}',fill='white')
    for col,(title,right,depth) in enumerate([('side: horizontal=z, vertical=y',np.array([0.,0.,1.]),np.array([1.,0.,0.])),('fixed oblique camera',np.array([.5,0.,.8660254]),np.array([.8660254,0.,-.5]))]):
        image=Image.new('RGB',(680,210),(30,33,38)); d=ImageDraw.Draw(image)
        xy=np.stack(((vertices-center)@right,(vertices-center)[:,1]),axis=1)
        xy[:,0]=xy[:,0]*570+340; xy[:,1]=110-xy[:,1]*570
        pixels=np.rint(xy).astype(int)
        order=np.argsort(centers@depth)[::-1]
        brightness=(80+145*np.abs(normals@depth)).clip(0,255).astype(int)
        # Painter sorting is an intentionally inexpensive preview; GLB is retained for inspection.
        for index in order:
            poly=pixels[faces[index]]
            if (poly[:,0]<0).all() or (poly[:,0]>=680).all() or (poly[:,1]<0).all() or (poly[:,1]>=210).all(): continue
            g=int(brightness[index]); d.polygon([tuple(x) for x in poly],fill=(g,g,g))
        d.text((8,6),title,fill='white')
        canvas.paste(image,(10+col*700,row*H+110))
    print('PREVIEW',tag,flush=True)
out=Path(args.out)
if out.exists(): raise FileExistsError(out)
canvas.save(out)
print(out,flush=True)
