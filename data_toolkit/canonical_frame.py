"""Explicit right-handed common frame for NEW data only.

Native glTF Y-up -> Blender/pipeline raw Z-up, then view yaw.
Raw length slot is y; formal GLB conversion (x,z,-y) makes length z/up y.
OBJ/unknown-up inputs must be separately calibrated; no implicit fallback.
This does NOT rotate existing generated meshes or transpose encoded latents.
"""
import numpy as np
FRAME_VERSION='trellis.common-frame.gltf-y-up.v1'

def native_gltf_to_internal(vertices,yaw_degrees=0.0):
    v=np.asarray(vertices)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():
        raise ValueError('finite native glTF Nx3 vertices required')
    if not np.isfinite(yaw_degrees):raise ValueError('finite yaw required')
    base=v[:,[0,2,1]].astype(np.float64)*np.array([1.,-1.,1.])
    a=np.deg2rad(yaw_degrees);c,s=np.cos(a),np.sin(a)
    rz=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    return base@rz

def legacy_ss_support_to_internal(coords,resolution=64):
    """Diagnostic/re-encoding INPUT support only; never transform SS means.
    Existing SS index0=down,index1=length,index2=negative depth.
    """
    c=np.asarray(coords)
    if resolution<=1 or c.ndim!=2 or c.shape[1]!=3 or not np.isfinite(c).all():
        raise ValueError('valid Nx3 support and grid resolution required')
    if not np.equal(c,np.floor(c)).all() or (c<0).any() or (c>=resolution).any():
        raise ValueError('integer grid support required')
    out=c[:,[2,1,0]].copy()
    out[:,[0,2]]=resolution-1-out[:,[0,2]]
    return out

def internal_to_formal_glb(vertices):
    v=np.asarray(vertices)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():raise ValueError('finite Nx3 vertices required')
    return v[:,[0,2,1]]*np.array([1.,1.,-1.])
