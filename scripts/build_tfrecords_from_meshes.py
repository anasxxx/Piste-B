import os, argparse, random, numpy as np, tensorflow as tf, trimesh
from PIL import Image

MESH_EXT={".obj",".ply",".stl"}

def mesh_to_vox(mesh_path, res=64):
    m = trimesh.load(mesh_path, force='mesh')
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.dump().geometries])
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError("Not a mesh: "+mesh_path)
    # normalize to unit cube
    b=m.bounds; c=(b[0]+b[1])/2.; m.apply_translation(-c)
    s=(b[1]-b[0]).max() or 1.0; m.apply_scale(1.0/s)
    vg = m.voxelized(1.0/res)
    mat = vg.matrix.astype(np.float32)  # (X,Y,Z)
    vol = np.zeros((res,res,res), np.float32)
    sx,sy,sz=mat.shape
    ox=max((res-sx)//2,0); oy=max((res-sy)//2,0); oz=max((res-sz)//2,0)
    vol[ox:ox+sx, oy:oy+sy, oz:oz+sz] = mat[:min(sx,res-ox), :min(sy,res-oy), :min(sz,res-oz)]
    vol = np.transpose(vol,(2,1,0))  # (Z,Y,X)
    return vol[...,None]  # (Z,Y,X,1)

def proj_front_silhouette(vol):
    v = vol[...,0]  # (Z,Y,X)
    sil = (v>0.5).any(axis=0).astype(np.uint8)  # (Y,X) 0/1
    img = np.repeat((sil*255)[...,None], 3, axis=-1).astype(np.uint8)  # RGB mask
    return np.array(Image.fromarray(img).resize((256,256), Image.NEAREST), dtype=np.uint8)

def serialize(img,vox):
    img_enc=tf.io.encode_png(tf.convert_to_tensor(img)).numpy()
    return tf.train.Example(features=tf.train.Features(feature={
        "image":  tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_enc])),
        "voxels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(vox).numpy()])),
        "D":      tf.train.Feature(int64_list=tf.train.Int64List(value=[vox.shape[0]])),
        "H":      tf.train.Feature(int64_list=tf.train.Int64List(value=[vox.shape[1]])),
        "W":      tf.train.Feature(int64_list=tf.train.Int64List(value=[vox.shape[2]])),
    })).SerializeToString()

def collect(root, max_items=None, shuffle=True):
    ms=[]
    for dp,_,fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in MESH_EXT:
                ms.append(os.path.join(dp,f))
    if shuffle: random.shuffle(ms)
    return ms[:max_items] if max_items else ms

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--res",type=int,default=64)
    ap.add_argument("--max",type=int,default=1200)
    a=ap.parse_args()
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    meshes = collect(a.root, max_items=a.max)
    print(f"Found {len(meshes)} meshes to process.")
    ok=0
    with tf.io.TFRecordWriter(a.out) as w:
        for i,mp in enumerate(meshes,1):
            try:
                vox = mesh_to_vox(mp, a.res)
                img = proj_front_silhouette(vox)
                w.write(serialize(img,vox)); ok+=1
                if ok%20==0: print(f"[{ok}] wrote examples...")
            except Exception as e:
                print(f"[{i}] skip {mp}: {e}")
    print(f"Done. Wrote {ok} examples to {a.out}")

if __name__=="__main__":
    main()
