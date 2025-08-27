
import os, argparse, numpy as np, tensorflow as tf
from PIL import Image, ImageDraw

def random_image(size=(256,256)):
    img = Image.new("RGB", size, (0,0,0))
    draw = ImageDraw.Draw(img)
    # draw a random rectangle or ellipse
    if np.random.rand() > 0.5:
        x0,y0 = np.random.randint(10, size[0]//2), np.random.randint(10, size[1]//2)
        x1,y1 = np.random.randint(size[0]//2, size[0]-10), np.random.randint(size[1]//2, size[1]-10)
        draw.rectangle([x0,y0,x1,y1], fill=(np.random.randint(256), np.random.randint(256), np.random.randint(256)))
    else:
        cx, cy = np.random.randint(50, size[0]-50), np.random.randint(50, size[1]-50)
        r = np.random.randint(20, min(size)//3)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(np.random.randint(256), np.random.randint(256), np.random.randint(256)))
    return img

def random_voxels(D=64, H=64, W=64):
    # sphere or cube inside volume
    vox = np.zeros((D,H,W,1), dtype=np.float32)
    if np.random.rand() > 0.5:
        # sphere
        cx, cy, cz = np.random.randint(16,48), np.random.randint(16,48), np.random.randint(16,48)
        r = np.random.randint(8,16)
        zz, yy, xx = np.ogrid[:D, :H, :W]
        mask = (xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2 <= r*r
        vox[mask] = 1.0
    else:
        # cube
        x0, y0, z0 = np.random.randint(8,24), np.random.randint(8,24), np.random.randint(8,24)
        s = np.random.randint(12,24)
        vox[z0:z0+s, y0:y0+s, x0:x0+s, 0] = 1.0
    return vox

def serialize_example(img_arr, vox_arr):
    img_png = Image.fromarray(img_arr.astype("uint8"))
    buf = tf.io.encode_png(tf.convert_to_tensor(np.array(img_png))).numpy()
    example = tf.train.Example(features=tf.train.Features(feature={
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[buf])),
        "voxels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(vox_arr).numpy()])),
        "H": tf.train.Feature(int64_list=tf.train.Int64List(value=[vox_arr.shape[1]])),
        "W": tf.train.Feature(int64_list=tf.train.Int64List(value=[vox_arr.shape[2]])),
        "D": tf.train.Feature(int64_list=tf.train.Int64List(value=[vox_arr.shape[0]])),
    }))
    return example.SerializeToString()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/tfrecords")
    ap.add_argument("--count", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "toy.tfrecord")
    with tf.io.TFRecordWriter(path) as w:
        for i in range(args.count):
            img = np.array(random_image())
            vox = random_voxels()
            w.write(serialize_example(img, vox))
    print(f"Wrote {args.count} examples to {path}")

if __name__ == "__main__":
    main()
