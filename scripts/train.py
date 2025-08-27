
import os, time, argparse, yaml, tensorflow as tf
from models.fashion_3d_gan import Fashion3DGAN

def tfrecord_example_parser(example_proto):
    feats = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "voxels": tf.io.FixedLenFeature([], tf.string),
        "H": tf.io.FixedLenFeature([], tf.int64),
        "W": tf.io.FixedLenFeature([], tf.int64),
        "D": tf.io.FixedLenFeature([], tf.int64),
    }
    f = tf.io.parse_single_example(example_proto, feats)
    H = tf.cast(f["H"], tf.int32); W = tf.cast(f["W"], tf.int32); D = tf.cast(f["D"], tf.int32)
    img = tf.io.decode_png(f["image"], channels=3)
    img = tf.image.resize(img, (256,256))
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    vox = tf.io.parse_tensor(f["voxels"], out_type=tf.float32)
    vox = tf.reshape(vox, (D, H, W, 1))
    return img, vox

def make_dataset(tfrecord_dir, batch):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No TFRecords in {tfrecord_dir}. Run scripts/make_toy_data.py first.")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.shuffle(256).map(tfrecord_example_parser, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--hours", type=float, default=0.25)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if cfg["gpu_config"].get("memory_growth", True):
        for d in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(d, True)
            except:
                pass

    gan = Fashion3DGAN(cfg)
    ds = make_dataset(os.path.join(cfg["dataset_path"], "tfrecords"), cfg["gpu_config"]["batch_size"])

    ckpt = tf.train.Checkpoint(gen=gan.gen, dis=gan.dis, opt_g=gan.opt_g, opt_d=gan.opt_d)
    mgr = tf.train.CheckpointManager(ckpt, "models/checkpoints", max_to_keep=5)
    if args.resume:
        ckpt.restore(args.resume)

    end_time = time.time() + args.hours*3600.0
    step = 0
    for img, vox in ds.repeat():
        logs = gan.train_step(img, vox)
        step += 1
        if step % 20 == 0:
            print({k: float(v.numpy()) for k,v in logs.items()})
        if step % 200 == 0:
            mgr.save()
        if time.time() > end_time:
            break
    mgr.save()
    print("Training finished. Checkpoints saved to models/checkpoints.")

if __name__ == "__main__":
    main()
