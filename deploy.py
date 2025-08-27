
import argparse, subprocess, sys, os, yaml

def setup():
    print("Run the following commands:")
    print("  conda env update -f environment.yaml")
    print("  conda activate fashion3d")

def test():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow {tf.__version__}; GPUs detected: {gpus}")

def analyze(dataset_path=None):
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    p = dataset_path or cfg["dataset_path"]
    tfrec = os.path.join(p, "tfrecords")
    print(f"Dataset root: {p}")
    print(f"TFRecord dir: {tfrec}")
    files = [f for f in (os.listdir(tfrec) if os.path.isdir(tfrec) else []) if f.endswith('.tfrecord')] if os.path.exists(tfrec) else []
    print(f"Found {len(files)} TFRecord shards." if files else "No TFRecords found. Run: python scripts/make_toy_data.py")

def api():
    subprocess.run([sys.executable, "-m", "uvicorn", "api.server:app", "--reload"], check=True)

def demo():
    test(); api()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.setup: setup()
    if args.test: test()
    if args.analyze: analyze(args.dataset_path)
    if args.api: api()
    if args.demo: demo()
