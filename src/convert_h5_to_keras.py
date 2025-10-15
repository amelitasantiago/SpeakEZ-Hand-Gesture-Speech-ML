# tools/convert_h5_to_keras.py
import os, sys, json, h5py, argparse
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # use TF backend for keras3

def load_tf(h5_path):
    import tensorflow as tf
    return tf.keras.models.load_model(h5_path, compile=False)

def load_keras3(h5_path):
    import keras  # standalone keras3
    return keras.models.load_model(h5_path, compile=False, safe_mode=False)

def load_via_json_patch(h5_path):
    # Replace legacy "batch_shape" with "batch_input_shape" in the saved config
    import tensorflow as tf
    with h5py.File(h5_path, "r") as f:
        cfg = f.attrs.get("model_config")
        if cfg is None:
            # TF sometimes stores it under "model_config" dataset
            if "model_config" in f:
                cfg = f["model_config"][()]
        if cfg is None:
            raise RuntimeError("No model_config found in H5; cannot patch.")
        if isinstance(cfg, bytes):
            cfg = cfg.decode("utf-8")
    patched = cfg.replace('"batch_shape":', '"batch_input_shape":')
    model = tf.keras.models.model_from_json(patched)
    # Load weights from the same H5
    model.load_weights(h5_path)
    return model

def save_as_keras(model, out_path):
    try:
        import keras
        keras.saving.save_model(model, out_path)  # keras3 native .keras
    except Exception:
        # Fallback: try model.save (some TF versions also accept .keras)
        model.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_in")
    ap.add_argument("keras_out")
    args = ap.parse_args()
    h5_in, keras_out = args.h5_in, args.keras_out
    assert os.path.exists(h5_in), f"Missing: {h5_in}"
    os.makedirs(os.path.dirname(keras_out) or ".", exist_ok=True)

    loaders = [("tf.keras", load_tf), ("keras3", load_keras3), ("json_patch", load_via_json_patch)]
    last_err = None
    for name, fn in loaders:
        try:
            print(f"[convert] Trying loader: {name}")
            model = fn(h5_in)
            print(f"[convert] Loaded via {name}")
            save_as_keras(model, keras_out)
            print(f"[convert] Saved -> {keras_out}")
            return
        except Exception as e:
            last_err = e
            print(f"[convert] {name} failed: {e}")
    raise SystemExit(f"All loaders failed. Last error: {last_err}")

if __name__ == "__main__":
    main()
