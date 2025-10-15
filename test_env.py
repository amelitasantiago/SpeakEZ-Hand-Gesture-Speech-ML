import sys
import tensorflow as tf
import mediapipe as mp
import numpy as np
try:
    import jax
    jax_ver = jax.__version__
except:
    jax_ver = "N/A"

print(f"Python: {sys.version}")
print(f"TF: {tf.__version__} (Eager: {tf.executing_eagerly()})")
print(f"MP: {mp.__version__}")
print(f"NP: {np.__version__}")
print(f"JAX: {jax_ver}")
# Quick op: tf.constant([1.]).numpy()  # Shouldn't crash