import tensorflow as tf
m = tf.keras.models.load_model("C:/Users/ameli/speakez/models/final/asl_baseline_cnn_128_final.h5", compile=False)
m.save("C:/Users/ameli/speakez/models/final/best_asl_baseline_128.keras")  # new format
