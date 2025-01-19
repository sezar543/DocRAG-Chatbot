import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print(f"GPUs are available. Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"GPU: {gpu}")
else:
    print("GPUs are not available.")