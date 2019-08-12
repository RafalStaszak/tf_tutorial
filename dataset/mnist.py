import tensorflow as tf


def get(batch_size, split=0.2):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    total_len = len(x_train)
    val_len = int(split * total_len)
    train_len = total_len - val_len

    x_val = x_train[val_len:]
    y_val = y_train[val_len:]

    x_train = x_train[:train_len]
    y_train = y_train[:train_len]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

    return train_dataset, val_dataset, test_dataset
