import tensorflow as tf


def get(batch_size, split=0.2):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    total_len = len(x_train)
    val_len = int(split * total_len)
    train_len = total_len - val_len
    x_val = x_train[val_len:]
    x_train = x_train[:train_len]

    train_dataset = _prepare_dataset(x_train, batch_size)
    val_dataset = _prepare_dataset(x_val, batch_size)
    test_dataset = _prepare_dataset(x_test)

    return train_dataset, val_dataset, test_dataset


def _prepare_dataset(data, batch_size=1):
    def _norm_image(image):
        image = tf.cast(image, dtype=tf.float32)/255.0
        return image

    def _generate_noise(image):
        return tf.random.uniform(minval=-1.0, maxval=1.0, shape=tf.shape(image), dtype=tf.float32), image

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(_norm_image).map(_generate_noise)

    dataset = dataset.batch(batch_size)
    return dataset


def dictify(data):
    (noise, image) = data
    return {
        'noise': noise,
        'image': image
    }
