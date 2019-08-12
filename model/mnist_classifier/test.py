import tensorflow as tf

tf.enable_eager_execution()
from tensorflow.contrib import summary as summary
from model.summaries import BaseLogs
import dataset.mnist as dataset
import matplotlib.pyplot as plt
import os
from model.mnist_classifier.network import MnistClassifier

checkpoint_directory = '../../models/'

train_dataset, val_dataset, test_dataset = dataset.get(batch_size=5)

# network = RotateNet(checkpoint_directory=checkpoint_directory, suffix='bottle')

model = MnistClassifier(checkpoint_directory=checkpoint_directory)

model.restore_model()

for step, data in enumerate(test_dataset):
    inputs = data
    image, label = inputs
    outputs = model(inputs, training=True)

    # 0 0 1 0 -> argmax -> 2

    print('Label: ', label)
    print('Pred vector:', outputs)
    print('Pred: ', np.argmax(outputs, axis=1))
    plt.imshow(image[0])
    plt.show()
