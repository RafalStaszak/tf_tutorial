import tensorflow as tf

tf.enable_eager_execution()
from tensorflow.contrib import summary as summary
from model.summaries import BaseLogs
from model.rotate_net.network import *
import dataset.mnist as dataset
import os
from model.mnist_classifier.network import MnistClassifier


class Logs(BaseLogs):

    def summary(self, inputs, outputs, losses, step):
        super().summary(inputs, outputs, losses, step)
        with summary.always_record_summaries():
            if step.numpy() % 20 == 0:
                summary.scalar('summary/loss', losses['loss'], step=step)


checkpoint_directory = '../../models/'
log_directory = '../../logs/MnistClassifier/'

epochs = 30
lr = 1e-4

train_step = tf.Variable(0, dtype=tf.int64, trainable=False)
val_step = tf.Variable(0, dtype=tf.int64, trainable=False)
global_step = tf.train.get_or_create_global_step()

train_dataset, val_dataset, test_dataset = dataset.get(batch_size=5)

# network = RotateNet(checkpoint_directory=checkpoint_directory, suffix='bottle')

model = MnistClassifier(checkpoint_directory=checkpoint_directory, learning_rate=lr)

train_logs = Logs(os.path.join(log_directory, 'train'))
val_logs = Logs(os.path.join(log_directory, 'val'))

# network.restore_model()

for e in range(epochs):
    print('Epoch: ', e)
    for step, data in enumerate(train_dataset):
        inputs = data

        with tf.GradientTape(persistent=True) as tape:
            outputs = model(inputs, training=True)
            losses = model.compute_loss(inputs, outputs)
        model.optimize(losses, tape)

        train_logs.summary(inputs, outputs, losses, train_step)
        train_step = train_step + 1

    model.save_model(e)

    if val_dataset is not None:
        for step, data in enumerate(val_dataset):
            inputs = data
            outputs = model(inputs, training=False)
            losses = model.compute_loss(inputs, outputs)

            val_logs.summary(inputs, outputs, losses, val_step)
            val_step = val_step + 1
