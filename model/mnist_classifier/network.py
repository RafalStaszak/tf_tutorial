import os
from tensorflow.contrib import eager as tfe
from system.misc import makedirs
from algorithms.image import *


class MnistClassifier(tf.keras.Model):
    MODEL_NAME = 'MnistClassifier'

    def __init__(self, input_dims=(28, 28), checkpoint_directory=None, suffix='', learning_rate=1e-4):
        super(MnistClassifier, self).__init__()

        self.checkpoint_directory = checkpoint_directory
        self.suffix = suffix

        self.input_dims = input_dims

        self.c1 = tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')
        self.c2 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu', padding='same')
        self.c3 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu', padding='same')
        self.c4 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu', padding='same')
        self.c5 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu', padding='same')
        self.c6 = tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')
        self.dense_conv = tf.keras.layers.Dense(128)
        self.dense_out = tf.keras.layers.Dense(10)
        self.max_pool = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()

        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def call(self, inputs, training=None, mask=None):
        image, label = inputs
        image = tf.expand_dims(image, axis=-1)
        image = tf.cast(image, tf.float32) / 255.0

        x = self.c1(image)
        x = self.c2(x)
        x = self.max_pool(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.max_pool(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.flatten(x)
        x = self.dense_conv(x)
        x = self.dense_out(x)
        return x

    def restore_model(self):
        """ Function to restore trained model.
        """
        self((tf.zeros((1,) + self.input_dims), [5]), training=False)
        try:
            saver = tfe.Saver(self.variables)
            saver.restore(
                tf.train.latest_checkpoint(
                    os.path.join(self.checkpoint_directory, MnistClassifier.MODEL_NAME + self.suffix)))
        except ValueError:
            print('RotateNet model cannot be found.')

    def save_model(self, step):
        """ Function to save trained model.
        """
        makedirs(os.path.join(self.checkpoint_directory, MnistClassifier.MODEL_NAME))
        tfe.Saver(
            self.variables).save(
            os.path.join(self.checkpoint_directory, MnistClassifier.MODEL_NAME + self.suffix,
                         MnistClassifier.MODEL_NAME),
            global_step=step)

    def compute_loss(self, inputs, outputs):
        image, label = inputs
        label = tf.one_hot(label, 10, dtype=tf.float32)
        loss = tf.losses.mean_squared_error(label, outputs)

        return {
            'loss': loss,
        }

    def optimize(self, losses, tape, global_step=None):
        def _apply_grads(loss, tape, var, opt, step):
            grads = tape.gradient(loss, var)
            opt.apply_gradients(zip(grads, var), global_step=step)

        _apply_grads(losses['loss'], tape,
                     self.trainable_variables, self.opt, global_step)
