import tensorflow as tf
from tensorflow.contrib import summary as summary
from system.misc import makedirs
import matplotlib.pyplot as plt
import io
import os
import shutil


class BaseMetrics:

    def __init__(self) -> None:
        super().__init__()

    def compute(self, inputs, outputs, losses):
        pass


class BaseLogs:

    def __init__(self, path, metrics=None, clear_logs=True, **kwargs) -> None:
        if os.path.exists(path):
            if clear_logs:
                self._clear_logs(path)
        else:
            makedirs(path)
            
        self.writer = summary.create_file_writer(path)
        self.metrics = metrics
        self.kwargs = kwargs

    def summary(self, inputs, outputs, losses, step):
        self.writer.set_as_default()
        if self.metrics is not None:
            self.metrics.compute(inputs, outputs, losses)

    def _clear_logs(self, path):
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))


def from_plot(plot_function, **kwargs):
    fig = plt.figure()
    plot_function(**kwargs)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
