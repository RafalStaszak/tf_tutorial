import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import LeakyReLU


class UpSample(tf.keras.Model):
    """
    Simple UpSample made by a Convolution with given stride followed by a batch normalization.
    """

    def __init__(self, filters=1, kernel_size=3, momentum=0.99, strides=2, dropout=0.0,
                 activation=tf.nn.relu, apply_bn=True, regularizer=l2(0.01)):
        super(UpSample, self).__init__()
        self.activation = activation
        self.apply_bn = apply_bn
        self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, 'same',
                                                    kernel_regularizer=regularizer)
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        # simple convolution followed by the batch normalization
        x = self.conv(inputs)
        if self.apply_bn is True:
            x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        # activation after batch normalization
        return self.activation(x)


class UpSample3D(tf.keras.Model):
    """
    Simple UpSample made by a Convolution with given stride followed by a batch normalization.
    """

    def __init__(self, filters=1, kernel_size=3, momentum=0.99, strides=2, dropout=0.0,
                 activation=tf.nn.relu, apply_bn=True, regularizer=l2(0.01)):
        super(UpSample3D, self).__init__()
        self.activation = activation
        self.apply_bn = apply_bn
        self.conv = tf.keras.layers.Conv3DTranspose(filters, kernel_size, strides, 'same',
                                                    kernel_regularizer=regularizer)
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        # simple convolution followed by the batch normalization
        x = self.conv(inputs)
        if self.apply_bn is True:
            x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        # activation after batch normalization
        return self.activation(x)


class UpSampleBilinear(tf.keras.Model):
    """
    Simple UpSample made by a Convolution with given stride followed by a batch normalization.
    """

    def __init__(self, filters=1, kernel_size=3, momentum=0.99, strides=2, dropout=0.0,
                 activation=tf.nn.relu, target_size=(1, 1), apply_bn=True, regularizer=l2(0.01)):
        super(UpSampleBilinear, self).__init__()
        self.activation = activation
        self.apply_bn = apply_bn
        self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, 'same',
                                                    kernel_regularizer=regularizer)
        self.up_sample = lambda x: tf.image.resize_images(x, target_size)
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        # simple convolution followed by the batch normalization
        x = self.conv(inputs)
        if self.apply_bn is True:
            x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.up_sample(x)
        x = self.dropout(x, training=training)

        # activation after batch normalization
        return x


class UpSampleBilinearInstance(tf.keras.Model):

    def __init__(self, filters=1, kernel_size=3, upsample_size=(2, 2), dropout=0.0, activation=LeakyReLU(0.2),
                 apply_in=True,
                 regularizer=l2(0.01)):
        super().__init__()
        self.activation = activation
        self.apply_in = apply_in
        self.instance_norm = InstanceNormalization(center=False, scale=False) if apply_in else None
        self.up = tf.keras.layers.UpSampling2D(size=upsample_size)
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same',
                                           kernel_regularizer=regularizer)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        x = self.up(inputs)
        x = self.conv(x)
        x = self.activation(x)
        if self.apply_in is True:
            x = self.instance_norm(x, training=training)
        x = self.dropout(x, training=training)

        return x


class DownSample(tf.keras.Model):

    def __init__(self, filters=1, kernel_size=3, momentum=0.99, strides=2, dropout=0.0,
                 activation=tf.nn.relu, apply_bn=True, regularizer=l2(0.01)):
        super(DownSample, self).__init__()
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                           padding='same', kernel_regularizer=regularizer)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        # run convolution and maxpooling in parallel
        x = self.conv(inputs)
        if self.apply_bn is True:
            x = self.bn(x, training=training)
        x = self.dropout(x, training=training)

        # activation used at the end of layer
        return self.activation(x)


class DownSample3D(tf.keras.Model):

    def __init__(self, filters=1, kernel_size=3, momentum=0.99, strides=2, dropout=0.0,
                 activation=tf.nn.relu, apply_bn=True, regularizer=l2(0.01)):
        super(DownSample3D, self).__init__()
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.conv = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides,
                                           padding='same', kernel_regularizer=regularizer)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        # run convolution and maxpooling in parallel
        x = self.conv(inputs)
        if self.apply_bn is True:
            x = self.bn(x, training=training)
        x = self.dropout(x, training=training)

        # activation used at the end of layer
        return self.activation(x)


class DownSampleInstance(tf.keras.Model):

    def __init__(self, filters=1, kernel_size=3, strides=2, dropout=0.0, activation=LeakyReLU(0.2), apply_in=True,
                 regularizer=l2(0.01)):
        super().__init__()
        self.activation = activation
        self.apply_in = apply_in
        self.instance_norm = InstanceNormalization(center=False, scale=False) if apply_in else None
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                           kernel_regularizer=regularizer)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.activation(x)
        if self.apply_in is True:
            x = self.instance_norm(x, training=training)
        x = self.dropout(x, training=training)

        return x


class DownSampleBilinear(tf.keras.Model):

    def __init__(self, filters=1, kernel_size=3, momentum=0.99, dropout=0.0,
                 activation=tf.nn.relu, target_size=(1, 1), apply_bn=True, regularizer=l2(0.01)):
        super(DownSampleBilinear, self).__init__()
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.down_sample = lambda x: tf.image.resize_images(x, target_size)
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                           kernel_regularizer=regularizer)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        # run convolution and maxpooling in parallel
        x = self.conv(inputs)
        if self.apply_bn is True:
            x = self.bn(x, training=training)
        x = self.down_sample(x)
        x = self.dropout(x, training=training)

        # activation used at the end of layer
        return self.activation(x)


class DownSampleMaxPool(tf.keras.Model):

    def __init__(self, filters=1, kernel_size=3, momentum=0.99, strides=2, dropout=0.0,
                 activation=tf.nn.relu, apply_bn=True, regularizer=l2(0.01)):
        super(DownSampleMaxPool, self).__init__()
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.down_sample = tf.keras.layers.MaxPool2D(strides, strides, padding='same')
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                           kernel_regularizer=regularizer)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        # run convolution and maxpooling in parallel
        x = self.conv(inputs)
        if self.apply_bn is True:
            x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.down_sample(x)
        x = self.dropout(x, training=training)

        # activation used at the end of layer
        return x


class DownSampleConv(tf.keras.Model):

    def __init__(self, filters=1, kernel_size=3, momentum=0.99, strides=1, dropout=0.0,
                 activation=tf.nn.relu, apply_bn=True, regularizer=l2(0.01)):
        super(DownSampleConv, self).__init__()
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.down_sample = tf.keras.layers.MaxPool2D(strides, strides, padding='same')
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                           kernel_regularizer=regularizer)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training=None, mask=None):
        # run convolution and maxpooling in parallel
        x = self.conv(inputs)
        x = self.down_sample(x)
        if self.apply_bn is True:
            x = self.bn(x, training=training)
        x = self.dropout(x, training=training)

        # activation used at the end of layer
        return self.activation(x)


# ====================== ResNet Cell and Block ======================================
class ResNetCell(tf.keras.Model):

    def __init__(self, kernel_size, filters, dilation=1, dropout=0.0, momentum=0.99):
        super(ResNetCell, self).__init__()
        self.dropout = dropout

        self.bn1 = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')

        self.bn2 = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        # no dilation convolutions (preactivation)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)

        # dilation convolutions (preactivation
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # dropout added only if rate > 0
        x = self.dropout(x, training=training)

        # output without activation
        return x + inputs


class ResNetBlock(tf.keras.Model):
    """
    Mode = 0 - downsample
    Mode = 1 - upsample
    Mode = 2 - convolution without stride to prepare data to have correct number of features
    Mode > 2 and < 0 - None
    """

    DOWNSAMPLE = 0
    UPSAMPLE = 1
    CONV = 2
    NONE = 3

    def __init__(self, num_cells, out_f, kernel_size, dilation=1, dropout=0.0, momentum=0.99, stride=2, mode=0):
        super().__init__()
        self.mode = mode
        if mode == ResNetBlock.DOWNSAMPLE:
            self.pre_cell = DownSample(out_f, kernel_size, momentum, stride)
        elif mode == ResNetBlock.UPSAMPLE:
            self.pre_cell = UpSample(out_f, kernel_size, momentum, stride)
        elif mode == ResNetBlock.CONV:
            self.pre_cell = tf.keras.layers.Conv2D(out_f, kernel_size, 1, 'same', activation=tf.nn.relu)
        else:
            self.pre_cell = None

        self.cells = [ResNetCell(kernel_size, out_f, dilation, dropout, momentum) for _ in range(num_cells)]

    def call(self, inputs, training=None, mask=None):
        x = inputs

        #  run pre-cell, this is operation like upsampling and downsampling since resnet cells don't have stride
        # todo: refactor
        if self.pre_cell is not None:
            if self.mode == ResNetBlock.CONV:
                x = self.pre_cell(inputs)
            else:
                x = self.pre_cell(inputs, training=training)

        # run all residual cells in a sequence
        for cell in self.cells:
            x = cell(x, training=training)

        return x

    @staticmethod
    def spec(num_cells, out_f, kernel_size, dilation=1, dropout=0.0, momentum=0.99, stride=2, mode=0):
        return {
            'num_cells': num_cells,
            'out_f': out_f,
            'kernel_size': kernel_size,
            'dilation': dilation,
            'dropout': dropout,
            'momentum': momentum,
            'stride': stride,
            'mode': mode
        }


class ResNet3DCell(ResNetCell):

    def __init__(self, kernel_size, filters, dilation=1, dropout=0.0, momentum=0.99):
        super(tf.keras.Model, self).__init__()
        self.dropout = dropout

        self.bn1 = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.conv1 = tf.keras.layers.Conv3D(filters, kernel_size, padding='same')

        self.bn2 = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.conv2 = tf.keras.layers.Conv3D(filters, kernel_size, padding='same', dilation_rate=dilation)

        self.dropout = tf.keras.layers.Dropout(dropout)


class ResNet3DBlock(ResNetBlock):

    def __init__(self, num_cells, out_f, kernel_size, dilation=1, dropout=0.0, momentum=0.99, stride=2, mode=0):
        super(tf.keras.Model, self).__init__()
        self.mode = mode
        if mode == ResNetBlock.DOWNSAMPLE:
            self.pre_cell = DownSample3D(out_f, kernel_size, momentum, stride)
        elif mode == ResNetBlock.UPSAMPLE:
            self.pre_cell = UpSample3D(out_f, kernel_size, momentum, stride)
        elif mode == ResNetBlock.CONV:
            self.pre_cell = tf.keras.layers.Conv3D(out_f, kernel_size, 1, 'same', activation=tf.nn.relu)
        else:
            self.pre_cell = None

        self.cells = [ResNet3DCell(kernel_size, out_f, dilation, dropout, momentum) for _ in range(num_cells)]


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.initializers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.initializers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = tf.keras.layers.InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = tf.keras.backend.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = tf.keras.backend.mean(inputs, reduction_axes, keepdims=True)
        stddev = tf.keras.backend.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = tf.keras.backend.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = tf.keras.backend.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
