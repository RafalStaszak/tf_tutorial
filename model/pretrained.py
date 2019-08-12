import tensorflow as tf


def vgg16(outputs):
    base_model = tf.keras.applications.vgg16.VGG16()
    model = tf.keras.Model(inputs=base_model.input,
                           outputs=outputs)
    return model


def vgg19(outputs):
    base_model = tf.keras.applications.vgg19.VGG19()
    model = tf.keras.Model(inputs=base_model.input,
                           outputs=outputs)
    return model


def resnet50(outputs):
    base_model = tf.keras.applications.resnet50.ResNet50()
    model = tf.keras.Model(inputs=base_model.input,
                           outputs=outputs)
    return model


def inception_v3(outputs):
    base_model = tf.keras.applications.inception_v3.InceptionV3()
    model = tf.keras.Model(inputs=base_model.input,
                           outputs=outputs)
    return model
