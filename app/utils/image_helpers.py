"""Preprocesamiento de radiografías compartido por la app y los scripts de entrenamiento."""
import tensorflow as tf

IMG_SIZE = (224, 224)
preprocess_input = tf.keras.applications.resnet50.preprocess_input


def load_xray(path, grayscale=True):
    """Lee una radiografía y la deja lista para ResNet50: (224, 224, 3) con preprocess_input.

    En escala de grises se descarta el color (p. ej. fotos de placas con tono azulado),
    que no aporta información diagnóstica y podría usarse como atajo.
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    if grayscale:
        img = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
    img = tf.image.resize(tf.cast(img, tf.float32), IMG_SIZE)
    return preprocess_input(img)
