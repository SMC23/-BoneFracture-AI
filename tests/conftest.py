"""Fixtures compartidas. Ninguna carga pesos reales ni accede a internet."""
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app import create_app

PREDICCION_FALSA = {
    "in_domain": True,
    "warnings": [],
    "body_part": "Elbow",
    "body_part_confidence": 0.99,
    "class": "Fractura Simple",
    "confidence": 0.8,
    "fracture_probability": 0.9,
}
DIAGNOSTICO = {"body_part": "Elbow", "class": "Fractura Simple"}


def guardar_imagen(path, color=None, size=(300, 200)):
    """Guarda una imagen sintética: gris con un degradado o de un color sólido."""
    if color is None:
        gradiente = np.tile(np.linspace(0, 255, size[0], dtype=np.uint8), (size[1], 1))
        Image.fromarray(gradiente).save(path)  # array 2D uint8 -> imagen en escala de grises
    else:
        Image.new("RGB", size, color).save(path)
    return str(path)


@pytest.fixture
def radiografia_png(tmp_path):
    return guardar_imagen(tmp_path / "rx.png")


@pytest.fixture
def vision():
    """Sustituto de BoneVisionManager: responde una predicción fija sin TensorFlow."""
    mock = MagicMock()
    mock.predict_image.return_value = dict(PREDICCION_FALSA)
    return mock


@pytest.fixture
def assistant():
    """Sustituto de MedicalAssistant: responde un texto fijo sin PyTorch."""
    mock = MagicMock()
    mock.responder_consulta.return_value = "Respuesta de prueba"
    return mock


@pytest.fixture
def app(vision, assistant):
    flask_app = create_app(vision=vision, assistant=assistant)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()
