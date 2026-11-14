"""Pipeline de visión en dos etapas: región anatómica -> tipo de fractura."""
import json
import logging
import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from app.utils.image_helpers import load_xray

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")


def _load(path):
    logger.info("Cargando modelo: %s", path)
    return load_model(path, compile=False)


def _color_saturation(filepath):
    """Saturación media (0-1). Una radiografía digital es gris (~0); una foto de una placa o pantalla, no."""
    with Image.open(filepath) as im:
        hsv = np.asarray(im.convert("RGB").resize((224, 224)).convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean() / 255.0)


class BoneVisionManager:
    """Región anatómica + fractura en dos etapas (¿hay fractura? -> ¿simple o conminuta?).

    El orden de las clases coincide con scripts/train_body_parts.py y scripts/train_fracture.py.
    """
    CLASS_PARTS = ["Elbow", "Hand", "Shoulder"]
    CLASS_TYPES = ["Fractura Conminuta", "Fractura Simple"]

    # Control de dominio (medido en MURA/FracAtlas/Mendeley): el 99 % de las radiografías de codo,
    # mano u hombro de MURA supera 0.80 de confianza en la región; las radiografías digitales tienen
    # saturación ~0, mientras que las fotos de placas o pantallas superan 0.20.
    MIN_BODY_PART_CONFIDENCE = 0.80
    MAX_COLOR_SATURATION = 0.20

    def __init__(self, weights_dir=WEIGHTS_DIR):
        self.model_parts = _load(os.path.join(weights_dir, "ResNet50_BodyParts.h5"))
        self.model_detector = _load(os.path.join(weights_dir, "fracture_detector.h5"))  # [Fractura, Sano]
        self.model_type = _load(os.path.join(weights_dir, "fracture_type.h5"))          # [Conminuta, Simple]

        # Umbral de P(fractura) elegido en validación durante el entrenamiento
        with open(os.path.join(weights_dir, "fracture_config.json"), encoding="utf-8") as f:
            self.threshold = float(json.load(f)["fracture_threshold"])
        logger.info("Umbral de fractura: %.3f", self.threshold)

    def predict_image(self, filepath):
        """Devuelve la parte del cuerpo y el diagnóstico (Conminuta / Simple / Sano) con su confianza."""
        # Mismo preprocesamiento que en el entrenamiento (escala de grises + resnet50.preprocess_input)
        batch = np.expand_dims(load_xray(filepath).numpy(), axis=0)

        preds_parts = self.model_parts.predict(batch, verbose=0)[0]
        idx_parts = int(np.argmax(preds_parts))

        p_fractura = float(self.model_detector.predict(batch, verbose=0)[0][0])
        if p_fractura < self.threshold:
            diagnostico, confianza = "Sano", 1.0 - p_fractura
        else:
            preds_type = self.model_type.predict(batch, verbose=0)[0]
            idx_type = int(np.argmax(preds_type))
            diagnostico = self.CLASS_TYPES[idx_type]
            # Probabilidad conjunta: P(fractura) * P(tipo | fractura)
            confianza = p_fractura * float(preds_type[idx_type])

        warnings = self._domain_warnings(filepath, float(preds_parts[idx_parts]))
        return {
            "in_domain": not warnings,
            "warnings": warnings,
            "body_part": self.CLASS_PARTS[idx_parts],
            "body_part_confidence": float(preds_parts[idx_parts]),
            "class": diagnostico,
            "confidence": confianza,
            "fracture_probability": p_fractura,
        }

    def _domain_warnings(self, filepath, body_part_confidence):
        """Avisos cuando la imagen no se parece a los datos de entrenamiento (resultado poco fiable)."""
        warnings = []
        if body_part_confidence < self.MIN_BODY_PART_CONFIDENCE:
            warnings.append("No parece una radiografía de codo, mano u hombro (las únicas regiones "
                            "que conoce el modelo).")
        if _color_saturation(filepath) > self.MAX_COLOR_SATURATION:
            warnings.append("La imagen tiene color: parece una foto de una placa o de una pantalla, "
                            "no una radiografía digital.")
        return warnings
