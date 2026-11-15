"""Lógica de decisión de BoneVisionManager con redes falsas.

No se prueba la red (eso exige los pesos), sino lo que hace la app con sus salidas:
umbral de fractura, cálculo de confianza y avisos de dominio.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.models.vision_model import BoneVisionManager
from tests.conftest import guardar_imagen

UMBRAL = 0.375  # exacto en float32 (0.37 sería 0.3700000047…), para probar el caso "justo en el umbral"


def red_falsa(salida):
    red = MagicMock()
    red.predict.return_value = np.array([salida], dtype=np.float32)
    return red


def crear_manager(p_fractura, partes=(0.98, 0.01, 0.01), tipo=(0.3, 0.7)):
    """BoneVisionManager sin ejecutar __init__ (que leería los .h5)."""
    m = BoneVisionManager.__new__(BoneVisionManager)
    m.model_parts = red_falsa(partes)
    m.model_detector = red_falsa([p_fractura, 1 - p_fractura])  # salida: [Fractura, Sano]
    m.model_type = red_falsa(tipo)                                # salida: [Conminuta, Simple]
    m.threshold = UMBRAL
    return m


def test_bajo_el_umbral_es_sano_y_no_consulta_el_tipo(radiografia_png):
    m = crear_manager(p_fractura=0.10)
    r = m.predict_image(radiografia_png)

    assert r["class"] == "Sano"
    assert r["confidence"] == pytest.approx(0.90)
    assert r["fracture_probability"] == pytest.approx(0.10)
    m.model_type.predict.assert_not_called()


@pytest.mark.parametrize("tipo,esperado", [((0.3, 0.7), "Fractura Simple"), ((0.8, 0.2), "Fractura Conminuta")])
def test_sobre_el_umbral_usa_el_tipo_mas_probable(radiografia_png, tipo, esperado):
    r = crear_manager(p_fractura=0.80, tipo=tipo).predict_image(radiografia_png)
    assert r["class"] == esperado
    assert r["confidence"] == pytest.approx(0.80 * max(tipo))  # P(fractura) * P(tipo | fractura)


def test_justo_en_el_umbral_cuenta_como_fractura(radiografia_png):
    r = crear_manager(p_fractura=UMBRAL).predict_image(radiografia_png)
    assert r["class"] != "Sano"


def test_region_se_toma_del_maximo(radiografia_png):
    r = crear_manager(p_fractura=0.1, partes=(0.05, 0.05, 0.90)).predict_image(radiografia_png)
    assert r["body_part"] == "Shoulder"
    assert r["body_part_confidence"] == pytest.approx(0.90)


def test_radiografia_normal_esta_en_dominio(radiografia_png):
    r = crear_manager(p_fractura=0.1).predict_image(radiografia_png)
    assert r["in_domain"] is True
    assert r["warnings"] == []


def test_region_poco_segura_genera_aviso(radiografia_png):
    r = crear_manager(p_fractura=0.1, partes=(0.5, 0.3, 0.2)).predict_image(radiografia_png)
    assert r["in_domain"] is False
    assert len(r["warnings"]) == 1
    assert "codo, mano u hombro" in r["warnings"][0]


def test_imagen_a_color_genera_aviso(tmp_path):
    foto = guardar_imagen(tmp_path / "foto.png", color=(220, 40, 40))
    r = crear_manager(p_fractura=0.1).predict_image(foto)
    assert r["in_domain"] is False
    assert any("color" in w for w in r["warnings"])


def test_el_resultado_es_serializable_a_json(radiografia_png):
    import json
    r = crear_manager(p_fractura=0.8).predict_image(radiografia_png)
    json.dumps(r)  # numpy.float32 rompería jsonify en el API
