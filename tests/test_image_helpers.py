"""Pruebas del preprocesamiento de radiografías (el mismo que se usó al entrenar)."""
import numpy as np
import pytest
from PIL import Image

from app.models.vision_model import _color_saturation
from app.utils.image_helpers import IMG_SIZE, load_xray
from tests.conftest import guardar_imagen


@pytest.mark.parametrize("nombre,size", [("a.png", (300, 200)), ("b.jpg", (120, 500)), ("c.png", (224, 224))])
def test_load_xray_devuelve_tensor_224x224x3_float32(tmp_path, nombre, size):
    x = load_xray(guardar_imagen(tmp_path / nombre, size=size)).numpy()
    assert x.shape == IMG_SIZE + (3,)
    assert x.dtype == np.float32


def test_load_xray_descarta_el_color(tmp_path):
    """Tras la escala de grises los 3 canales son iguales; preprocess_input solo les resta
    una media distinta a cada uno, así que la diferencia entre canales debe ser constante."""
    rng = np.random.default_rng(0)
    colores = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    ruta = tmp_path / "color.png"
    Image.fromarray(colores).save(ruta)

    x = load_xray(str(ruta)).numpy()
    assert np.std(x[..., 0] - x[..., 1]) < 1e-3
    assert np.std(x[..., 1] - x[..., 2]) < 1e-3


def test_load_xray_misma_imagen_en_gris_o_rgb_da_el_mismo_tensor(tmp_path):
    gris = guardar_imagen(tmp_path / "gris.png")
    rgb = tmp_path / "rgb.png"
    Image.open(gris).convert("RGB").save(rgb)
    np.testing.assert_allclose(load_xray(gris).numpy(), load_xray(str(rgb)).numpy(), atol=1e-4)


def test_saturacion_de_una_radiografia_gris_es_cero(radiografia_png):
    assert _color_saturation(radiografia_png) == pytest.approx(0.0, abs=1e-6)


def test_saturacion_de_una_foto_a_color_supera_el_limite(tmp_path):
    from app.models.vision_model import BoneVisionManager
    roja = guardar_imagen(tmp_path / "roja.png", color=(220, 40, 40))
    assert _color_saturation(roja) > BoneVisionManager.MAX_COLOR_SATURATION
