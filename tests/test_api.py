"""Tests de integración de la API. Requieren los pesos en weights/."""
import io
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")
SAMPLE_XRAY = os.path.join(ROOT, "tests", "fixtures", "elbow.png")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(WEIGHTS, "fracture_detector.h5")),
    reason="Pesos no disponibles (descárgalos en weights/)",
)

DIAGNOSTICO = {"body_part": "Elbow", "class": "Fractura Simple", "confidence": 0.9, "body_part_confidence": 0.9}


@pytest.fixture(scope="module")
def client():
    from run import app
    app.config["TESTING"] = True
    return app.test_client()


def test_home(client):
    assert client.get("/").status_code == 200


def test_upload_sin_archivo(client):
    assert client.post("/api/upload").status_code == 400


def test_upload_rechaza_extension(client):
    data = {"file": (io.BytesIO(b"hola"), "notas.txt")}
    assert client.post("/api/upload", data=data, content_type="multipart/form-data").status_code == 400


def test_upload_radiografia(client):
    with open(SAMPLE_XRAY, "rb") as f:
        data = {"file": (f, "elbow.png")}
        res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    pred = res.get_json()["prediction"]
    assert pred["body_part"] == "Elbow"
    assert pred["class"] in {"Fractura Conminuta", "Fractura Simple", "Sano"}
    assert 0.0 <= pred["confidence"] <= 1.0
    assert pred["in_domain"] is True and pred["warnings"] == []


def test_upload_imagen_a_color_avisa_fuera_de_dominio(client):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (220, 40, 40)).save(buf, format="PNG")
    buf.seek(0)
    res = client.post("/api/upload", data={"file": (buf, "foto.png")}, content_type="multipart/form-data")
    assert res.status_code == 200
    pred = res.get_json()["prediction"]
    assert pred["in_domain"] is False
    assert any("color" in w for w in pred["warnings"])


def test_chat_responde(client):
    res = client.post("/api/chat", json={"pregunta": "¿necesito cirugia?", "diagnostico_cnn": DIAGNOSTICO})
    assert res.status_code == 200
    assert res.get_json()["respuesta"]


def test_chat_datos_incompletos(client):
    assert client.post("/api/chat", json={"pregunta": "hola"}).status_code == 400


def test_chat_pregunta_larga_no_falla(client):
    pregunta = "¿es grave? " * 50
    res = client.post("/api/chat", json={"pregunta": pregunta, "diagnostico_cnn": DIAGNOSTICO})
    assert res.status_code == 200
