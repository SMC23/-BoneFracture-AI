"""Pruebas del API de Flask con modelos falsos (MagicMock): no se cargan .h5 ni se descargan pesos."""
import io
import os

import pytest

import app as app_package
from app import create_app
from tests.conftest import DIAGNOSTICO, PREDICCION_FALSA


def subir(client, contenido, nombre):
    return client.post("/api/upload", data={"file": (io.BytesIO(contenido), nombre)},
                       content_type="multipart/form-data")


def test_create_app_no_carga_modelos_reales_si_se_inyectan(monkeypatch, vision, assistant):
    # Si create_app intentara cargar los modelos reales, esta prueba fallaría
    def prohibido():
        raise AssertionError("no debería cargar los modelos reales")
    monkeypatch.setattr(app_package, "_load_vision", prohibido)
    monkeypatch.setattr(app_package, "_load_assistant", prohibido)

    flask_app = create_app(vision=vision, assistant=assistant)
    assert flask_app.extensions["vision"] is vision
    assert flask_app.extensions["assistant"] is assistant


def test_home_responde_la_pagina(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"BoneFracture AI" in res.data


# ---------- /api/upload ----------

def test_upload_sin_archivo_da_400(client, vision):
    assert client.post("/api/upload").status_code == 400
    vision.predict_image.assert_not_called()


@pytest.mark.parametrize("nombre", ["notas.txt", "imagen.gif", "sin_extension"])
def test_upload_rechaza_extensiones_no_permitidas(client, vision, nombre):
    res = subir(client, b"contenido", nombre)
    assert res.status_code == 400
    vision.predict_image.assert_not_called()


def test_upload_valido_devuelve_la_prediccion(client, vision, radiografia_png):
    with open(radiografia_png, "rb") as f:
        res = subir(client, f.read(), "rx.PNG")  # la extensión no distingue mayúsculas

    assert res.status_code == 200
    assert res.get_json()["prediction"] == PREDICCION_FALSA
    vision.predict_image.assert_called_once()


def test_upload_borra_el_archivo_temporal(client, vision, radiografia_png):
    existia_durante_la_prediccion = {}

    def predecir(path):
        existia_durante_la_prediccion["valor"] = os.path.exists(path)
        return PREDICCION_FALSA
    vision.predict_image.side_effect = predecir

    with open(radiografia_png, "rb") as f:
        subir(client, f.read(), "rx.png")

    ruta_temporal = vision.predict_image.call_args.args[0]
    assert existia_durante_la_prediccion["valor"] is True   # el modelo recibió un archivo real
    assert not os.path.exists(ruta_temporal)                 # y después se borró


def test_upload_error_del_modelo_da_500_y_borra_el_temporal(client, vision):
    vision.predict_image.side_effect = RuntimeError("imagen corrupta")
    res = subir(client, b"no es una imagen", "rx.png")

    assert res.status_code == 500
    assert "error" in res.get_json()
    assert not os.path.exists(vision.predict_image.call_args.args[0])


# ---------- /api/chat ----------

def test_chat_pasa_pregunta_y_diagnostico_al_asistente(client, assistant):
    res = client.post("/api/chat", json={"pregunta": "¿necesito cirugia?", "diagnostico_cnn": DIAGNOSTICO})

    assert res.status_code == 200
    assert res.get_json() == {"respuesta": "Respuesta de prueba"}
    assistant.responder_consulta.assert_called_once_with("¿necesito cirugia?", DIAGNOSTICO)


@pytest.mark.parametrize("payload", [
    {},
    {"pregunta": "hola"},
    {"pregunta": "", "diagnostico_cnn": DIAGNOSTICO},
    {"pregunta": "hola", "diagnostico_cnn": "Elbow"},
    {"pregunta": "hola", "diagnostico_cnn": {"body_part": "Elbow"}},
])
def test_chat_datos_incompletos_da_400(client, assistant, payload):
    assert client.post("/api/chat", json=payload).status_code == 400
    assistant.responder_consulta.assert_not_called()


def test_chat_sin_json_da_400(client):
    assert client.post("/api/chat", data="texto plano").status_code == 400


def test_chat_asistente_no_disponible_da_503(app, client):
    app.extensions["assistant"] = None  # como si el .pth no hubiera cargado
    res = client.post("/api/chat", json={"pregunta": "hola", "diagnostico_cnn": DIAGNOSTICO})
    assert res.status_code == 503


def test_chat_error_del_asistente_da_500(client, assistant):
    assistant.responder_consulta.side_effect = RuntimeError("fallo interno")
    res = client.post("/api/chat", json={"pregunta": "hola", "diagnostico_cnn": DIAGNOSTICO})
    assert res.status_code == 500
