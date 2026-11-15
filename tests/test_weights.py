"""Pruebas de la descarga automática de pesos (sin internet: hf_hub_download se simula)."""
from unittest.mock import MagicMock

import huggingface_hub
import pytest

from app.utils.weights import WEIGHT_FILES, ensure_weights


def crear_pesos(carpeta, archivos):
    for nombre in archivos:
        (carpeta / nombre).write_bytes(b"x")


@pytest.fixture
def descarga(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", mock)
    return mock


def test_no_descarga_si_estan_todos(tmp_path, descarga, monkeypatch):
    monkeypatch.setenv("HF_WEIGHTS_REPO", "usuario/repo")
    crear_pesos(tmp_path, WEIGHT_FILES)

    ensure_weights(str(tmp_path))
    descarga.assert_not_called()


def test_sin_variable_de_entorno_explica_que_falta(tmp_path, descarga, monkeypatch):
    monkeypatch.delenv("HF_WEIGHTS_REPO", raising=False)
    crear_pesos(tmp_path, WEIGHT_FILES[1:])  # falta el primero

    with pytest.raises(FileNotFoundError, match=WEIGHT_FILES[0]):
        ensure_weights(str(tmp_path))
    descarga.assert_not_called()


def test_descarga_solo_los_que_faltan(tmp_path, descarga, monkeypatch):
    monkeypatch.setenv("HF_WEIGHTS_REPO", "usuario/repo")
    faltantes = {"fracture_detector.h5", "fracture_config.json"}
    crear_pesos(tmp_path, [f for f in WEIGHT_FILES if f not in faltantes])

    ensure_weights(str(tmp_path))

    descargados = {c.kwargs["filename"] for c in descarga.call_args_list}
    assert descargados == faltantes
    assert all(c.kwargs["repo_id"] == "usuario/repo" for c in descarga.call_args_list)
    assert all(c.kwargs["local_dir"] == str(tmp_path) for c in descarga.call_args_list)
