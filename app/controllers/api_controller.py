"""Rutas de la API: /upload (análisis de la radiografía) y /chat (preguntas al Transformer).

Los modelos no se crean aquí: create_app() los guarda en current_app.extensions.
"""
import logging
import os
import tempfile

from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route("/upload", methods=["POST"])
def upload_image():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Tipo de archivo no permitido (usa PNG o JPG)"}), 400

    # La radiografía se guarda en un archivo temporal y se elimina tras la predicción
    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        file.save(tmp_path)
        prediction = current_app.extensions["vision"].predict_image(tmp_path)
    except Exception:
        logger.exception("Error al procesar la imagen")
        return jsonify({"error": "No se pudo procesar la imagen"}), 500
    finally:
        os.remove(tmp_path)

    return jsonify({"message": "Imagen analizada con éxito", "prediction": prediction}), 200


@api_bp.route("/chat", methods=["POST"])
def chat_interactivo():
    data = request.get_json(silent=True) or {}
    pregunta = data.get("pregunta")
    diagnostico_cnn = data.get("diagnostico_cnn")

    if not pregunta or not isinstance(diagnostico_cnn, dict) \
            or "body_part" not in diagnostico_cnn or "class" not in diagnostico_cnn:
        return jsonify({"respuesta": "Faltan datos para procesar la consulta."}), 400

    assistant = current_app.extensions["assistant"]
    if assistant is None:
        return jsonify({"respuesta": "El asistente no está disponible en este momento."}), 503

    try:
        respuesta = assistant.responder_consulta(pregunta, diagnostico_cnn)
    except Exception:
        logger.exception("Error del Transformer en /chat")
        return jsonify({"respuesta": "Ocurrió un error interno al generar la respuesta."}), 500

    return jsonify({"respuesta": respuesta}), 200
