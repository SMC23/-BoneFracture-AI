"""Application factory de Flask.

Separar la creación de la app de la carga de los modelos permite inyectar modelos
falsos en las pruebas, sin leer los .h5 ni descargar pesos.
"""
import logging

from flask import Flask, render_template

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB por radiografía


def _load_vision():
    from app.models.vision_model import BoneVisionManager
    from app.utils.weights import ensure_weights

    ensure_weights()
    return BoneVisionManager()


def _load_assistant():
    try:
        from app.models.llm_assistant import MedicalAssistant
        return MedicalAssistant()
    except Exception:
        logger.exception("No se pudo cargar el chatbot; /api/chat no estará disponible")
        return None


def create_app(vision=None, assistant=None):
    """Crea la app. Sin argumentos carga los modelos reales; las pruebas pasan objetos falsos."""
    flask_app = Flask(__name__, template_folder="views/templates")
    flask_app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    flask_app.extensions["vision"] = vision if vision is not None else _load_vision()
    flask_app.extensions["assistant"] = assistant if assistant is not None else _load_assistant()

    from app.controllers.api_controller import api_bp
    flask_app.register_blueprint(api_bp, url_prefix="/api")

    @flask_app.route("/")
    def home():
        return render_template("index.html")

    return flask_app
