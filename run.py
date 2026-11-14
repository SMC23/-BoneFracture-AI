"""Punto de entrada del servidor Flask."""
import logging
import os

from dotenv import load_dotenv
from flask import Flask, render_template

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

from app.controllers.api_controller import api_bp  # noqa: E402  (carga los modelos)

app = Flask(__name__, template_folder="app/views/templates")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB por radiografía
app.register_blueprint(api_bp, url_prefix="/api")


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Iniciando servidor en http://localhost:{port}")
    app.run(debug=debug, port=port)
