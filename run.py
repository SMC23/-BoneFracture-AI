"""Punto de entrada del servidor Flask (gunicorn usa run:app)."""
import logging
import os

from dotenv import load_dotenv

from app import create_app

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Iniciando servidor en http://localhost:{port}")
    app.run(debug=debug, port=port)
