"""Descarga automática de los pesos desde Hugging Face Hub si no están en weights/."""
import logging
import os

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
WEIGHT_FILES = ("ResNet50_BodyParts.h5", "fracture_detector.h5", "fracture_type.h5",
                "fracture_config.json", "MedicalGPT_Scratch.pth")


def ensure_weights(weights_dir=WEIGHTS_DIR):
    """Descarga los pesos que falten desde el repo indicado en HF_WEIGHTS_REPO."""
    missing = [f for f in WEIGHT_FILES if not os.path.exists(os.path.join(weights_dir, f))]
    if not missing:
        return

    repo_id = os.environ.get("HF_WEIGHTS_REPO")
    if not repo_id:
        raise FileNotFoundError(
            f"Faltan pesos en {weights_dir}: {', '.join(missing)}. "
            "Colócalos manualmente o define HF_WEIGHTS_REPO=<usuario>/<repo> para descargarlos."
        )

    from huggingface_hub import hf_hub_download

    os.makedirs(weights_dir, exist_ok=True)
    for filename in missing:
        logger.info("Descargando %s desde %s", filename, repo_id)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=weights_dir)
