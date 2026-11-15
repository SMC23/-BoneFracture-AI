"""Entrena y evalúa el clasificador de región anatómica (Elbow / Hand / Shoulder).

Datos: MURA (Stanford ML Group), regiones codo, mano y hombro.
  - data/mura/train_valid -> entrenamiento + validación, separados POR PACIENTE
  - data/mura/test        -> partición oficial de validación de MURA (pacientes distintos), usada como test

Uso (desde la raíz del proyecto):  python -m scripts.train_body_parts
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from scripts.pipeline import BASE_DIR, SEED, assemble_and_save, build_backbone, evaluate, extract_features, train_head

MURA_DIR = Path(BASE_DIR) / "data" / "mura"
CLASSES = ["Elbow", "Hand", "Shoulder"]
IMG_EXT = {".png", ".jpg", ".jpeg"}


def listar(split_dir):
    """Devuelve (rutas, etiquetas, id_paciente). Estructura: <parte>/<paciente>/<estudio>/<imagen>."""
    paths, labels, patients = [], [], []
    for label, part in enumerate(CLASSES):
        for img in sorted((split_dir / part).rglob("*")):
            if img.suffix.lower() in IMG_EXT and not img.name.startswith("._"):
                paths.append(str(img))
                labels.append(label)
                patients.append(img.relative_to(split_dir / part).parts[0])
    return paths, np.array(labels), np.array(patients)


def main():
    paths, y, patients = listar(MURA_DIR / "train_valid")
    test_paths, y_test, test_patients = listar(MURA_DIR / "test")

    overlap = set(patients) & set(test_patients)
    assert not overlap, f"{len(overlap)} pacientes aparecen en entrenamiento y en test"

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx, val_idx = next(splitter.split(paths, y, groups=patients))
    print(f"train: {len(train_idx)} imgs · val: {len(val_idx)} imgs · test: {len(test_paths)} imgs")

    backbone = build_backbone()
    x_all = extract_features(paths, "mura_train_valid", backbone)
    x_test = extract_features(test_paths, "mura_test", backbone)

    head, _ = train_head(x_all[train_idx], y[train_idx], x_all[val_idx], y[val_idx], len(CLASSES))
    evaluate(head, x_test, y_test, CLASSES, "body_parts", "Región anatómica")
    assemble_and_save(head, "ResNet50_BodyParts.h5", backbone)


if __name__ == "__main__":
    main()
