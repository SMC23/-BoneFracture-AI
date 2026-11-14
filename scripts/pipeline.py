"""Utilidades para entrenar y evaluar en CPU un clasificador sobre ResNet50 congelado.

1. Se extraen una sola vez las características de ResNet50 (ImageNet) y se guardan en caché.
2. Se entrena una cabeza densa sobre esas características.
3. Se une ResNet50 + cabeza en un único .h5, que es el que usa la app.
4. Se evalúa sobre un conjunto de test que no participó en el entrenamiento.
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

from app.utils.image_helpers import IMG_SIZE, load_xray

matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
SEED = 42


def build_backbone():
    return tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,), pooling="avg"
    )


def extract_features(paths, cache_name, backbone=None, batch_size=64):
    """Devuelve las características (N, 2048) de ResNet50, guardadas en data/cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.npz")
    paths = [str(p) for p in paths]
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if list(cached["paths"]) == paths:
            print(f"[cache] {cache_name}: {len(paths)} imágenes")
            return cached["features"]

    backbone = backbone or build_backbone()
    ds = (tf.data.Dataset.from_tensor_slices(paths)
          .map(load_xray, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    print(f"[features] {cache_name}: extrayendo {len(paths)} imágenes...")
    features = backbone.predict(ds, verbose=2)
    np.savez(cache_path, features=features, paths=np.array(paths, dtype=object))
    return features


def build_head(num_classes, hidden=256, dropout=0.4):
    inputs = tf.keras.Input(shape=(2048,))
    x = tf.keras.layers.Dropout(0.2)(inputs)
    x = tf.keras.layers.Dense(hidden, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="head")


def train_head(x_train, y_train, x_val, y_val, num_classes, epochs=100):
    tf.keras.utils.set_random_seed(SEED)
    head = build_head(num_classes)
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    history = head.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=64, verbose=2,
        class_weight=dict(enumerate(weights)),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                    restore_best_weights=True)],
    )
    return head, history


def assemble_and_save(head, filename, backbone=None):
    """Une ResNet50 + cabeza en un modelo que recibe imágenes ya pasadas por preprocess_input."""
    backbone = backbone or build_backbone()
    model = tf.keras.Model(backbone.input, head(backbone.output))
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = os.path.join(WEIGHTS_DIR, filename)
    model.save(path)
    print(f"[ok] Modelo guardado en {path}")
    return path


def evaluate(head, x_test, y_test, class_names, report_name, title):
    """Calcula las métricas de test y guarda metrics.json, la matriz de confusión y las curvas ROC."""
    out_dir = os.path.join(REPORTS_DIR, report_name)
    os.makedirs(out_dir, exist_ok=True)

    y_score = head.predict(x_test, verbose=0)
    y_pred = y_score.argmax(axis=1)
    labels = list(range(len(class_names)))
    kwargs = dict(labels=labels, target_names=class_names, digits=3, zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, **kwargs)

    y_bin = label_binarize(y_test, classes=labels)
    if len(class_names) == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    aucs = {}
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        aucs[name] = float(auc(fpr, tpr))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {aucs[name]:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title(f"Curvas ROC (test) · {title}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred, labels=labels), annot=True, fmt="d",
                cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión (test) · {title}")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close()

    summary = {"n_test": int(len(y_test)), "accuracy": report["accuracy"], "auc": aucs,
               "per_class": {c: report[c] for c in class_names},
               "macro_f1": report["macro avg"]["f1-score"]}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"TEST · {title}  (n = {len(y_test)})")
    print(classification_report(y_test, y_pred, **kwargs))
    print("AUC:", {k: round(v, 3) for k, v in aucs.items()})
    print("=" * 60)
    return summary
