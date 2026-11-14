# =====================================================================
# BoneFracture AI · Fine-tuning en Google Colab (TODO EN UN SOLO BLOQUE) · v2
# Detector entrenado con FracAtlas + MURA (más hospitales y equipos -> generaliza mejor)
#
# ANTES DE EJECUTAR: sube a Google Drive, en  Mi unidad/BoneFractureAI/ :
#     train_valid.zip   (MURA: Elbow/Hand/Shoulder, train + valid)
#     mura_test.zip     (MURA: set de validación oficial, usado como test)
# 1) Entorno de ejecución -> Cambiar tipo de entorno -> GPU T4
# 2) Pega este bloque completo en UNA celda y ejecútala
# 3) Acepta el permiso de Google Drive (solo al inicio). Luego puedes irte,
#    pero deja la pestaña del navegador abierta.
# 4) Resultado: Mi unidad/BoneFractureAI/bonefracture_finetuned.zip   (~1,5 h)
# =====================================================================
from google.colab import drive
drive.mount("/content/drive")

import tensorflow as tf, keras, sys
print("TensorFlow", tf.__version__, "| Keras", keras.__version__, "| Python", sys.version.split()[0])
gpus = tf.config.list_physical_devices("GPU")
print("GPU:", gpus)
assert gpus, "No hay GPU: Entorno de ejecución → Cambiar tipo de entorno → GPU"

import os, zipfile, subprocess, shutil
from huggingface_hub import hf_hub_download

DRIVE_DIR = "/content/drive/MyDrive/BoneFractureAI"
DATA = "/content/data"
os.makedirs(DATA, exist_ok=True)
os.chdir(DATA)

# MURA (Stanford ML Group, requiere registro) -> se copia desde tu Google Drive
import glob
for name, folder in [("train_valid.zip", "train_valid"), ("mura_test.zip", "test")]:
    src = os.path.join(DRIVE_DIR, name)
    if not os.path.exists(src):   # también lo busca en cualquier carpeta de Mi unidad (o un acceso directo)
        found = glob.glob(f"/content/drive/MyDrive/**/{name}", recursive=True)
        assert found, (f"No encuentro {name} en el Google Drive de ESTA cuenta. "
                       "Súbelo a Mi unidad/BoneFractureAI/ y vuelve a ejecutar.")
        src = found[0]
    print("Usando", src)
    if not os.path.exists(os.path.join(DATA, "mura", folder)):
        shutil.copy(src, name)
        zipfile.ZipFile(name).extractall("mura")
        os.remove(name)

# FracAtlas (copia en Hugging Face Hub del dataset de figshare, doi:10.6084/m9.figshare.22363012)
if not os.path.exists("fracatlas"):
    p = hf_hub_download("runananya/fracatlas", "archive (9).zip", repo_type="dataset", local_dir=".")
    zipfile.ZipFile(p).extractall("fracatlas")
    os.remove(p)

# X-ray Bone Fracture Dataset, Mendeley Data doi:10.17632/8d9kn57pdj.1 (solo originales, ~2 GB)
if not os.path.exists("fracture_types"):
    url = ("https://data.mendeley.com/public-files/datasets/8d9kn57pdj/files/"
           "e8af0356-1d2f-4ae2-8407-002348aca996/file_downloaded")
    subprocess.run(["curl", "-L", "-s", "-S", "-A", "Mozilla/5.0", "-o", "fracture_types.zip", url], check=True)
    z = zipfile.ZipFile("fracture_types.zip")
    z.extractall("fracture_types", members=[n for n in z.namelist() if n.startswith("Bone Fracture/Orginal/")])
    os.remove("fracture_types.zip")
print(os.listdir(DATA))

from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit

SEED = 42
FRACATLAS = Path(DATA) / "fracatlas" / "FracAtlas"
ORIGINALS = Path(DATA) / "fracture_types" / "Bone Fracture" / "Orginal"
MURA_TV = Path(DATA) / "mura" / "train_valid"
MURA_TEST = Path(DATA) / "mura" / "test"

def legible(p):
    try:
        with Image.open(p) as im:
            im.load()
        return True
    except Exception:
        return False

def filtrar(paths, labels, groups=None):
    ok = [legible(p) for p in paths]
    print(f"  descartadas {ok.count(False)} imágenes ilegibles")
    keep = lambda xs: [x for x, k in zip(xs, ok) if k]
    out = [str(p) for p in keep(paths)], np.array(keep(labels))
    return out + (np.array(keep(groups)),) if groups is not None else out

def split(y):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
    return tr, va, te

def mura(root):
    """MURA: study*_positive = anormal (0), study*_negative = normal (1). Grupo = paciente."""
    paths = sorted(Path(root).glob("*/*/*/*.png"))
    labels = [0 if "positive" in p.parent.name else 1 for p in paths]
    patients = [p.parent.parent.name for p in paths]
    parts = [p.parent.parent.parent.name for p in paths]
    return paths, labels, patients, parts

# Etapa A · detector: 0 = Fractura/anormal, 1 = Sano
# FracAtlas sin implantes (para que el metal no sea un atajo) + MURA (anormal = proxy de fractura)
meta = pd.read_csv(FRACATLAS / "dataset.csv")
meta = meta[meta.hardware == 0]
pfa = [FRACATLAS / "images" / ("Fractured" if f else "Non_fractured") / i
       for i, f in zip(meta.image_id, meta.fractured)]
yfa = [0 if f else 1 for f in meta.fractured]
pfa, yfa = filtrar(pfa, yfa)
split_fa = split(yfa)

pm, ym, gm, _ = mura(MURA_TV)
pm, ym, gm = filtrar(pm, ym, gm)
# Split por PACIENTE: ningún paciente aparece a la vez en train y validación
tr_m, va_m = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(pm, ym, gm))

pmt, ymt, gmt, part_mt = mura(MURA_TEST)
keep_t = [legible(p) for p in pmt]
pmt = [str(p) for p, k in zip(pmt, keep_t) if k]
ymt = np.array([y for y, k in zip(ymt, keep_t) if k])
part_mt = np.array([x for x, k in zip(part_mt, keep_t) if k])
assert not set(gm) & set(gmt), "Hay pacientes de MURA test dentro de train/valid"

rng = np.random.default_rng(SEED)
def mezclar(paths, labels):
    o = rng.permutation(len(paths))
    return [paths[i] for i in o], np.asarray(labels)[o]

pa_tr, ya_tr = mezclar([pfa[i] for i in split_fa[0]] + [pm[i] for i in tr_m],
                       np.concatenate([yfa[split_fa[0]], ym[tr_m]]))
pa_va = [pfa[i] for i in split_fa[1]] + [pm[i] for i in va_m]
ya_va = np.concatenate([yfa[split_fa[1]], ym[va_m]])
print("A · detector | train:", np.bincount(ya_tr), "| val:", np.bincount(ya_va),
      "| test FracAtlas:", np.bincount(yfa[split_fa[2]]), "| test MURA:", np.bincount(ymt))

# Etapa B · tipo: 0 = Conminuta, 1 = Simple
pb, yb = [], []
for label, folder in [(0, "Comminuted Bone Fracture"), (1, "Simple Bone Fracture")]:
    files = sorted(p for p in (ORIGINALS / folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    pb += files
    yb += [label] * len(files)
pb, yb = filtrar(pb, yb)
split_b = split(yb)
print("B · tipo | por clase:", np.bincount(yb), "| train/val/test:", [len(i) for i in split_b])

IMG_SIZE = (224, 224)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE
preprocess_input = keras.applications.resnet50.preprocess_input

def load_raw(path):
    """Escala de grises, 1 canal uint8 (así la caché en RAM ocupa 3 veces menos)."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(tf.cast(img, tf.float32), IMG_SIZE)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

# La aumentación se aplica sobre píxeles 0-255, ANTES de preprocess_input
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.06),
    keras.layers.RandomZoom(0.10),
    keras.layers.RandomContrast(0.15),
], name="augment")

def make_ds(paths, labels, n_classes, training):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), keras.utils.to_categorical(labels, n_classes)))
    ds = ds.map(lambda p, y: (load_raw(p), y), num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH)
    ds = ds.map(lambda x, y: (tf.image.grayscale_to_rgb(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

from sklearn.utils.class_weight import compute_class_weight

def build_model(n_classes):
    base = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False
    inp = keras.Input(IMG_SIZE + (3,))
    x = base(inp, training=False)              # BatchNorm siempre en modo inferencia
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

def entrenar(train_ds, val_ds, y_train, n_classes, epochs_head=5, epochs_ft=25, patience=5):
    keras.utils.set_random_seed(SEED)
    model, base = build_model(n_classes)
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)))

    # Fase 1: solo la cabeza (ResNet50 congelado)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_head, class_weight=cw, verbose=2)

    # Fase 2: fine-tuning de los bloques conv4 y conv5 (BatchNorm sigue congelado)
    base.trainable = True
    for layer in base.layers:
        layer.trainable = (layer.name.startswith(("conv4", "conv5"))
                           and not isinstance(layer, keras.layers.BatchNormalization))
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_ft, class_weight=cw, verbose=2,
                   callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                                                            restore_best_weights=True)])
    return model, (h1.history, h2.history)

import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

OUT = Path("/content/out")
(OUT / "weights").mkdir(parents=True, exist_ok=True)

def evaluar(y_true, y_pred, y_score, class_names, nombre, titulo, extra=None):
    d = OUT / "reports" / nombre
    d.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    kw = dict(labels=labels, target_names=class_names, digits=3, zero_division=0)
    rep = classification_report(y_true, y_pred, output_dict=True, **kw)
    print("=" * 60, f"\nTEST · {titulo} (n = {len(y_true)})")
    print(classification_report(y_true, y_pred, **kw))
    aucs = {}
    if y_score is not None:
        plt.figure(figsize=(7, 6))
        for i, c in enumerate(class_names):
            fpr, tpr, _ = roc_curve(np.asarray(y_true) == i, y_score[:, i])
            aucs[c] = float(auc(fpr, tpr))
            plt.plot(fpr, tpr, label=f"{c} (AUC = {aucs[c]:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.legend(loc="lower right")
        plt.xlabel("Tasa de falsos positivos")
        plt.ylabel("Tasa de verdaderos positivos")
        plt.title(f"Curvas ROC (test) · {titulo}")
        plt.savefig(d / "roc_curve.png", dpi=120, bbox_inches="tight")
        plt.show()
        print("AUC:", {k: round(v, 3) for k, v in aucs.items()})
    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=labels), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión (test) · {titulo}")
    plt.savefig(d / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.show()
    resumen = {"n_test": int(len(y_true)), "accuracy": rep["accuracy"], "auc": aucs,
               "per_class": {c: rep[c] for c in class_names},
               "macro_f1": rep["macro avg"]["f1-score"], **(extra or {})}
    (d / "metrics.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumen

def curvas(hist, nombre):
    h = {k: hist[0][k] + hist[1][k] for k in ("loss", "val_loss", "auc", "val_auc")}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k in ("loss", "val_loss"):
        ax[0].plot(h[k], label=k)
    for k in ("auc", "val_auc"):
        ax[1].plot(h[k], label=k)
    for a in ax:
        a.axvline(len(hist[0]["loss"]) - 0.5, color="gray", ls="--", lw=1)
        a.legend()
        a.set_xlabel("época")
    ax[0].set_title("Pérdida")
    ax[1].set_title("AUC (línea gris: inicio del fine-tuning)")
    d = OUT / "reports" / nombre
    d.mkdir(parents=True, exist_ok=True)
    plt.savefig(d / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.show()

# ---------------- Etapa A: detector ----------------
tr_a = make_ds(pa_tr, ya_tr, 2, True)
va_a = make_ds(pa_va, ya_va, 2, False)
detector, hist_a = entrenar(tr_a, va_a, ya_tr, 2, epochs_head=3, epochs_ft=15, patience=4)
curvas(hist_a, "fracture_detector")

# Umbral elegido en VALIDACIÓN (índice de Youden: máx. sensibilidad + especificidad), nunca en test
p_val = detector.predict(va_a, verbose=0)[:, 0]
fpr, tpr, thr = roc_curve(ya_va == 0, p_val)
umbral = float(thr[np.argmax(tpr - fpr)])
print(f"Umbral de P(fractura) elegido en validación: {umbral:.3f}")

def test_detector(paths, y, nombre, titulo, extra=None):
    score = detector.predict(make_ds(paths, y, 2, False), verbose=0)
    pred = np.where(score[:, 0] >= umbral, 0, 1)
    return evaluar(y, pred, score, ["Fractura", "Sano"], nombre, titulo, extra={"threshold": umbral, **(extra or {})}), score

res_a, score_fa = test_detector([pfa[i] for i in split_fa[2]], yfa[split_fa[2]],
                                "fracture_detector", "¿Hay fractura? · FracAtlas")
res_m, score_mt = test_detector(pmt, ymt, "fracture_detector_mura", "¿Anormal? · MURA test oficial")
auc_por_parte = {p: float(roc_auc_score(ymt[part_mt == p] == 0, score_mt[part_mt == p, 0]))
                 for p in sorted(set(part_mt))}
print("AUC MURA por región:", {k: round(v, 3) for k, v in auc_por_parte.items()})
res_m["auc_by_body_part"] = auc_por_parte
(OUT / "reports" / "fracture_detector_mura" / "metrics.json").write_text(
    json.dumps(res_m, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------- Etapa B: tipo ----------------
sel_b = lambda idx: ([pb[i] for i in idx], yb[idx])
tr_b, va_b, te_b = (make_ds(*sel_b(split_b[0]), 2, True), make_ds(*sel_b(split_b[1]), 2, False),
                    make_ds(*sel_b(split_b[2]), 2, False))
tipo, hist_b = entrenar(tr_b, va_b, yb[split_b[0]], 2)
curvas(hist_b, "fracture_type")

score_b = tipo.predict(te_b, verbose=0)
y_te_b = yb[split_b[2]]
res_b = evaluar(y_te_b, score_b.argmax(1), score_b, ["Fractura Conminuta", "Fractura Simple"],
                "fracture_type", "Simple vs. conminuta")

# ---------------- Pipeline completo (sanos de FracAtlas + fracturas de Mendeley) ----------------
sanas = split_fa[2][yfa[split_fa[2]] == 1]
ds_sanas = make_ds([pfa[i] for i in sanas], yfa[sanas], 2, False)

p_frac = np.concatenate([detector.predict(ds_sanas, verbose=0)[:, 0], detector.predict(te_b, verbose=0)[:, 0]])
tipo_pred = np.concatenate([tipo.predict(ds_sanas, verbose=0).argmax(1), score_b.argmax(1)])
y_true = np.concatenate([np.full(len(sanas), 2), y_te_b])
y_pred = np.where(p_frac >= umbral, tipo_pred, 2)

recall_externo = float(np.mean(p_frac[len(sanas):] >= umbral))
print(f"Fracturas de Mendeley detectadas por el detector (fuente externa): {recall_externo:.1%}")
res_total = evaluar(y_true, y_pred, None, ["Fractura Conminuta", "Fractura Simple", "Sano"],
                    "fracture_pipeline", "Pipeline completo",
                    extra={"external_fracture_recall": recall_externo})

# ---------------- Guardar ----------------
detector.save(OUT / "weights" / "fracture_detector.h5")
tipo.save(OUT / "weights" / "fracture_type.h5")
(OUT / "weights" / "fracture_config.json").write_text(json.dumps(
    {"fracture_threshold": umbral, "detector_data": "FracAtlas + MURA",
     "tensorflow": tf.__version__, "keras": keras.__version__}, indent=2))

# Comprobación: el .h5 se recarga y predice lo mismo
recargado = keras.models.load_model(OUT / "weights" / "fracture_detector.h5", compile=False)
lote = next(iter(va_a))[0]
assert np.allclose(recargado.predict(lote, verbose=0), detector.predict(lote, verbose=0), atol=1e-4)

shutil.make_archive("/content/bonefracture_finetuned", "zip", OUT)
print("Resumen")
print(f"  Detector FracAtlas · accuracy {res_a['accuracy']:.3f} · recall fractura "
      f"{res_a['per_class']['Fractura']['recall']:.3f} · AUC {res_a['auc']['Fractura']:.3f}")
print(f"  Detector MURA      · accuracy {res_m['accuracy']:.3f} · recall anormal "
      f"{res_m['per_class']['Fractura']['recall']:.3f} · AUC {res_m['auc']['Fractura']:.3f}")
print(f"  Tipo               · accuracy {res_b['accuracy']:.3f}")
print(f"  Pipeline           · accuracy {res_total['accuracy']:.3f} · macro F1 {res_total['macro_f1']:.3f}"
      f" · fracturas Mendeley detectadas {recall_externo:.1%}")

# Copia del resultado en Google Drive
os.makedirs(DRIVE_DIR, exist_ok=True)
shutil.copy("/content/bonefracture_finetuned.zip", os.path.join(DRIVE_DIR, "bonefracture_finetuned.zip"))
print("✅ Guardado en Google Drive: Mi unidad/BoneFractureAI/bonefracture_finetuned.zip")
