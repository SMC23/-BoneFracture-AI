"""Entrena el Transformer decoder del chatbot con un corpus sintético de preguntas clínicas.

Uso (desde la raíz del proyecto):  python -m scripts.train_chatbot
"""
import os

import torch
import torch.optim as optim

from app.models.scratch_chatbot import MedicalGPTFromScratch
from app.models.tokenizer import SimpleTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "MedicalGPT_Scratch.pth")

MAX_LEN = 64
EPOCHS = 80  # suficientes para que memorice todas las combinaciones del corpus

PARTES_CUERPO = ["Codo", "Mano", "Hombro"]

# Pares (pregunta, respuesta) según el diagnóstico de la CNN
CONOCIMIENTO = {
    "Fractura Conminuta": [
        ("¿es grave la fractura?", "es una lesion grave porque el hueso se rompio en multiples fragmentos pequeños ."),
        ("¿que tipo de fractura es?", "el analisis de vision indica una fractura conminuta , lo que significa multiples fragmentos ."),
        ("¿necesito cirugia?", "si , las fracturas conminutas casi siempre requieren cirugia para alinear los fragmentos ."),
        ("¿cuanto tarda en sanar?", "el tiempo de recuperacion estimado para esta lesion es de 3 a 6 meses ."),
    ],
    "Fractura Simple": [
        ("¿es grave la fractura?", "es una lesion moderada . el hueso se rompio en dos partes pero esta alineado ."),
        ("¿que tipo de fractura es?", "la red neuronal detecto una fractura simple , un trazo unico sin desplazamiento severo ."),
        ("¿necesito cirugia?", "no necesariamente , la mayoria de las fracturas simples sanan con inmovilizacion o yeso ."),
        ("¿cuanto tarda en sanar?", "este tipo de fractura suele consolidar bien en un periodo de 6 a 8 semanas ."),
    ],
    "Sano": [
        ("¿es grave la fractura?", "no se detecto ninguna fractura . las estructuras oseas se encuentran estables ."),
        ("¿que tipo de fractura es?", "el sistema clasifica el estudio como sano . no se observan trazos de fractura ."),
        ("¿necesito cirugia?", "para nada , el hueso esta completamente intacto . no requiere intervencion quirurgica ."),
        ("¿cuanto tarda en sanar?", "al no haber fractura , el dolor por el trauma deberia disminuir en 5 a 7 dias ."),
    ],
}


def generar_corpus():
    """Combina cada región con cada diagnóstico, con el mismo formato que usa la app al preguntar."""
    return [
        f"Contexto: {parte} con {diagnostico}. Pregunta: {pregunta} Respuesta: {respuesta}"
        for parte in PARTES_CUERPO
        for diagnostico, pares in CONOCIMIENTO.items()
        for pregunta, respuesta in pares
    ]


def main():
    textos = generar_corpus()
    tokenizer = SimpleTokenizer.from_texts(textos)
    pad = tokenizer.vocab["<PAD>"]
    print(f"Corpus: {len(textos)} secuencias · vocabulario: {len(tokenizer.vocab)} tokens")

    secuencias = []
    for texto in textos:
        ids = tokenizer.encode(texto)[:MAX_LEN]
        secuencias.append(ids + [pad] * (MAX_LEN - len(ids)))
    x = torch.tensor(secuencias, dtype=torch.long)
    # Modelo autorregresivo: el objetivo es el siguiente token
    y = torch.cat([x[:, 1:], torch.zeros((x.size(0), 1), dtype=torch.long)], dim=1)

    model = MedicalGPTFromScratch(vocab_size=len(tokenizer.vocab), embed_dim=128, depth=3,
                                  num_heads=4, max_seq_len=MAX_LEN)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  época {epoch + 1}/{EPOCHS} · loss {loss.item():.4f}")

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "vocab": tokenizer.vocab}, WEIGHTS_PATH)
    print(f"[ok] Modelo guardado en {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
