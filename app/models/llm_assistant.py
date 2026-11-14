"""Asistente conversacional basado en el Transformer decoder implementado desde cero."""
import logging
import os

import torch

from app.models.scratch_chatbot import MedicalGPTFromScratch
from app.models.tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "MedicalGPT_Scratch.pth")

TRADUCCIONES = {"Elbow": "Codo", "Hand": "Mano", "Shoulder": "Hombro"}


class MedicalAssistant:
    def __init__(self, weights_path=WEIGHTS_PATH):
        logger.info("Cargando chatbot Transformer desde: %s", weights_path)
        checkpoint = torch.load(weights_path, map_location="cpu")

        self.tokenizer = SimpleTokenizer(checkpoint["vocab"])
        self.model = MedicalGPTFromScratch(
            vocab_size=len(checkpoint["vocab"]), embed_dim=128, depth=3, num_heads=4, max_seq_len=64
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def responder_consulta(self, pregunta_usuario, diagnostico_cnn):
        """Genera una respuesta condicionada al diagnóstico de la CNN."""
        parte_es = TRADUCCIONES.get(diagnostico_cnn["body_part"], diagnostico_cnn["body_part"])
        fractura_es = diagnostico_cnn["class"]

        # Mismo formato de secuencia que en el entrenamiento
        prompt = f"Contexto: {parte_es} con {fractura_es}. Pregunta: {pregunta_usuario} Respuesta:"
        input_ids = torch.tensor([self.tokenizer.encode(prompt)])

        # Evita superar la ventana de contexto con preguntas muy largas
        max_prompt = self.model.max_seq_len - 25
        input_ids = input_ids[:, -max_prompt:]

        with torch.no_grad():
            salida_ids = self.model.generate(input_ids, max_new_tokens=25, temperature=0.3)

        respuesta_tokens = salida_ids[0][input_ids.size(1):]
        return self.tokenizer.decode(respuesta_tokens.tolist()).strip().capitalize()
