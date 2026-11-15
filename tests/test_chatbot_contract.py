"""Contrato entre el entrenamiento del chatbot (scripts/train_chatbot.py) y la app (llm_assistant.py).

Si el formato del prompt de la app deja de coincidir con el del corpus, el modelo responde mal
sin que nada falle. Estas pruebas usan un checkpoint pequeño con pesos aleatorios.
"""
import pytest
import torch

from app.models.llm_assistant import TRADUCCIONES, MedicalAssistant
from app.models.scratch_chatbot import MedicalGPTFromScratch
from app.models.tokenizer import SimpleTokenizer
from scripts.train_chatbot import CONOCIMIENTO, PARTES_CUERPO, generar_corpus

INGLES = {v: k for k, v in TRADUCCIONES.items()}  # "Codo" -> "Elbow"


@pytest.fixture
def asistente(tmp_path):
    """MedicalAssistant real cargado desde un checkpoint de prueba (misma arquitectura, pesos aleatorios)."""
    torch.manual_seed(0)
    tokenizer = SimpleTokenizer.from_texts(generar_corpus())
    modelo = MedicalGPTFromScratch(len(tokenizer.vocab), embed_dim=128, depth=3, num_heads=4, max_seq_len=64)
    ruta = tmp_path / "chatbot.pth"
    torch.save({"model_state_dict": modelo.state_dict(), "vocab": tokenizer.vocab}, ruta)
    return MedicalAssistant(weights_path=str(ruta))


@pytest.fixture
def prompts_recibidos(asistente, monkeypatch):
    """Sustituye generate() para capturar el prompt exacto que la app le pasa al modelo."""
    capturados = []

    def generate_falso(idx, max_new_tokens, temperature=0.3):
        capturados.append(idx.clone())
        return torch.cat([idx, torch.full((1, max_new_tokens), asistente.tokenizer.pad_id)], dim=1)

    monkeypatch.setattr(asistente.model, "generate", generate_falso)
    return capturados


def test_corpus_cubre_todas_las_combinaciones():
    corpus = generar_corpus()
    preguntas = sum(len(pares) for pares in CONOCIMIENTO.values())
    assert len(corpus) == len(PARTES_CUERPO) * preguntas == 36
    assert len(set(corpus)) == len(corpus)


def test_todas_las_regiones_del_corpus_se_pueden_traducir():
    assert set(PARTES_CUERPO) == set(TRADUCCIONES.values())


@pytest.mark.parametrize("parte", PARTES_CUERPO)
@pytest.mark.parametrize("diagnostico", list(CONOCIMIENTO))
def test_prompt_de_la_app_coincide_con_el_corpus(asistente, prompts_recibidos, parte, diagnostico):
    for pregunta, respuesta in CONOCIMIENTO[diagnostico]:
        asistente.responder_consulta(pregunta, {"body_part": INGLES[parte], "class": diagnostico})

        prompt = prompts_recibidos[-1][0].tolist()
        frase = f"Contexto: {parte} con {diagnostico}. Pregunta: {pregunta} Respuesta: {respuesta}"
        esperado = asistente.tokenizer.encode(frase)
        assert prompt == esperado[:len(prompt)]                   # el prompt es el inicio exacto de la frase
        assert asistente.tokenizer.itos[prompt[-1]] == ":"        # y termina justo en "Respuesta:"
        assert asistente.tokenizer.unk_id not in prompt          # sin palabras desconocidas


def test_pregunta_muy_larga_no_excede_el_contexto(asistente, prompts_recibidos):
    respuesta = asistente.responder_consulta("¿es grave? " * 100, {"body_part": "Hand", "class": "Sano"})
    assert isinstance(respuesta, str)
    assert prompts_recibidos[-1].size(1) <= asistente.model.max_seq_len - 25


def test_responde_texto_con_mayuscula_inicial(asistente):
    respuesta = asistente.responder_consulta("¿necesito cirugia?", {"body_part": "Elbow", "class": "Sano"})
    assert isinstance(respuesta, str)
    assert respuesta == respuesta.strip()
    assert respuesta[:1] == respuesta[:1].upper()
