"""Pruebas del Transformer decoder con pesos aleatorios (no necesita el .pth entrenado)."""
import pytest
import torch

from app.models.scratch_chatbot import MedicalGPTFromScratch

VOCAB, MAX_LEN = 30, 16


@pytest.fixture
def modelo():
    torch.manual_seed(0)
    m = MedicalGPTFromScratch(vocab_size=VOCAB, embed_dim=32, depth=2, num_heads=4, max_seq_len=MAX_LEN)
    m.eval()  # sin dropout: la salida es determinista
    return m


def test_forward_devuelve_logits_por_posicion(modelo):
    x = torch.randint(0, VOCAB, (3, 10))
    logits, loss = modelo(x)
    assert logits.shape == (3, 10, VOCAB)
    assert loss is None


def test_forward_calcula_la_loss_con_targets(modelo):
    x = torch.randint(0, VOCAB, (2, 8))
    _, loss = modelo(x, targets=x)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0


def test_atencion_causal_el_futuro_no_afecta_al_pasado(modelo):
    """Propiedad clave de un decoder tipo GPT: la posición t solo ve las posiciones <= t."""
    x = torch.randint(0, VOCAB, (1, 12))
    x_modificado = x.clone()
    x_modificado[0, 8] = (x[0, 8] + 1) % VOCAB  # cambia solo el token de la posición 8

    with torch.no_grad():
        antes, _ = modelo(x)
        despues, _ = modelo(x_modificado)

    # Posiciones 0-7 (anteriores al cambio): idénticas
    assert torch.allclose(antes[:, :8], despues[:, :8], atol=1e-6)
    # Posición 8 en adelante: sí cambian (la prueba no pasa por casualidad)
    assert not torch.allclose(antes[:, 8:], despues[:, 8:], atol=1e-6)


def test_rechaza_secuencias_mas_largas_que_el_contexto(modelo):
    with pytest.raises(AssertionError):
        modelo(torch.randint(0, VOCAB, (1, MAX_LEN + 1)))


def test_generate_agrega_exactamente_n_tokens_validos(modelo):
    prompt = torch.randint(0, VOCAB, (1, 5))
    with torch.no_grad():
        salida = modelo.generate(prompt, max_new_tokens=7)

    assert salida.shape == (1, 12)
    assert torch.equal(salida[:, :5], prompt)  # el prompt no se modifica
    assert int(salida.min()) >= 0 and int(salida.max()) < VOCAB


def test_generate_recorta_el_contexto_si_se_pasa_del_limite(modelo):
    prompt = torch.randint(0, VOCAB, (1, MAX_LEN))
    with torch.no_grad():
        salida = modelo.generate(prompt, max_new_tokens=5)  # 16 + 5 > 16 sin fallar
    assert salida.shape == (1, MAX_LEN + 5)
