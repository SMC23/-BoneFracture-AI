"""Pruebas del tokenizador del chatbot."""
from app.models.tokenizer import PAD, UNK, SimpleTokenizer


def test_tokenize_pasa_a_minusculas_y_separa_signos():
    assert SimpleTokenizer.tokenize("¿Es GRAVE la fractura?") == ["¿", "es", "grave", "la", "fractura", "?"]
    assert SimpleTokenizer.tokenize("Contexto: Codo, sano.") == ["contexto", ":", "codo", ",", "sano", "."]


def test_tokenize_conserva_tildes_y_enie():
    assert SimpleTokenizer.tokenize("Cirugía pequeña") == ["cirugía", "pequeña"]


def test_from_texts_es_determinista_con_pad_y_unk_primero():
    tok_a = SimpleTokenizer.from_texts(["b a", "c"])
    tok_b = SimpleTokenizer.from_texts(["c", "a b"])  # mismo contenido en otro orden

    assert tok_a.vocab == tok_b.vocab
    assert tok_a.vocab == {PAD: 0, UNK: 1, "a": 2, "b": 3, "c": 4}


def test_encode_usa_unk_para_palabras_desconocidas():
    tok = SimpleTokenizer.from_texts(["es grave"])
    assert tok.encode("es muy grave") == [tok.vocab["es"], tok.unk_id, tok.vocab["grave"]]


def test_decode_omite_pad_y_unk():
    tok = SimpleTokenizer.from_texts(["es grave"])
    ids = [tok.vocab["es"], tok.unk_id, tok.vocab["grave"], tok.pad_id, tok.pad_id]
    assert tok.decode(ids) == "es grave"


def test_encode_decode_ida_y_vuelta():
    tok = SimpleTokenizer.from_texts(["¿necesito cirugia?"])
    assert tok.decode(tok.encode("¿Necesito cirugia?")) == "¿ necesito cirugia ?"
