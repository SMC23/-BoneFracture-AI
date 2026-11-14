"""Tokenizador a nivel de palabra compartido por el entrenamiento y la inferencia del chatbot."""
import re

PAD, UNK = "<PAD>", "<UNK>"
TOKEN_PATTERN = re.compile(r"\w+|[¿?.,:]")


class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.itos = {v: k for k, v in vocab.items()}
        self.pad_id = vocab[PAD]
        self.unk_id = vocab[UNK]

    @classmethod
    def from_texts(cls, textos):
        """Construye el vocabulario a partir de un corpus."""
        palabras = set()
        for t in textos:
            palabras.update(cls.tokenize(t))
        vocab = {PAD: 0, UNK: 1}
        for idx, palabra in enumerate(sorted(palabras), start=2):
            vocab[palabra] = idx
        return cls(vocab)

    @staticmethod
    def tokenize(texto):
        return TOKEN_PATTERN.findall(texto.lower())

    def encode(self, texto):
        return [self.vocab.get(w, self.unk_id) for w in self.tokenize(texto)]

    def decode(self, ids):
        return " ".join(self.itos.get(i, "") for i in ids if i not in (self.pad_id, self.unk_id))
