

from .llm import load_llm

_model = None
_tokenizer = None


def load_model():


    global _model, _tokenizer

    if _model is None:
        _model, _tokenizer = load_llm()

    return _model, _tokenizer