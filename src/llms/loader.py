from .llm import load_llm, get_model_config as _get_model_config, get_generation_config as _get_generation_config

_model = None
_tokenizer = None
_current_model_key = None


def load_model(model_key: str = None):
    global _model, _tokenizer, _current_model_key

    if _model is None or _current_model_key != model_key:
        _model, _tokenizer = load_llm(model_key)
        _current_model_key = model_key

    return _model, _tokenizer


def get_current_model_key() -> str:
    return _current_model_key


def get_model_config(model_key: str = None) -> dict:
    return _get_model_config(model_key)


def get_generation_config(model_key: str = None) -> dict:
    return _get_generation_config(model_key)