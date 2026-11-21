MODEL_REGISTRY = {
    "whisper": "openai/whisper-large-v3",
    "xlsr": "facebook/wav2vec2-xls-r-300m",
    "w2vbert": "facebook/w2v-bert-2.0",
}


def resolve_checkpoint(model_family: str) -> str:
    return MODEL_REGISTRY[model_family]