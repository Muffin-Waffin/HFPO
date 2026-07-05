MODEL_REGISTRY = {
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-8B",
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "float16",
        "use_double_quant": True,
        "dtype": "float16",
        "device_map": "auto",
        "max_new_tokens": 256,
        "temperature": 0.2,
    },
    "phi-4": {
        "model_name": "microsoft/phi-4",
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "float16",
        "use_double_quant": True,
        "dtype": "float16",
        "device_map": "auto",
        "max_new_tokens": 256,
        "temperature": 0.2,
    },
    "llama3-8b": {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "float16",
        "use_double_quant": True,
        "dtype": "float16",
        "device_map": "auto",
        "max_new_tokens": 256,
        "temperature": 0.2,
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "float16",
        "use_double_quant": True,
        "dtype": "float16",
        "device_map": "auto",
        "max_new_tokens": 256,
        "temperature": 0.2,
    },
}

DEFAULT_MODEL = "qwen3-8b"

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.2

GA_POPULATION_SIZE = 10
GA_GENERATIONS = 5
GA_MUTATION_RATE = 0.3
GA_CROSSOVER_RATE = 0.7
GA_TOURNAMENT_SIZE = 3
EVALUATION_SUBSET_SIZE = 100
RANDOM_SEED = 42

# Adaptive Mutation Configuration
USE_ADAPTIVE_MUTATION = True
MUTATION_EVOLUTION_INTERVAL = 5
MUTATION_TOURNAMENT_SIZE = 3
MUTATION_ELITE_COUNT = 2
MIN_CHILDREN_FOR_RANKING = 10
TOP_CHILDREN_TO_KEEP = 5