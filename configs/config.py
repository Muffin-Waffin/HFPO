# ============================================================================
# FedGAPrompt Configuration File
# ============================================================================
# This file contains all configuration parameters for the FedGAPrompt framework.
# 
# ============================================================================
# WHAT TO CHANGE vs WHAT NOT TO CHANGE:
# ============================================================================
# 
# SAFE TO CHANGE (tune these for your experiments):
#   - DEFAULT_MODEL: Choose which model to use from MODEL_REGISTRY
#   - MAX_NEW_TOKENS: Max tokens to generate (higher = longer outputs, more memory)
#   - TEMPERATURE: Sampling temperature (higher = more creative, lower = deterministic)
#   - GA_POPULATION_SIZE: GA population size (larger = more diversity, slower)
#   - GA_GENERATIONS: Number of GA generations (more = better optimization, slower)
#   - GA_MUTATION_RATE: Mutation probability (higher = more exploration)
#   - GA_CROSSOVER_RATE: Crossover probability (higher = more recombination)
#   - GA_TOURNAMENT_SIZE: Tournament size for selection (larger = more selective)
#   - EVALUATION_SUBSET_SIZE: Evaluation subset size (larger = better eval, slower)
#   - RANDOM_SEED: Random seed for reproducibility
#   - USE_ADAPTIVE_MUTATION: Enable/disable adaptive mutation
#   - MUTATION_EVOLUTION_INTERVAL: Generations between mutation adaptations
#   - MUTATION_TOURNAMENT_SIZE: Tournament size for mutation selection
#   - MUTATION_ELITE_COUNT: Number of elite mutations to keep
#   - MIN_CHILDREN_FOR_RANKING: Min children needed for ranking
#   - TOP_CHILDREN_TO_KEEP: Top children to keep per generation
#
# CHANGE WITH CAUTION (requires understanding of model/hardware):
#   - MODEL_REGISTRY entries: Only add new models if you know the HF model ID
#   - load_in_4bit, quant_type, compute_dtype, use_double_quant: Requires bitsandbytes
#   - dtype, device_map: Requires compatible GPU memory (float16/bfloat16 on GPU)
#   - max_new_tokens per model: Must fit in GPU memory with batch size
#
# DO NOT CHANGE (unless you really know what you're doing):
#   - quant_type: "nf4" is optimal for 4-bit; "fp4" is alternative
#   - compute_dtype: "float16" recommended for most GPUs; "bfloat16" for Ampere+
#   - use_double_quant: Keep True for memory savings with 4-bit
#   - device_map: "auto" is recommended; manual mapping requires expertise
#   - dtype: Must match compute_dtype for quantization compatibility
#
# ============================================================================

# ============================================================================
# MODEL REGISTRY - Pre-configured model configurations
# ============================================================================
# WHAT TO CHANGE: 
#   - Add new models here by copying an existing entry and changing model_name
#   - model_name: Must be a valid Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-chat-hf")
#   - Adjust max_new_tokens per model based on your GPU memory
# 
# WHAT NOT TO CHANGE (without expertise):
#   - quant_type: "nf4" (NormalFloat4) is optimal for 4-bit quantization
#   - compute_dtype: "float16" works on most GPUs; "bfloat16" needs Ampere+
#   - use_double_quant: Keep True for additional memory savings with 4-bit
#   - device_map: "auto" automatically distributes across GPUs; manual mapping needs expertise
#   - dtype: Must match compute_dtype for quantization compatibility
#   - load_in_4bit: Set False only if you have enough VRAM for full precision
#   - quant_type/compute_dtype/use_double_quant: Changing these requires bitsandbytes expertise
#
# MEMORY CONSIDERATIONS (approximate VRAM for 4-bit quantized):
#   - 7B model: ~6GB VRAM | 8B model: ~7GB VRAM | 13B model: ~10GB VRAM
#   - Increase max_new_tokens = more KV cache = more VRAM per generation
# ============================================================================
MODEL_REGISTRY = {
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-8B",              # HF model ID - CHANGE to use different model
        "load_in_4bit": True,                        # 4-bit quantization (CHANGE only if >24GB VRAM)
        "quant_type": "nf4",                         # DO NOT CHANGE - nf4 is optimal for 4-bit
        "compute_dtype": "float16",                  # CHANGE to "bfloat16" only on Ampere+ GPUs
        "use_double_quant": True,                    # DO NOT CHANGE - saves ~0.5GB VRAM
        "dtype": "float16",                          # Must match compute_dtype
        "device_map": "auto",                        # DO NOT CHANGE unless manual GPU mapping
        "max_new_tokens": 256,                       # CHANGE: Higher = longer output, more VRAM
        "temperature": 0.2,                          # CHANGE: 0.0=deterministic, 1.0=creative
    },
    "phi-4": {
        "model_name": "microsoft/phi-4",             # CHANGE to use different Phi model
        "load_in_4bit": True,                        # CHANGE only if >24GB VRAM
        "quant_type": "nf4",                         # DO NOT CHANGE
        "compute_dtype": "float16",                  # CHANGE to "bfloat16" only on Ampere+
        "use_double_quant": True,                    # DO NOT CHANGE
        "dtype": "float16",                          # Must match compute_dtype
        "device_map": "auto",                        # DO NOT CHANGE
        "max_new_tokens": 256,                       # CHANGE based on VRAM
        "temperature": 0.2,                          # CHANGE: 0.0-1.0
    },
    "llama3-8b": {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",  # CHANGE for different Llama variant
        "load_in_4bit": True,                        # CHANGE only if >24GB VRAM
        "quant_type": "nf4",                         # DO NOT CHANGE
        "compute_dtype": "float16",                  # CHANGE to "bfloat16" on Ampere+
        "use_double_quant": True,                    # DO NOT CHANGE
        "dtype": "float16",                          # Must match compute_dtype
        "device_map": "auto",                        # DO NOT CHANGE
        "max_new_tokens": 256,                       # CHANGE based on VRAM
        "temperature": 0.2,                          # CHANGE: 0.0-1.0
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",  # CHANGE for different Mistral
        "load_in_4bit": True,                        # CHANGE only if >24GB VRAM
        "quant_type": "nf4",                         # DO NOT CHANGE
        "compute_dtype": "float16",                  # CHANGE to "bfloat16" on Ampere+
        "use_double_quant": True,                    # DO NOT CHANGE
        "dtype": "float16",                          # Must match compute_dtype
        "device_map": "auto",                        # DO NOT CHANGE
        "max_new_tokens": 256,                       # CHANGE based on VRAM
        "temperature": 0.2,                          # CHANGE: 0.0-1.0
    },
}

# ============================================================================
# DEFAULT MODEL SELECTION
# ============================================================================
# WHAT TO CHANGE: Set to one of the keys in MODEL_REGISTRY above
#   Options: "qwen3-8b", "phi-4", "llama3-8b", "mistral-7b"
# WHAT NOT TO CHANGE: Must match a key in MODEL_REGISTRY exactly
# ============================================================================
DEFAULT_MODEL = "qwen3-8b"

# ============================================================================
# GENERATION PARAMETERS (used as defaults, can be overridden per-experiment)
# ============================================================================
# WHAT TO CHANGE:
#   - MAX_NEW_TOKENS: Increase for longer generations (costs more VRAM/time)
#                     Typical: 128-512 for prompts, 1024+ for long-form
#   - TEMPERATURE: 0.0 = deterministic, 0.7 = balanced, 1.0 = creative
#                   Lower for factual tasks, higher for creative generation
#
# WHAT NOT TO CHANGE: These are runtime defaults; override per-experiment instead
# ============================================================================
MAX_NEW_TOKENS = 500      # CHANGE: 128-2048 depending on task & VRAM
TEMPERATURE = 0.2         # CHANGE: 0.0 (deterministic) to 1.0 (creative)

# ============================================================================
# GENETIC ALGORITHM (GA) PARAMETERS
# ============================================================================
# WHAT TO CHANGE (tune for your task):
#   - GA_POPULATION_SIZE: Larger = more diversity, slower per generation
#       Recommended: 10-30 (default 15). Increase for complex search spaces.
#   - GA_GENERATIONS: More generations = better optimization, more compute
#       Recommended: 5-20 (default 5). Increase for difficult optimization.
#   - GA_MUTATION_RATE: Probability of mutation per gene (0.0-1.0)
#       Recommended: 0.1-0.5 (default 0.3). Higher = more exploration.
#   - GA_CROSSOVER_RATE: Probability of crossover (0.0-1.0)
#       Recommended: 0.5-0.9 (default 0.7). Higher = more recombination.
#   - GA_TOURNAMENT_SIZE: Tournament selection size (2+)
#       Recommended: 2-5 (default 3). Larger = more selective pressure.
#   - EVALUATION_SUBSET_SIZE: Number of samples for fitness evaluation
#       Recommended: 50-200 (default 100). Larger = better fitness estimate, slower.
#   - RANDOM_SEED: Set for reproducibility (default 42). Change for different runs.
#
# WHAT TO CHANGE WITH CAUTION:
#   - GA_TOURNAMENT_SIZE > POPULATION_SIZE/2 reduces diversity significantly
#   - GA_MUTATION_RATE > 0.5 may prevent convergence
#   - GA_CROSSOVER_RATE < 0.3 reduces recombination benefits
#   - EVALUATION_SUBSET_SIZE too small (<20) gives noisy fitness estimates
#
# ADAPTIVE MUTATION PARAMETERS (only used if USE_ADAPTIVE_MUTATION=True):
#   - USE_ADAPTIVE_MUTATION: Enable dynamic mutation rate adaptation (True/False)
#       Enable for complex landscapes; disable for simple/well-understood spaces
#   - MUTATION_EVOLUTION_INTERVAL: Generations between mutation rate updates
#       Recommended: 3-10 (default 5). Smaller = faster adaptation, more variance
#   - MUTATION_TOURNAMENT_SIZE: Tournament size for selecting mutation parents
#       Recommended: 2-5 (default 3). Independent of GA_TOURNAMENT_SIZE
#   - MUTATION_ELITE_COUNT: Number of top mutations to preserve
#       Recommended: 1-5 (default 2). Too high = premature convergence
#   - MIN_CHILDREN_FOR_RANKING: Minimum offspring before ranking mutations
#       Recommended: 5-20 (default 10). Too low = noisy ranking
#   - TOP_CHILDREN_TO_KEEP: Top offspring retained per generation
#       Recommended: 3-10 (default 5). Too high = reduced diversity
#
# WHAT NOT TO CHANGE WITHOUT DEEP GA KNOWLEDGE:
#   - Adaptive mutation internal mechanics (selection, ranking algorithms)
#   - Tournament selection implementation details
# ============================================================================
GA_POPULATION_SIZE = 10
GA_GENERATIONS = 5
GA_MUTATION_RATE = 0.6
GA_CROSSOVER_RATE = 0.5
GA_TOURNAMENT_SIZE = 3
EVALUATION_SUBSET_SIZE = 100
RANDOM_SEED = 50

# Adaptive Mutation Configuration
USE_ADAPTIVE_MUTATION = True
MUTATION_EVOLUTION_INTERVAL = 5
MUTATION_TOURNAMENT_SIZE = 3
MUTATION_ELITE_COUNT = 2
MIN_CHILDREN_FOR_RANKING = 10
TOP_CHILDREN_TO_KEEP = 5