#!/usr/bin/env python3
"""Phase 0: Trace the generation pipeline to verify wiring.

Instantiates PromptGenerator exactly as run_evolution.py does,
feeds it contaminated parent pairs, and prints output at each stage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path BEFORE imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.evolution.prompt_generator.generator import PromptGenerator
from src.evolution.prompt_generator.templates import PromptTemplateBuilder
from src.evolution.prompt_generator.cleaner import PromptCleaner
from src.evolution.prompt_generator.validator import PromptValidator
from src.evolution.prompt_generator.models import PromptGenerationRequest, ParentPerformance
from src.core.prompt_candidate import PromptCandidate


def load_contaminated_parents(jsonl_path: Path, max_pairs: int = 5) -> list[tuple[PromptCandidate, PromptCandidate | None, str]]:
    """Load contaminated parent pairs from generated_prompts.jsonl."""
    pairs = []
    with open(jsonl_path) as f:
        for line in f:
            if len(pairs) >= max_pairs:
                break
            record = json.loads(line)
            if record.get("origin") != "crossover":
                continue
            
            # Check for contamination markers in generated text
            generated_text = record.get("generated", {}).get("text", "").lower()
            if not any(marker in generated_text for marker in [
                "here is the offspring", "this prompt combines", 
                "parent a", "parent b", "offspring prompt"
            ]):
                continue
            
            parent_a_data = record.get("parent_a", {})
            parent_b_data = record.get("parent_b", {})
            
            parent_a = PromptCandidate(
                id=parent_a_data.get("id", "parent_a"),
                text=parent_a_data.get("text", ""),
                generation=record["generation"] - 1,
                parent_ids=[],
                ancestry_ids=[],
                origin=parent_a_data.get("origin", "seed"),
            )
            
            parent_b = None
            if parent_b_data:
                parent_b = PromptCandidate(
                    id=parent_b_data.get("id", "parent_b"),
                    text=parent_b_data.get("text", ""),
                    generation=record["generation"] - 1,
                    parent_ids=[],
                    ancestry_ids=[],
                    origin=parent_b_data.get("origin", "seed"),
                )
            
            pairs.append((parent_a, parent_b, record["generated"]["text"]))
    
    return pairs


def trace_pipeline():
    print("=" * 80)
    print("PHASE 0: PIPELINE TRACE")
    print("=" * 80)
    
    # 1. Load contaminated examples from current run
    jsonl_path = Path("results/hfpo_run/generated_prompts.jsonl")
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found")
        return 1
    
    pairs = load_contaminated_parents(jsonl_path, max_pairs=3)
    print(f"\nLoaded {len(pairs)} contaminated crossover examples\n")
    
    # 2. Instantiate PromptGenerator exactly as run_evolution.py does
    print("Instantiating PromptGenerator components...")
    template_builder = PromptTemplateBuilder()
    cleaner = PromptCleaner()
    validator = PromptValidator()
    llm = _create_llm()  # Uses the same LLM wrapper as run_evolution.py
    
    generator = PromptGenerator(
        llm=llm,
        template_builder=template_builder,
        cleaner=cleaner,
        validator=validator,
    )
    print("✓ PromptGenerator instantiated\n")
    
    # 3. Trace each pair through the pipeline
    for i, (parent_a, parent_b, original_generated) in enumerate(pairs):
        print(f"{'='*80}")
        print(f"PAIR {i+1}/{len(pairs)}")
        print(f"{'='*80}")
        print(f"\nPARENT A: {parent_a.text[:120]}...")
        print(f"PARENT B: {parent_b.text[:120] if parent_b else 'N/A'}...")
        print(f"\nORIGINAL GENERATED (contaminated): {original_generated[:200]}...")
        
        # Build request
        request = PromptGenerationRequest(
            parent_a=parent_a,
            parent_b=parent_b,
            generation=1,
            task_description="Answer medical multiple-choice questions accurately.",
            temperature=0.2,
            existing_prompt_texts=set(),
            parent_a_performance=None,
            parent_b_performance=None,
            mutation_operator=None,
        )
        
        # Trace: raw LLM output
        print("\n" + "-" * 40)
        print("STAGE 1: Raw LLM Output")
        print("-" * 40)
        template = template_builder.build(request)
        raw_output = llm.generate(template)
        print(raw_output[:500] + ("..." if len(raw_output) > 500 else ""))
        
        # Trace: cleaned output
        print("\n" + "-" * 40)
        print("STAGE 2: After PromptCleaner")
        print("-" * 40)
        clean_text = cleaner.clean(raw_output)
        print(clean_text[:500] + ("..." if len(clean_text) > 500 else ""))
        
        # Trace: validation
        print("\n" + "-" * 40)
        print("STAGE 3: PromptValidator")
        print("-" * 40)
        try:
            validator.validate(clean_text)
            print("✓ VALIDATION PASSED")
        except ValueError as e:
            print(f"✗ VALIDATION FAILED: {e}")
        
        # Trace: full generator.generate()
        print("\n" + "-" * 40)
        print("STAGE 4: Full generator.generate()")
        print("-" * 40)
        try:
            result = generator.generate(request)
            print(f"✓ GENERATED: {result.candidate.text[:200]}...")
        except ValueError as e:
            print(f"✗ GENERATION FAILED: {e}")


def _create_llm():
    """Create the same LLM wrapper used in run_evolution.py."""
    from experiments.run_evolution import _QwenReasoningLLM
    from src.llms.loader import load_model
    import configs.config as config
    
    model, tokenizer = load_model(config.DEFAULT_MODEL)
    return _QwenReasoningLLM(model, tokenizer, config.MAX_NEW_TOKENS)


if __name__ == "__main__":
    sys.exit(trace_pipeline())