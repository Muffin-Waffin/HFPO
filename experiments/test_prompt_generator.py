from src.core.prompt_candidate import PromptCandidate
from src.evolution.prompt_generator.cleaner import PromptCleaner
from src.evolution.prompt_generator.generator import PromptGenerator
from src.evolution.prompt_generator.models import PromptGenerationRequest
from src.evolution.prompt_generator.templates import PromptTemplateBuilder
from src.evolution.prompt_generator.validator import PromptValidator


class FakeLLM:
    """Simple fake reasoning model."""

    def generate(self, prompt: str, temperature: float) -> str:
        return """
# Prompt

"You are a careful medical reasoning assistant.
Always reason carefully before selecting the final answer."
"""


class FakeLineageTracker:
    """Simple fake lineage tracker."""

    def build_ancestry(self, parent_ids: list[str]) -> list[str]:
        return parent_ids.copy()


parent = PromptCandidate(
    id="parent_1",
    text="You are a helpful medical assistant. Think carefully before answering.",
    generation=0,
    parent_ids=[],
    ancestry_ids=[],
    origin="seed",
)

request = PromptGenerationRequest(
    parent_a=parent,
    parent_b=None,
    generation=1,
    task_description="Answer multiple-choice medical questions.",
    temperature=0.7,
    existing_prompt_texts={parent.text},
)

generator = PromptGenerator(
    llm=FakeLLM(),
    template_builder=PromptTemplateBuilder(),
    cleaner=PromptCleaner(),
    validator=PromptValidator(),
    lineage_tracker=FakeLineageTracker(),
)

result = generator.generate(request)

print("=" * 60)
print("Generation Result")
print("=" * 60)

print(result)

print("\nCandidate")
print(result.candidate)

print("\nPrompt Text")
print(result.candidate.text)

print("\nMetadata")
print(result.metadata)

print("\nCandidate Metadata")
print(result.candidate.metadata)