# Contamination Baseline Audit

Total prompts analyzed: 109

## Summary by Category

| Category | Count | % | Crossover | Mutation | Unknown |
|----------|-------|---|-----------|----------|---------|
| clean | 55 | 50.5% | 16 | 39 | 0 |
| meta_commentary | 53 | 48.6% | 53 | 0 | 0 |
| vignette_collapse | 1 | 0.9% | 0 | 1 | 0 |
| duplicate | 0 | 0.0% | 0 | 0 | 0 |
| other | 0 | 0.0% | 0 | 0 | 0 |

## Summary by Origin

| Origin | Total | Clean | Meta | Vignette | Other |
|--------|-------|-------|------|----------|-------|
| crossover | 69 | 16 | 53 | 0 | 0 |
| mutation | 40 | 39 | 0 | 1 | 0 |
| unknown | 0 | 0 | 0 | 0 | 0 |

## Examples

### Meta Commentary

- **generated_prompts_1.jsonl** (crossover): Here is the offspring prompt:

"Consider the patient's medical history and current symptoms. As a meticulous clinician, carefully weigh the evidence and eliminate incorrect options before selecting th...
  - Markers: ['meta:here is the offspring', 'meta:this prompt combines', 'meta:parent a', 'meta:parent b', 'meta:offspring prompt', 'meta:combines the', 'meta:from parent', 'meta:the offspring']
- **generated_prompts_1.jsonl** (crossover): Here is the offspring prompt:

"Imagine yourself as a seasoned medical detective, tasked with solving a complex case. Apply your expertise and analytical mind to carefully evaluate each option, weighi...
  - Markers: ['meta:here is the offspring', 'meta:this prompt combines', 'meta:parent a', 'meta:parent b', 'meta:offspring prompt', 'meta:combines the', 'meta:the offspring']
- **generated_prompts_1.jsonl** (crossover): Here is the offspring prompt:

As a meticulous medical detective, scrutinize each option carefully, considering the most plausible explanation, and select the single most accurate answer.

This prompt...
  - Markers: ['meta:here is the offspring', 'meta:this prompt combines', 'meta:parent a', 'meta:parent b', 'meta:offspring prompt', 'meta:combines the', 'meta:the offspring']
- **generated_prompts_1.jsonl** (crossover): Here is the offspring prompt:

As a seasoned medical expert, critically evaluate each option and choose the most plausible diagnosis, considering the patient's symptoms, medical history, and laborator...
  - Markers: ['meta:here is the offspring', 'meta:this prompt combines', 'meta:parent a', 'meta:parent b', 'meta:offspring prompt', 'meta:combines the', 'meta:from parent', 'meta:the offspring']
- **generated_prompts_1.jsonl** (crossover): Here is the offspring prompt:

You are a meticulous researcher. Delve into the medical literature, scrutinize each option, and provide a definitive answer based on the most reliable evidence.

This pr...
  - Markers: ['meta:here is the offspring', 'meta:this prompt combines', 'meta:parent a', 'meta:parent b', 'meta:offspring prompt', 'meta:combines the', 'meta:the offspring']

### Vignette Collapse

- **generated_prompts.jsonl** (mutation): What is the most likely diagnosis or treatment plan for a patient presenting with a 2-day history of worsening headache, fever, and stiff neck, with a recent history of travel to an area with a high i...
  - Markers: ['vignette:presenting with', 'vignette:history of']

### Other


