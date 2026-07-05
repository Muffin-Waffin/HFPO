#!/usr/bin/env python3
"""Phase 1: Quantify baseline contamination across all historical runs.

Reads all generated_prompts.jsonl files under results/ and classifies each
generated prompt by contamination type.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal


@dataclass
class Classification:
    category: Literal["clean", "meta_commentary", "vignette_collapse", "duplicate", "other"]
    origin: Literal["crossover", "mutation", "unknown"]
    markers: list[str]


META_MARKERS = [
    "here is the offspring",
    "this prompt combines",
    "parent a",
    "parent b",
    "offspring prompt",
    "combines the",
    "from parent",
    "inherits from",
    "the offspring",
    "this offspring",
    "the resulting prompt",
    "resulting prompt",
]

VIGNETTE_MARKERS = [
    "presenting with",
    "year-old",
    "year old",
    "male with",
    "female with",
    "patient with",
    "history of",
    "complaining of",
    "presents with",
    "diagnosis of",
    "symptoms include",
]


def classify_prompt(text: str, origin: str) -> Classification:
    """Classify a generated prompt by contamination type."""
    text_lower = text.lower()
    markers = []

    # Check meta-commentary markers
    for marker in META_MARKERS:
        if marker in text_lower:
            markers.append(f"meta:{marker}")

    # Check vignette collapse markers (heuristic: presenting with + age/gender + symptoms)
    vignette_score = 0
    for marker in VIGNETTE_MARKERS:
        if marker in text_lower:
            vignette_score += 1
            markers.append(f"vignette:{marker}")

    if markers:
        # Determine primary category
        meta_markers = [m for m in markers if m.startswith("meta:")]
        vignette_markers = [m for m in markers if m.startswith("vignette:")]
        
        if meta_markers:
            return Classification("meta_commentary", origin, markers)
        elif len(vignette_markers) >= 2:  # Heuristic threshold
            return Classification("vignette_collapse", origin, markers)
        else:
            return Classification("other", origin, markers)

    return Classification("clean", origin, [])


def load_jsonl(filepath: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_all_jsonl_files(root: Path) -> list[Path]:
    """Find all generated_prompts.jsonlpts*.jsonl files."""
    return list(root.rglob("generated_prompts*.jsonl"))


def main():
    print("=" * 80)
    print("PHASE 1: BASELINE CONTAMINATION AUDIT")
    print("=" * 80)

    # Find all JSONL files
    jsonl_files = find_all_jsonl_files(Path("results"))
    print(f"\nFound {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  {f}")

    # Aggregate statistics
    total_by_category = Counter()
    total_by_origin = Counter()
    by_category_origin = defaultdict(Counter)
    all_records = []

    for filepath in jsonl_files:
        print(f"\nProcessing {filepath}...")
        records = load_jsonl(filepath)
        print(f"  Loaded {len(records)} records")

        for record in records:
            origin = record.get("origin", "unknown")
            text = record.get("generated", {}).get("text", "")
            
            classification = classify_prompt(text, origin)
            total_by_category[classification.category] += 1
            total_by_origin[origin] += 1
            by_category_origin[classification.category][origin] += 1
            all_records.append((filepath.name, classification, text[:200]))

    # Print summary table
    print("\n" + "=" * 80)
    print("CONTAMINATION BASELINE SUMMARY")
    print("=" * 80)
    total = sum(total_by_category.values())
    print(f"\nTotal prompts analyzed: {total}")
    print(f"\n{'Category':<20} {'Count':>8} {'%':>6} | {'Crossover':>10} {'Mutation':>10} {'Unknown':>8}")
    print("-" * 70)
    for category in ["clean", "meta_commentary", "vignette_collapse", "duplicate", "other"]:
        count = total_by_category[category]
        pct = (count / total * 100) if total > 0 else 0
        co = by_category_origin[category].get("crossover", 0)
        mu = by_category_origin[category].get("mutation", 0)
        un = by_category_origin[category].get("unknown", 0)
        print(f"{category:<20} {count:>8} {pct:>5.1f}% | {co:>10} {mu:>10} {un:>8}")

    # Print by origin
    print(f"\n{'Origin':<12} {'Total':>8} {'Clean':>8} {'Meta':>8} {'Vignette':>10} {'Other':>8}")
    print("-" * 55)
    for origin in ["crossover", "mutation", "unknown"]:
        tot = total_by_origin[origin]
        cl = by_category_origin["clean"].get(origin, 0)
        me = by_category_origin["meta_commentary"].get(origin, 0)
        vi = by_category_origin["vignette_collapse"].get(origin, 0)
        ot = by_category_origin["other"].get(origin, 0)
        print(f"{origin:<12} {tot:>8} {cl:>8} {me:>8} {vi:>10} {ot:>8}")

    # Show examples of each contamination type
    print("\n" + "=" * 80)
    print("EXAMPLES BY CATEGORY")
    print("=" * 80)
    
    examples_shown = defaultdict(int)
    max_examples = 3
    
    for filename, classification, text in all_records:
        if examples_shown[classification.category] < max_examples:
            print(f"\n[{classification.category.upper()}] from {filename} ({classification.origin})")
            print(f"  Markers: {classification.markers[:5]}")
            print(f"  Text: {text}...")
            examples_shown[classification.category] += 1

    # Save detailed results as markdown
    output_path = Path("results/prev/contamination_baseline.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# Contamination Baseline Audit\n\n")
        f.write(f"Total prompts analyzed: {total}\n\n")
        f.write("## Summary by Category\n\n")
        f.write("| Category | Count | % | Crossover | Mutation | Unknown |\n")
        f.write("|----------|-------|---|-----------|----------|---------|\n")
        for category in ["clean", "meta_commentary", "vignette_collapse", "duplicate", "other"]:
            count = total_by_category[category]
            pct = (count / total * 100) if total > 0 else 0
            co = by_category_origin[category].get("crossover", 0)
            mu = by_category_origin[category].get("mutation", 0)
            un = by_category_origin[category].get("unknown", 0)
            f.write(f"| {category} | {count} | {pct:.1f}% | {co} | {mu} | {un} |\n")
        
        f.write("\n## Summary by Origin\n\n")
        f.write("| Origin | Total | Clean | Meta | Vignette | Other |\n")
        f.write("|--------|-------|-------|------|----------|-------|\n")
        for origin in ["crossover", "mutation", "unknown"]:
            tot = total_by_origin[origin]
            cl = by_category_origin["clean"].get(origin, 0)
            me = by_category_origin["meta_commentary"].get(origin, 0)
            vi = by_category_origin["vignette_collapse"].get(origin, 0)
            ot = by_category_origin["other"].get(origin, 0)
            f.write(f"| {origin} | {tot} | {cl} | {me} | {vi} | {ot} |\n")

        f.write("\n## Examples\n\n")
        for category in ["meta_commentary", "vignette_collapse", "other"]:
            f.write(f"### {category.replace('_', ' ').title()}\n\n")
            count = 0
            for filename, classification, text in all_records:
                if classification.category == category and count < 5:
                    f.write(f"- **{filename}** ({classification.origin}): {text}...\n")
                    f.write(f"  - Markers: {classification.markers}\n")
                    count += 1
            f.write("\n")

    print(f"\n✓ Detailed results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())