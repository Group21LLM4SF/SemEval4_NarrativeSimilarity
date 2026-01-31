#!/usr/bin/env python3
"""Data augmentation script - generates augmented triplets with aspects"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.dataset import load_aspect_triplets, AspectTriplet
from src.llm.augmentor import AugmentTripletGenerator, NewTriplet
from src.llm.aspects_extractor import AspectsExtractor

import argparse


def convert_triplet_to_dict(triplet):
    """Convert AspectTriplet to dictionary format for augmentation"""
    return {
        "triplet_id": triplet.triplet_id,
        "anchor_text": triplet.anchor,
        "similar": triplet.positive,
        "dissimilar": triplet.negative
    }


def augment_response_to_triplets(response: NewTriplet, index: int) -> AspectTriplet:
    """Convert augmented response to AspectTriplet"""
    return AspectTriplet(
        triplet_id=f"{response.triplet_id}_{index}",
        anchor=response.anchor,
        positive=response.similar,
        negative=response.dissimilar,
        anchor_aspects=None,
        positive_aspects=None,
        negative_aspects=None
    )


def save_aspect_triplets(triplets, filepath):
    """Save triplets with aspects to jsonl file"""
    with open(filepath, 'w') as f:
        for triplet in triplets:
            try:
                json_line = {
                    'triplet_id': triplet.triplet_id,
                    'anchor': triplet.anchor,
                    'positive': triplet.positive,
                    'negative': triplet.negative,
                    # theme
                    'anchor_theme': triplet.anchor_aspects.abstract_theme,
                    'positive_theme': triplet.positive_aspects.abstract_theme,
                    'negative_theme': triplet.negative_aspects.abstract_theme,
                    # course_of_action
                    'anchor_action': triplet.anchor_aspects.course_of_action,
                    'positive_action': triplet.positive_aspects.course_of_action,
                    'negative_action': triplet.negative_aspects.course_of_action,
                    # outcomes
                    'anchor_outcome': triplet.anchor_aspects.outcomes,
                    'positive_outcome': triplet.positive_aspects.outcomes,
                    'negative_outcome': triplet.negative_aspects.outcomes
                }
                f.write(json.dumps(json_line) + '\n')
            except Exception:
                continue


def main(api_key: str):
    # Load data
    training_data = load_aspect_triplets("data/processed/train_from_dev_w_aspect_aug_triplets.jsonl")

    # Filter for original dev samples (no "_" in triplet_id)
    from_dev = [t for t in training_data if t.triplet_id.find("_") == -1]
    print(f"Loaded {len(from_dev)} original dev samples")

    # Convert to dict format
    from_dev_dicts = [convert_triplet_to_dict(t) for t in from_dev]

    # Load API key
    # with open('data/auth/together.ai.key') as f:
    #     api_key = f.read().strip()

    # Step 1: Generate augmented triplets
    print("Generating augmented triplets...")
    augmentor = AugmentTripletGenerator(api_key=api_key)
    llama_data_aug = augmentor.run_batch(
        from_dev_dicts,
        temps=[0.5, 0.8, 1.0],
        n_triplets_per_sample=3,
        batch_input_path="data/generated/inputs-prompts/llama_augment_from_partial_dev_large.jsonl",
        batch_output_path="data/generated/llama_augment_from_partial_dev_large.jsonl"
    )

    # Convert to AspectTriplet format
    augmented_triplets = [augment_response_to_triplets(t, idx)
                          for idx, t in enumerate(llama_data_aug)]
    print(f"Generated {len(augmented_triplets)} augmented triplets")

    # Step 2: Extract aspects
    print("Extracting aspects...")
    extractor = AspectsExtractor(api_key=api_key)
    aspects_only_response = extractor.run_batch(
        augmented_triplets,
        batch_input_path="data/generated/inputs-prompts/llama_augment_from_partial_dev_large_aspects_input.jsonl",
        batch_output_path="data/generated/llama_augment_from_partial_dev_large_aspects_output.jsonl"
    )

    # Step 3: Match aspects to triplets
    print("Matching aspects to triplets...")
    for asp in aspects_only_response:
        for triplet in augmented_triplets:
            if asp.triplet_id == triplet.triplet_id:
                triplet.anchor_aspects = asp.anchor
                triplet.positive_aspects = asp.positive
                triplet.negative_aspects = asp.negative
                break

    # Step 4: Save results
    output_path = "data/processed/llama_augment_from_partial_dev_large_with_aspects.jsonl"
    save_aspect_triplets(augmented_triplets, output_path)
    print(f"Saved {len(augmented_triplets)} triplets with aspects to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA/Qwen Augmentor Script")
    parser.add_argument('--api_key', type=str, required=True, help='API key for LLM services')
    args = parser.parse_args()
    main(args.api_key)
