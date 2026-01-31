"""Prompts for augmentation-based triplet generation."""

from pydantic import BaseModel, computed_field

SYSTEM_PROMPT_AUGMENT = """You generate narrative triplet augmentations based on DEEP STRUCTURAL SIMILARITY.

### SIMILARITY DIMENSIONS
Stories are similar when they share:
1. **Abstract Theme**: Core ideas, motifs, moral lessons, central problems
2. **Course of Action**: Sequence of events, conflicts, turning points, causality chain
3. **Outcomes**: Final resolution, consequences, character fates

### GENERATION RULES
- CHANGE: Genre, setting, character names, entities, writing style, tone, specific details
- LENGTH: Match word count of each reference story (±20 words)

The reference is ONLY for understanding the RELATIONSHIP pattern, 
not for copying content.

### EXAMPLE
Reference: A diplomat convinces banks to adopt climate currency → Success
Valid augmentation: A scientist convinces governments to adopt pandemic protocol → Success
(Same: institutional persuasion theme, negotiation arc, positive systemic outcome)
(Different: climate→health, banks→governments, currency→protocol)"""

USER_PROMPT_AUGMENT = """Reference Triplet:

**ANCHOR ({anchor_len} words):**
{anchor}

**SIMILAR ({similar_len} words):**
{similar}

**DISSIMILAR ({dissimilar_len} words):**
{dissimilar}

---

Generate {n_triplets} NEW triplets where:
- Each new_anchor has the SAME theme, action sequence, and outcome pattern as reference anchor
- Each new_similar preserves the SAME similarity relationship to its anchor
- Each new_dissimilar preserves the SAME dissimilarity relationship to its anchor
- All stories use DIFFERENT: characters, names, settings, genres, writing styles
- Match lengths: anchor~{anchor_len}, similar~{similar_len}, dissimilar~{dissimilar_len} words

Output JSON:
{{
  "triplets": [
    {{
      "anchor": "...",
      "similar": "...",
      "dissimilar": "...",
      "similarity_reason": "one sentence: why similar matches anchor in theme/action/outcome",
      "dissimilarity_reason": "one sentence: why dissimilar differs from anchor"
    }},
    ...
  ]
}}"""


def format_augment_prompt(row: dict, n_triplets: int = 2) -> str:
    """Format the user prompt from a dev data row."""
    if row['text_a_is_closer']:
        similar = row['text_a']
        dissimilar = row['text_b']
    else:
        similar = row['text_b']
        dissimilar = row['text_a']
    
    return USER_PROMPT_AUGMENT.format(
        anchor=row['anchor_text'],
        anchor_len=len(row['anchor_text'].split()),
        similar=similar,
        similar_len=len(similar.split()),
        dissimilar=dissimilar,
        dissimilar_len=len(dissimilar.split()),
        n_triplets=n_triplets
    )

def format_augment_prompt_2(row: dict, n_triplets: int = 2) -> str:
    """Format the user prompt from a dev data row."""
    
    return USER_PROMPT_AUGMENT.format(
        anchor=row['anchor_text'],
        anchor_len=len(row['anchor_text'].split()),
        similar=row['similar'],
        similar_len=len(row['similar'].split()),
        dissimilar=row['dissimilar'],
        dissimilar_len=len(row['dissimilar'].split()),
        n_triplets=n_triplets
    )

class NewTriplet(BaseModel):
    triplet_id: str | None = None
    anchor: str
    similar: str
    dissimilar: str
    similarity_reason: str
    dissimilarity_reason: str

    @property
    @computed_field
    def original_triplet_id(self) -> str | None:
        if self.triplet_id and "_" in self.triplet_id:
            return self.triplet_id.split("_")[0]
        return None

class AugmentedResponse(BaseModel):
    triplets: list[NewTriplet]