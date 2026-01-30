"""Prompts for LLM-based narrative similarity model."""

from pydantic import BaseModel, Field, computed_field

SYSTEM_PROMPT = """You are a narrative analysis expert. For each story, identify:
1. Abstract Theme: core problems, ideas, motifs (NOT concrete setting)
2. Course of Action: event sequence and order
3. Outcomes: how the story ends

IGNORE: writing style, setting details, character names, text length."""


EXAMPLE_PROMPT = """Example:

Story:
Anna loses her purse. She is terrified because there are important documents in it. She retraces her steps but cannot find it. Dan finds it and helpfully returns it to her.

Answer (return JSON with these exact keys):
{
  "abstract_theme": "Loss of a valuable item causing distress.",
  "course_of_action": "Item is lost, owner searches unsuccessfully, stranger finds and returns it.",
  "outcomes": "The lost item is recovered through external help."
}"""


class Aspects(BaseModel):
    """Aspects of narrative similarity."""
    abstract_theme: str = Field(...)
    course_of_action: str = Field(...)
    outcomes: str = Field(...)
    

class TripletAspects(BaseModel):
    """Aspects for a complete triplet."""
    triplet_id: str | None
    anchor: Aspects
    positive : Aspects
    negative : Aspects
    
def get_aspects_extraction_prompt(story: str) -> str:
    """Create prompt to extract aspects from a story."""
    return f"Now analyze this story:\n\nStory:\n{story}\n\nOutput:"