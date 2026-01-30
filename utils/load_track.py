"""Data loading utilities for Track A and Track B."""

from pydantic import BaseModel, computed_field
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from loguru import logger
from pydantic import TypeAdapter, ValidationError


"""Pydantic data models for Track A and Track B records."""

def stable_hash_int(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")  # 64-bit int from first 8 bytes


class TrackARecord(BaseModel):
    """Single triplet record for Track A.

    Attributes:
        anchor_text: The anchor story text
        text_a: First candidate story text
        text_b: Second candidate story text
        text_a_is_closer: Whether text_a is more similar to anchor (None for test data)
        model_name: Optional model name field (can be ignored at this level)
        triplet_id: Optional[str] = None Unique identifier for the triplet
    """

    anchor_text: str
    text_a: str
    text_b: str
    text_a_is_closer: bool | None = None
    model_name: str | None = None

    @computed_field
    @property
    def label(self) -> str | None:
        """Convert boolean to 'A'/'B' label.

        Returns:
            'A' if text_a_is_closer is True, 'B' if False, None if unset
        """
        if self.text_a_is_closer is None:
            return None
        return "A" if self.text_a_is_closer else "B"
    
    @computed_field
    @property
    def triplet_id(self) -> str | None:
        return str(stable_hash_int(f"{self.anchor_text}-{self.text_a}-{self.text_b}"))


class TrackBRecord(BaseModel):
    """Single story record for Track B.

    Attributes:
        text: The story text to be encoded
    """
    text: str

    @computed_field
    @property
    def text_id(self) -> str | None:
        return str(stable_hash_int(self.text))


def _load_and_validate_jsonl(
    filepath: str | Path,
    record_type: type[TrackARecord] | type[TrackBRecord],
    track_name: str,
) -> list[Any]:
    """Common logic for loading and validating JSONL files.

    Args:
        filepath: Path to JSONL file
        record_type: Pydantic model class for validation (TrackARecord or TrackBRecord)
        track_name: Name of track for logging (e.g., "Track A")

    Returns:
        List of validated record instances

    Raises:
        FileNotFoundError: If filepath does not exist
        json.JSONDecodeError: If JSON parsing fails
    """
    filepath = Path(filepath)
    logger.info(f"Loading {track_name} data from {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Read JSONL file
    records: list[dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON at line {line_num}: {e}")
                raise

    # Validate with Pydantic - skip invalid records
    adapter = TypeAdapter(record_type)
    validated_records: list[Any] = []
    skipped_count = 0

    for idx, record in enumerate(records):
        try:
            validated = adapter.validate_python(record)
            validated_records.append(validated)
        except ValidationError as e:
            skipped_count += 1
            logger.warning(
                f"Skipping invalid record at index {idx}: {e.error_count()} validation error(s). "
                f"First error: {e.errors()[0]['msg']}"
            )

    logger.info(
        f"Loaded {len(validated_records)} valid {track_name} records "
        f"({skipped_count} skipped due to validation errors)"
    )

    return validated_records


def load_track_a(filepath: str | Path) -> Dataset:
    """Load Track A data from JSONL.

    Args:
        filepath: Path to JSONL file containing Track A records

    Returns:
        HuggingFace Dataset with columns:
        - anchor_text: str
        - text_a: str
        - text_b: str
        - text_a_is_closer: bool | None
        - label: str | None ('A' or 'B')

    Raises:
        FileNotFoundError: If filepath does not exist

    Note:
        Records that fail validation are skipped and logged as warnings
    """
    # Load and validate records using common helper
    validated_records = _load_and_validate_jsonl(
        filepath=filepath,
        record_type=TrackARecord,
        track_name="Track A",
    )

    # Convert to HuggingFace Dataset
    dataset_dict = {
        "triplet_id": [r.triplet_id for r in validated_records],
        "anchor_text": [r.anchor_text for r in validated_records],
        "text_a": [r.text_a for r in validated_records],
        "text_b": [r.text_b for r in validated_records],
        "text_a_is_closer": [r.text_a_is_closer for r in validated_records],
        "label": [r.label for r in validated_records],
    }

    dataset = Dataset.from_dict(dataset_dict)
    logger.success(f"Created Dataset with {len(dataset)} examples")

    return dataset


def load_track_b(filepath: str | Path) -> Dataset:
    """Load Track B data from JSONL.

    Args:
        filepath: Path to JSONL file containing Track B records

    Returns:
        HuggingFace Dataset with columns:
        - text: str

    Raises:
        FileNotFoundError: If filepath does not exist

    Note:
        Records that fail validation are skipped and logged as warnings
    """
    # Load and validate records using common helper
    validated_records = _load_and_validate_jsonl(
        filepath=filepath,
        record_type=TrackBRecord,
        track_name="Track B",
    )

    # Convert to HuggingFace Dataset
    dataset_dict = {
        "text_id": [r.text_id for r in validated_records],
        "text": [r.text for r in validated_records],
    }

    dataset = Dataset.from_dict(dataset_dict)
    logger.success(f"Created Dataset with {len(dataset)} examples")

    return dataset

