"""Augmentation-based Triplet Generator using Together.ai batch API."""

import json
import time
from pathlib import Path

from together import Together
from loguru import logger
from tqdm import tqdm
import random 


from .prompts.augmentation import SYSTEM_PROMPT_AUGMENT, format_augment_prompt, NewTriplet, AugmentedResponse

# ==============================================================================
# GENERATOR CLASS
# ==============================================================================
class AugmentTripletGenerator:
    """Generate augmented triplets from dev data using Together.ai batch API."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    ):
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized with {model_name}")
    
    def _build_messages(self, row: dict, n_triplets: int = 2) -> list[dict]:   
        user_content = format_augment_prompt(row, n_triplets)
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT_AUGMENT},
            {"role": "user", "content": user_content}
        ]
        
    
    def create_batch_jsonl(
        self,
        dev_data: list[dict],
        temps: list[float] = [0.5, 0.8, 1.0],
        n_triplets_per_sample: int = 2,
        output_path: str = "augment_batch_requests.jsonl"
    ) -> Path:
        """Create batch JSONL from dev data."""
        logger.info(f"Creating requests from {len(dev_data) * len(temps)} dev samples...")
        
        output_file = Path(output_path)

        with open(output_file, "w") as f:
            for _ in range(len(temps)):
                for idx, row in enumerate(tqdm(dev_data, desc="Building batch")):
                    triplet_id = row['triplet_id']
                    request = {
                        "custom_id": f"augment-{idx}-{triplet_id}",
                        "body": {
                            "model": self.model_name,
                            "messages": self._build_messages(row, n_triplets_per_sample),
                            "temperature": random.choice(temps),
                            "response_format": {"type": "json_object"}
                        }
                    }
                    f.write(json.dumps(request) + "\n")
        
        logger.success(f"Created: {output_file} ({len(dev_data) * len(temps)} requests)")
        return output_file
    
    @staticmethod
    def parse_batch_output(output_jsonl_path: str) -> list[NewTriplet]:
        """Parse batch output JSONL, flatten all triplets."""
        logger.info(f"Parsing {output_jsonl_path}")
        results = []
        failed = 0
        with open(output_jsonl_path, "r") as f:
            for line in f:
                result = json.loads(line)
                custom_id = result.get("custom_id", "")
                original_triplet_id = custom_id.split("-")[-1] if custom_id else None
                try:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    payload = json.loads(content)  # dict from model JSON
                    if isinstance(payload, dict):

                        for triplet in payload.get("triplets", []):
                            triplet["original_triplet_id"] = original_triplet_id

                        response = AugmentedResponse.model_validate_json(json.dumps(payload))
                        results.extend(response.triplets)
                    else:
                        failed += 1
                        logger.warning(f"Failed {result['custom_id']}: Payload is not a dict")
                except Exception as e:
                    failed += 1
                    logger.warning(f"Failed {result['custom_id']}: {e}")
        logger.success(f"Parsed {len(results)} triplets ({failed} failed requests)")
        return results
    
    def run_upload(
        self,
        batch_input_path: str = "augment_batch_requests.jsonl",
        interval: int = 30
    ) -> str:
        """Upload batch input file and return file ID."""

        try:
            logger.info("Uploading...")
            file_response = self.client.files.upload(
                file=batch_input_path, purpose="batch-api", check=False
            )
            logger.success(f"Uploaded: {file_response.id}")
            
            batch = self.client.batches.create_batch(
                file_response.id, endpoint="/v1/chat/completions"
            )
            logger.success(f"Batch created: {batch.id}")
    
            with tqdm(desc="Processing") as pbar:
                while batch.status in ["VALIDATING", "IN_PROGRESS"]:
                    time.sleep(interval)
                    batch = self.client.batches.get_batch(batch.id)
                    pbar.set_postfix({"status": batch.status})
    
            if batch.status == "COMPLETED":
                logger.success("Completed!")
                return batch.output_file_id
            else:
                raise RuntimeError(f"Failed: {batch.status}")
        except KeyboardInterrupt:
            if batch:
                self.client.batches.cancel_batch(batch.id)
                logger.warning(f"Cancelled {batch.id}")
            raise

    def download_output(
        self,
        output_file_id: str,
        output_path: str = "augment_batch_output.jsonl"):

        """Download batch output file."""
        logger.info("Downloading output...")
        self.client.files.retrieve_content(
            id=output_file_id, output=output_path
        )
        logger.success(f"Downloaded to {output_path}")
        
    def cancel(self, batch_id: str):
        """Cancel a running batch."""
        self.client.batches.cancel_batch(batch_id)
        logger.warning(f"Cancelled {batch_id}")

    def run_batch(
        self,
        dev_data: list[dict],
        temps: list[float] = [0.5, 0.8, 1.0],
        n_triplets_per_sample: int = 2,
        batch_input_path: str = "augment_batch_requests.jsonl",
        batch_output_path: str = "augment_batch_output.jsonl",
        poll_interval: int = 30
    ) -> list[NewTriplet]:
        """Run full batch pipeline."""
        batch = None
        
        if Path(batch_output_path).exists():
            logger.info(f"Output {batch_output_path} exists.")
            if input("Use existing? (y/n): ").lower() == 'y':
                return self.parse_batch_output(batch_output_path)
        
        if Path(batch_input_path).exists():
            logger.info(f"Using existing: {batch_input_path}")
            batch_path = Path(batch_input_path)
        else:
            batch_path = self.create_batch_jsonl(
                dev_data, temps, n_triplets_per_sample, batch_input_path, 
            )
        
        output_file_id = self.run_upload(
            batch_input_path=batch_path, interval=poll_interval
        )
        
        self.download_output(
            output_file_id=output_file_id, output_path=batch_output_path
        )

        return self.parse_batch_output(batch_output_path)

    def save_triplets(self, triplets: list[NewTriplet], output_path: str) -> Path:
        output_file = Path(output_path)
        with open(output_file, "w") as f:
            for t in triplets:
                f.write(t.model_dump_json() + "\n")
        logger.success(f"Saved {len(triplets)} to {output_file}")
        return output_file
    

