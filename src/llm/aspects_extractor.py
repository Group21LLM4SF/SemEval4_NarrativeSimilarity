"""Criteria extraction using Together.ai batch API."""
import json
import time
from pathlib import Path

from together import Together
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from .prompts.aspects import Aspects, TripletAspects, SYSTEM_PROMPT, EXAMPLE_PROMPT, get_aspects_extraction_prompt

class AspectsExtractor:
    """Extract story aspects using Together.ai batch API."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        use_example: bool = True
    ):
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.use_example = use_example
        logger.info(f"Initialized AspectsExtractor with {model_name}")
    
    def _build_messages(self, story: str) -> list[dict]:
        """Build messages for single story extraction."""
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        if self.use_example:
            messages.append({"role": "user", "content": EXAMPLE_PROMPT})
        
        messages.append({
            "role": "user",
            "content": get_aspects_extraction_prompt(story)
        })
        
        return messages
    
    def dataset2batch_input_jsonl(
        self,
        dataset,
        output_path: str = "batch_aspects_requests.jsonl"
    ) -> Path:
        """Convert dataset to batch input JSONL format.
        
        Each triplet generates 3 separate requests (anchor, story_a, story_b).
        Custom IDs follow pattern: triplet-{idx}-{anchor|a|b}
        """
        logger.info(f"Converting {len(dataset)} triplets to batch format (3 requests each)...")
        
        output_file = Path(output_path)
        request_count = 0
        
        with open(output_file, "w") as f:
            for example in tqdm(dataset, desc="Building batch"):

                triplet_idx = example.triplet_id

                for text_key in ["anchor", "positive", "negative"]:
                    if text_key == "anchor":
                        story, part = example.anchor, "anchor"
                    elif text_key == "positive":
                        story, part = example.positive, "positive"
                    else:  # text_key == "negative"
                        story, part = example.negative, "negative"

                    request = {
                        "custom_id": f"triplet-{triplet_idx}-{part}",
                        "body": {
                            "model": self.model_name,
                            "messages": self._build_messages(story),
                            "temperature": 0.8,
                            "response_format": {"type": "json_object"}
                        }
                    }

                    f.write(json.dumps(request) + "\n")
                    request_count += 1
        
        logger.success(f"Created batch file: {output_file} ({request_count} requests)")
        return output_file
    
    
    def parse_batch_output(
        self,
        output_jsonl_path: str
    ) -> list[TripletAspects]:
        """Parse batch output JSONL to TripletAspects objects."""
        logger.info(f"Parsing batch output from {output_jsonl_path}")
        
        triplet_parts: dict[int, dict[str, Aspects]] = {}
        
        with open(output_jsonl_path, "r") as f:
            for line in f:
                result = json.loads(line)
                custom_id = result["custom_id"]
                
                # Parse: "triplet-{idx}-{part}"
                parts = custom_id.split("-")
                triplet_idx = str(parts[1])
                story_part = parts[2]
                
                # Extract and validate with Pydantic
                try:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    aspects = Aspects.model_validate_json(content)
                except Exception as e:
                    logger.warning(f"Failed to parse aspects for {custom_id}: {e}")
                    continue
                
                if triplet_idx not in triplet_parts:
                    triplet_parts[triplet_idx] = {}
                triplet_parts[triplet_idx][story_part] = aspects
        
        # Convert triplet_parts to list of TripletAspects
        results = []
        for triplet_idx, story_dict in triplet_parts.items():
            if not all(k in story_dict for k in ["anchor", "positive", "negative"]):
                # logger.warning(f"Incomplete data for triplet {triplet_idx}, skipping...")
                continue

            triplet = TripletAspects(
                triplet_id=triplet_idx,
                anchor=story_dict.get("anchor"),
                positive=story_dict.get("positive"),
                negative=story_dict.get("negative")
            )

            results.append(triplet)

        logger.success(f"Parsed {len(results)} complete triplets from output")
        
        return results
    
    def run_batch(
        self,
        dataset,
        batch_input_path: str = "batch_aspects_requests.jsonl",
        batch_output_path: str = "batch_aspects_output.jsonl",
        poll_interval: int = 30
    ) -> list[TripletAspects]:
        """Run full batch extraction pipeline."""
        batch = None
        try:

            if Path(batch_output_path).exists():
                logger.info(f"Batch output file {batch_output_path} already exists. Parsing directly...")
                # 
                user_choose = input(" Do you want to use the existing output file? (y/n): ")
                if user_choose.lower() == 'y':
                    return self.parse_batch_output(batch_output_path)
                else:
                    logger.info("Proceeding to create a new batch job...")

            if Path(batch_input_path).exists():
                logger.info(f"Batch input file {batch_input_path} already exists. Using existing file...")
                batch_path = Path(batch_input_path)
            else:
                logger.info("Creating batch input file...")
                batch_path = self.dataset2batch_input_jsonl(dataset, batch_input_path)
            
            
            logger.info("Uploading batch file...")
            file_response = self.client.files.upload(
                file=str(batch_path),
                purpose="batch-api",
                check=False
            )
            logger.info(f"File uploaded: {file_response.id}")
            
            logger.info("Creating batch job...")
            batch = self.client.batches.create_batch(
                file_response.id,
                endpoint="/v1/chat/completions"
            )
            logger.info(f"Batch job created: {batch.id}")
            
            with tqdm(desc="Batch processing") as pbar:
                while batch.status in ["VALIDATING", "IN_PROGRESS"]:
                    time.sleep(poll_interval)
                    batch = self.client.batches.get_batch(batch.id)
                    pbar.set_postfix({"status": batch.status})
            
            if batch.status == "COMPLETED":
                logger.success("Batch completed!")
                
                self.client.files.retrieve_content(
                    id=batch.output_file_id,
                    output=batch_output_path
                )
                logger.success(f"Results saved to {batch_output_path}")
                
                return self.parse_batch_output(batch_output_path)
            
            else:
                logger.error(f"Batch failed: {batch.status}")
                raise RuntimeError(f"Batch failed with status: {batch.status}")
        
        except KeyboardInterrupt:
            logger.warning("Received keyboard interrupt!")
            if batch is not None:
                logger.info(f"Cancelling batch job: {batch.id}")
                cancelled = self.client.batches.cancel_batch(batch.id)
                logger.warning(f"Batch cancelled: {cancelled.status}")
            raise
    
