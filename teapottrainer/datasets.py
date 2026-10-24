import random
from typing import List, Dict, Callable, Optional, Union, Any
from pydantic import BaseModel, ConfigDict, Field
from datasets import load_dataset, Dataset as HFDataset, IterableDataset as HFIterableDataset

class DatasetSourceConfig(BaseModel):
    """Configuration for an individual dataset source in a mix."""
    name: str
    subset: Optional[str] = None
    split: str = "train"
    target_tokens: int = 100_000_000
    weight: float = 1.0
    text_column: str = "text"
    is_sft: bool = False
    
    # Exclude functions from serialization to prevent any Pydantic serialization errors
    preprocess_fn: Optional[Callable[[Dict], Dict]] = Field(default=None, exclude=True)
    chat_template_fn: Optional[Callable[[Dict], tuple[str, str]]] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ─────────────────────────────────────────────────────────────
# Exported Dataset Preset Constants
# ─────────────────────────────────────────────────────────────

FINEWEB_EDU = DatasetSourceConfig(
    name="HuggingFaceFW/fineweb-edu",
    split="train",
    is_sft=False,
    text_column="text"
)

def _smoltalk_formatter(row):
    messages = row.get("messages", [])
    prompt, answer = "", ""
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
            prompt = messages[i]["content"]
            answer = messages[i+1]["content"]
            break
    if not prompt and messages:
        prompt = messages[0].get("content", "")
        answer = messages[-1].get("content", "")
    return prompt, answer

SMOLTALK = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="all",
    split="train",
    is_sft=True,
    chat_template_fn=_smoltalk_formatter
)

def _gsm8k_formatter(row):
    question = row.get("question", "")
    answer = row.get("answer", "")
    return f"Question: {question}\nAnswer:", f" {answer}"

GSM8K = DatasetSourceConfig(
    name="gsm8k",
    subset="main",
    split="train",
    is_sft=True,
    chat_template_fn=_gsm8k_formatter
)


class DatasetRegistry:
    """Manages loading, splitting, tokenizing, and mixing multiple data sources with leakage prevention."""
    
    @staticmethod
    def prepare_dataset_splits(
        sources: List[Union[DatasetSourceConfig, Any]], 
        tokenizer, 
        seq_len: int, 
        is_eval: bool = False,
        holdout_ratio: float = 0.05
    ):
        processed_data_streams = []
        
        for src in sources:
            actual_split = src.split
            if src.name == "gsm8k":
                actual_split = "test" if is_eval else "train"
            
            try:
                raw_ds = load_dataset(src.name, src.subset, split=actual_split, streaming=True)
            except Exception:
                raw_ds = load_dataset(src.name, split=actual_split, streaming=True)

            if is_eval and src.name != "gsm8k" and actual_split == "train":
                raw_ds = raw_ds.shard(num_shards=int(1 / holdout_ratio), index=0, contiguous=True)
            elif not is_eval and src.name != "gsm8k" and actual_split == "train":
                try:
                    raw_ds = raw_ds.shard(num_shards=int(1 / holdout_ratio), index=1, contiguous=True)
                except Exception:
                    pass

            def generator():
                for item in raw_ds:
                    if src.preprocess_fn:
                        item = src.preprocess_fn(item)

                    if src.is_sft and src.chat_template_fn:
                        prompt, answer = src.chat_template_fn(item)
                        if not prompt or not answer:
                            continue
                        
                        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                        answer_ids = tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False)
                        
                        full_ids = prompt_ids + answer_ids
                        labels_mask = [-100] * len(prompt_ids) + answer_ids

                        while len(full_ids) >= seq_len:
                            chunk_ids = full_ids[:seq_len]
                            chunk_labels = labels_mask[:seq_len]
                            
                            full_ids = full_ids[seq_len:]
                            labels_mask = labels_mask[seq_len:]
                            yield {
                                "input_ids": chunk_ids, 
                                "labels": chunk_labels,
                                "is_sft": True
                            }
                    else:
                        text = item.get(src.text_column, item.get("text", item.get("content", "")))
                        if not text:
                            continue
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        
                        buffer = ids
                        while len(buffer) >= seq_len:
                            chunk = buffer[:seq_len]
                            buffer = buffer[seq_len:]
                            yield {
                                "input_ids": chunk,
                                "labels": chunk,
                                "is_sft": False
                            }

            processed_data_streams.append(generator())

        return processed_data_streams