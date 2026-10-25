"""
Sliding-window perplexity evaluation reused from compare_awq_slicing.py.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict

from src.io_utils import dump_json

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class SlidingWindowEvaluator:
    def __init__(
        self,
        device: str = "cuda",
        seed: int = 42,
        stride: int = 512,
        max_length: int = 2048,
        cache_dir: str = "./data/cache/eval",
        hf_token: str | None = None,
    ):
        self.device = device
        self.seed = seed
        self.stride = stride
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, dict] = {}

    def load_wikitext2_test(self):
        cache_file = self.cache_dir / f"wikitext2_test_seed{self.seed}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as handle:
                return pickle.load(handle)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        full_text = "\n".join([x for x in dataset["text"] if x])
        result = [full_text]
        with open(cache_file, "wb") as handle:
            pickle.dump(result, handle)
        return result

    def load_c4_validation(self, n_samples: int = 500):
        cache_file = self.cache_dir / f"c4_validation_n{n_samples}_seed{self.seed}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as handle:
                return pickle.load(handle)

        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in tqdm(dataset, total=n_samples, desc="Collecting C4"):
            if len(texts) >= n_samples:
                break
            if len(item["text"].strip()) > 500:
                texts.append(item["text"])

        result = ["\n\n".join(texts)]
        with open(cache_file, "wb") as handle:
            pickle.dump(result, handle)
        return result

    @torch.no_grad()
    def evaluate_sliding_window(self, model, tokenizer, texts):
        model.eval()
        nlls = []
        total_tokens = 0

        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = encodings.input_ids

            if tokenizer.bos_token_id is not None:
                if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                    bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

            if input_ids.size(1) > self.max_length * 200:
                input_ids = input_ids[:, : self.max_length * 200]

            input_ids = input_ids.to(self.device)
            seq_len = input_ids.size(1)
            if seq_len < 2:
                continue

            prev_end_loc = 0
            for begin_loc in tqdm(range(0, seq_len, self.stride), desc="Windows", leave=False):
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - prev_end_loc

                input_chunk = input_ids[:, begin_loc:end_loc]
                target_chunk = input_chunk.clone()
                if begin_loc > 0:
                    target_chunk[:, :-trg_len] = -100

                if target_chunk.size(1) == 0:
                    break

                outputs = model(input_chunk, labels=target_chunk)
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc

                if end_loc == seq_len:
                    break

            total_tokens += seq_len

        if not nlls:
            return None

        total_nll = torch.stack(nlls).sum()
        perplexity = torch.exp(total_nll / total_tokens).item()
        return {"perplexity": perplexity, "total_tokens": total_tokens}

    def evaluate_model_on_dataset(self, model_path, model_name: str, texts, dataset_name: str):
        print(f"\nEvaluating {model_name} on {dataset_name}...")
        if isinstance(model_path, dict) and {"model", "tokenizer"}.issubset(model_path):
            model = model_path["model"]
            tokenizer = model_path["tokenizer"]
            if torch.cuda.is_available() and self.device == "cuda":
                model = model.to(self.device)
            return self.evaluate_sliding_window(model, tokenizer, texts)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            token=self.hf_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            token=self.hf_token,
        )
        result = self.evaluate_sliding_window(model, tokenizer, texts)
        del model
        torch.cuda.empty_cache()
        return result

    def run(self, model_paths: Dict[str, str], include_c4: bool = True, c4_samples: int = 500):
        datasets = {"WikiText-2": self.load_wikitext2_test()}
        if include_c4:
            datasets["C4"] = self.load_c4_validation(n_samples=c4_samples)

        for dataset_name, texts in datasets.items():
            self.results[dataset_name] = {}
            for model_name, model_path in model_paths.items():
                self.results[dataset_name][model_name] = self.evaluate_model_on_dataset(
                    model_path,
                    model_name,
                    texts,
                    dataset_name,
                )

        return self.results

    def save_results(self, output_path: str):
        output = Path(output_path)
        dump_json(output, self.results, indent=2)