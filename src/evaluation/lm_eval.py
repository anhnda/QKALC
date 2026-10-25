"""
lm-evaluation-harness integration for downstream benchmark evaluation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.io_utils import dump_json


class LMEvalHarnessRunner:
    def __init__(
        self,
        tasks: list[str],
        device: str = "cuda",
        batch_size: str = "auto",
        num_fewshot: int | None = None,
        output_dir: str = "./results/eval/lm_eval",
        run_name: str | None = None,
        hf_token: str | None = None,
    ):
        self.tasks = tasks
        self.device = device
        self.batch_size = batch_size
        self.num_fewshot = num_fewshot
        self.output_dir = Path(output_dir)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.hf_token = hf_token
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _model_args(self, model_path: str) -> str:
        dtype = "float16" if self.device == "cuda" else "float32"
        model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code=True"
        if self.hf_token:
            model_args += f",token={self.hf_token}"
        return model_args

    def _make_json_safe(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if callable(value):
            name = getattr(value, "__name__", value.__class__.__name__)
            return f"<callable {name}>"

        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(item) for item in value]

        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return self._make_json_safe(item_method())
            except (TypeError, ValueError):
                pass

        return repr(value)

    def _summarize_results(self, payload: dict) -> dict:
        results = payload.get("results", {})
        summary = {}
        for task_name, metrics in results.items():
            task_summary = {}
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    task_summary[metric_name] = value
            summary[task_name] = task_summary
        return summary

    def _write_raw_results(self, model_name: str, payload: dict):
        output_path = self.output_dir / f"{self.run_name}_{model_name}.json"
        safe_payload = self._make_json_safe(payload)
        dump_json(output_path, safe_payload, indent=2)

    def evaluate_model(self, model_name: str, model_path) -> dict:
        try:
            from lm_eval import evaluator
        except ImportError as exc:
            raise RuntimeError(
                "lm-eval is not installed. Install the 'lm-eval' package or disable lm-eval with --no-lm-eval."
            ) from exc

        if isinstance(model_path, dict) and {"model", "tokenizer"}.issubset(model_path):
            from lm_eval.models.huggingface import HFLM

            model = model_path["model"]
            tokenizer = model_path["tokenizer"]
            if self.device == "cuda":
                model = model.to(self.device)
            eval_batch_size = 1 if self.batch_size == "auto" else self.batch_size
            hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)
            payload = evaluator.simple_evaluate(
                model=hflm,
                tasks=self.tasks,
                device=self.device,
                batch_size=eval_batch_size,
                num_fewshot=self.num_fewshot,
                log_samples=False,
            )
        else:
            payload = evaluator.simple_evaluate(
                model="hf",
                model_args=self._model_args(model_path),
                tasks=self.tasks,
                device=self.device,
                batch_size=self.batch_size,
                num_fewshot=self.num_fewshot,
                log_samples=False,
            )
        safe_payload = self._make_json_safe(payload)
        self._write_raw_results(model_name, payload)
        return {
            "tasks": list(self.tasks),
            "summary": self._summarize_results(payload),
            "raw": safe_payload,
        }

    def run(self, model_paths: Dict[str, str], on_progress=None) -> dict:
        """Evaluate each model. If on_progress is given, it is called as
        on_progress(results_so_far) after every model, so the caller can
        checkpoint partial results to disk instead of waiting for all models.
        """
        results = {}
        for model_name, model_path in model_paths.items():
            print(f"\nRunning lm-eval for {model_name} on {', ' .join(self.tasks)}...")
            results[model_name] = self.evaluate_model(model_name, model_path)
            if on_progress is not None:
                try:
                    on_progress(results)
                except Exception as exc:  # checkpointing must never kill the run
                    print(f"[lm-eval] on_progress checkpoint failed: {exc}")
        return results
