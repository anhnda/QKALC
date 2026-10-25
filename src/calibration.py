"""
Calibration data loaders reused from the original experimental scripts.
"""

from __future__ import annotations

import hashlib
import pickle
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset


def _normalize_cache_identity(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return text.strip("-")[:48] or "unknown"


def _get_tokenizer_identity(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "name", None) or type(tokenizer).__name__
    vocab_size = _get_tokenizer_vocab_size(tokenizer)
    fingerprint_source = "|".join(
        [
            str(name),
            str(vocab_size),
            str(getattr(tokenizer, "is_fast", None)),
            str(getattr(tokenizer, "vocab_files_names", None)),
        ]
    )
    digest = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()[:10]
    return f"{_normalize_cache_identity(name)}-{digest}"


def _get_tokenizer_vocab_size(tokenizer) -> int | None:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int):
        return vocab_size
    try:
        computed_size = len(tokenizer)
    except (TypeError, AttributeError):
        return None
    return computed_size if isinstance(computed_size, int) else None


def _calibration_cache_file(
    dataset_name: str,
    tokenizer,
    n_samples: int,
    seqlen: int,
    seed: int,
    return_tensors: bool,
    cache_dir: str,
    split: str | None = None,
) -> Path:
    cache_path = Path(cache_dir)
    suffix = f"_split{split}" if split is not None else ""
    tokenizer_key = _get_tokenizer_identity(tokenizer)
    filename = (
        f"{dataset_name}_calib_n{n_samples}_len{seqlen}_seed{seed}"
        f"{suffix}_tensors{return_tensors}_tok{tokenizer_key}.pkl"
    )
    return cache_path / filename


def _legacy_calibration_cache_file(
    dataset_name: str,
    n_samples: int,
    seqlen: int,
    seed: int,
    return_tensors: bool,
    cache_dir: str,
    split: str | None = None,
) -> Path:
    cache_path = Path(cache_dir)
    suffix = f"_split{split}" if split is not None else ""
    return cache_path / f"{dataset_name}_calib_n{n_samples}_len{seqlen}_seed{seed}{suffix}_tensors{return_tensors}.pkl"


def _validate_tensor_samples_with_vocab(samples, vocab_size: int | None, cache_label: str):
    if vocab_size is None:
        return
    for sample_idx, sample in enumerate(samples):
        if not isinstance(sample, torch.Tensor) or sample.numel() == 0:
            continue
        min_token = int(sample.min().item())
        max_token = int(sample.max().item())
        if min_token < 0 or max_token >= vocab_size:
            raise ValueError(
                f"{cache_label} contains token ids outside tokenizer vocabulary range "
                f"[0, {vocab_size - 1}] (sample {sample_idx}, min={min_token}, max={max_token}). "
                "Delete this cache and regenerate calibration data for the current model."
            )


def _load_cached_samples(cache_file: Path, tokenizer, return_tensors: bool, cache_label: str):
    with open(cache_file, "rb") as handle:
        samples = pickle.load(handle)
    if return_tensors:
        _validate_tensor_samples_with_vocab(samples, _get_tokenizer_vocab_size(tokenizer), cache_label)
    return samples


def _try_load_calibration_cache(
    cache_file: Path,
    tokenizer,
    return_tensors: bool,
    cache_label: str,
):
    if not cache_file.exists():
        return None
    try:
        print(f"  Loading {cache_label}: {cache_file}")
        return _load_cached_samples(cache_file, tokenizer, return_tensors, cache_label)
    except ValueError as exc:
        print(f"  Ignoring invalid {cache_label}: {exc}")
        try:
            cache_file.unlink()
            print(f"  Deleted invalid cache file: {cache_file}")
        except OSError:
            pass
        return None


def get_c4_calibration_data(
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
    return_tensors: bool = False,
    cache_dir: str = "./data/cache/calibration",
):
    print("\n[C4 Calibration Data - Optimized]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Seed: {seed}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = _calibration_cache_file(
        "c4",
        tokenizer,
        n_samples,
        seqlen,
        seed,
        return_tensors,
        cache_dir,
    )
    cached = _try_load_calibration_cache(cache_file, tokenizer, return_tensors, "calibration cache")
    if cached is not None:
        return cached

    legacy_cache_file = _legacy_calibration_cache_file("c4", n_samples, seqlen, seed, return_tensors, cache_dir)
    legacy_cached = _try_load_calibration_cache(
        legacy_cache_file,
        tokenizer,
        return_tensors,
        "legacy calibration cache",
    )
    if legacy_cached is not None:
        try:
            with open(cache_file, "wb") as handle:
                pickle.dump(legacy_cached, handle)
            print(f"  Migrated legacy cache to: {cache_file}")
            try:
                legacy_cache_file.unlink()
                print(f"  Deleted legacy cache file: {legacy_cache_file}")
            except OSError:
                pass
        except OSError:
            pass
        return legacy_cached

    random.seed(seed)
    url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
    traindata = load_dataset(
        "json",
        data_files={"train": url},
        split="train",
        streaming=True,
    )

    dataset = []
    skipped = 0
    char_threshold = seqlen * 3

    for data in traindata:
        text = data["text"]
        if len(text) < char_threshold:
            skipped += 1
            continue

        trainenc = tokenizer(text, return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            skipped += 1
            continue

        max_start = trainenc.input_ids.shape[1] - seqlen
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + seqlen
        inp = trainenc.input_ids[:, start_idx:end_idx]

        if return_tensors:
            dataset.append(inp)
        else:
            dataset.append(tokenizer.decode(inp[0], skip_special_tokens=True))

        if len(dataset) == n_samples:
            break

    print(f"  Collected {len(dataset)} samples from C4")
    print(f"  Skipped {skipped} documents")

    with open(cache_file, "wb") as handle:
        pickle.dump(dataset, handle)

    return dataset


def get_wikitext2_calibration_data(
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
    split: str = "train",
    return_tensors: bool = False,
    cache_dir: str = "./data/cache/calibration",
):
    print("\n[WikiText-2 Calibration Data]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Split: {split}")
    print(f"  Seed: {seed}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = _calibration_cache_file(
        "wikitext2",
        tokenizer,
        n_samples,
        seqlen,
        seed,
        return_tensors,
        cache_dir,
        split=split,
    )
    cached = _try_load_calibration_cache(cache_file, tokenizer, return_tensors, "calibration cache")
    if cached is not None:
        return cached

    legacy_cache_file = _legacy_calibration_cache_file(
        "wikitext2",
        n_samples,
        seqlen,
        seed,
        return_tensors,
        cache_dir,
        split=split,
    )
    legacy_cached = _try_load_calibration_cache(
        legacy_cache_file,
        tokenizer,
        return_tensors,
        "legacy calibration cache",
    )
    if legacy_cached is not None:
        try:
            with open(cache_file, "wb") as handle:
                pickle.dump(legacy_cached, handle)
            print(f"  Migrated legacy cache to: {cache_file}")
            try:
                legacy_cache_file.unlink()
                print(f"  Deleted legacy cache file: {legacy_cache_file}")
            except OSError:
                pass
        except OSError:
            pass
        return legacy_cached

    random.seed(seed)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item["text"] for item in dataset if item["text"].strip()]

    all_tokens = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        all_tokens.append(tokens)

    all_tokens = torch.cat(all_tokens, dim=0)
    num_chunks = len(all_tokens) // seqlen
    if num_chunks < n_samples:
        n_samples = num_chunks

    chunk_indices = random.sample(range(num_chunks), n_samples)
    calibration_samples = []
    for idx in chunk_indices:
        start = idx * seqlen
        end = start + seqlen
        chunk_tokens = all_tokens[start:end]
        if return_tensors:
            calibration_samples.append(chunk_tokens.unsqueeze(0).clone())
        else:
            calibration_samples.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

    with open(cache_file, "wb") as handle:
        pickle.dump(calibration_samples, handle)

    return calibration_samples

def load_calibration_data(
    dataset_name: str,
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
    return_tensors: bool = False,
    cache_dir: str = "./data/cache/calibration",
):
    dataset_name = dataset_name.lower()

    if dataset_name == "c4":
        return get_c4_calibration_data(
            tokenizer,
            n_samples=n_samples,
            seqlen=seqlen,
            seed=seed,
            return_tensors=return_tensors,
            cache_dir=cache_dir,
        )
    if dataset_name == "wikitext2":
        return get_wikitext2_calibration_data(
            tokenizer,
            n_samples=n_samples,
            seqlen=seqlen,
            seed=seed,
            split="train",
            return_tensors=return_tensors,
            cache_dir=cache_dir,
        )
    if dataset_name == "wikitext2-simple":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        return [item["text"] for item in dataset if len(item["text"].strip()) > 100][:n_samples]

    raise ValueError(f"Unknown calibration dataset: {dataset_name}")
