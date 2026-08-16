"""Run LaViT on HR-Bench and export latent/answer attention analyses.

Edit the global variables below, then run:

    python evaluation/utils/hrbench_attention_analysis.py

There is deliberately no command-line interface. Generation uses the native
Hugging Face LaViT model. The sampled sequence is then replayed one token at a
time with a KV cache so only one query row of decoder attention is materialized
at a time; a full prompt-by-prompt attention square is never retained.
"""

from __future__ import annotations

import csv
import gc
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# =============================================================================
# Global configuration -- edit values here; no CLI arguments are used
# =============================================================================

MODEL_PATH = "/home/fit/renjujty/WORK/jty/lmllms/lavit/"
HRBENCH_PATH = (
    "/home/fit/renjujty/WORK/jty/lmllms/hrbench/hr_bench_4k.parquet"
)
OUTPUT_DIR = "outputs/hrbench_attention"

RESULTS_FILE = "results.jsonl"
RUN_CONFIG_FILE = "run_config.json"
CATEGORY_ATTENTION_CSV_FILE = "category_attention.csv"
LATENT_TOPK_CSV_FILE = "latent_topk.csv"
ATTENTION_SUBDIR = "attention"
PLOT_SUBDIR = "plots"

# "sequential": START_INDEX ... START_INDEX + NUM_SAMPLES
# "random": deterministic sampling without replacement using RANDOM_SEED
SELECTION_MODE = "sequential"
START_INDEX = 199
NUM_SAMPLES = 1
RANDOM_SEED = 0

DEVICE = "cuda"
TORCH_DTYPE = "bfloat16"
TRUST_REMOTE_CODE = True
USE_CACHE = True

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 8192 * 28 * 28
MAX_OUTPUT_TOKENS = 4096

# Keep the sampling settings from Monet's vLLM inference example. Hugging Face
# and vLLM do not promise identical random streams for the same seed.
TEMPERATURE = 0.1
TOP_K = 50
TOP_P = 0.8
REPETITION_PENALTY = 1.01
BEST_OF = 1
STOP = None

LATENT_TOP_K = 20
ATTENTION_STORAGE_DTYPE = "float16"  # "float16" or "float32"
# Fixed semantics selected for this analysis. They are explicit globals so a
# run_config snapshot documents them rather than hiding them in helper logic.
ANSWER_SCOPE = "all_non_latent_text"
LATENT_TOPK_SOURCE = "contextual_hidden_state"
CLEAN_LATENT_MARKER = "<latent>"
PLOT_LAYER = -1                       # Python-style decoder layer index
PLOT_DPI = 180
PLOT_MAX_TOKEN_LABELS = 80
PLOT_FIGURE_WIDTH = 18.0
PLOT_ROW_HEIGHT = 0.38


REQUIRED_COLUMNS = {
    "index", "question", "answer", "category", "A", "B", "C", "D",
    "cycle_category", "image",
}
LVR_TOKEN_PATTERN = re.compile(r"<lvr\d*>", flags=re.IGNORECASE)
SOURCE_KIND_NAMES = np.asarray([
    "input_text", "input_visual", "latent", "generated_text", "special",
])
SOURCE_INPUT_TEXT = 0
SOURCE_INPUT_VISUAL = 1
SOURCE_LATENT = 2
SOURCE_GENERATED_TEXT = 3
SOURCE_SPECIAL = 4
QUERY_KIND_NAMES = np.asarray(["latent", "answer"])
QUERY_LATENT = 0
QUERY_ANSWER = 1


def select_sample_indices(
    total: int,
    mode: str = SELECTION_MODE,
    start_index: int = START_INDEX,
    count: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED,
) -> list[int]:
    """Select deterministic HR-Bench row ordinals."""
    if total <= 0:
        raise ValueError("HR-Bench is empty.")
    if count <= 0 or count > total:
        raise ValueError(f"NUM_SAMPLES must be in [1, {total}], received {count}.")
    if mode == "sequential":
        if start_index < 0 or start_index + count > total:
            raise ValueError(
                f"Sequential range [{start_index}, {start_index + count}) "
                f"is outside dataset size {total}."
            )
        return list(range(start_index, start_index + count))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(total, size=count, replace=False).tolist()
    raise ValueError("SELECTION_MODE must be 'sequential' or 'random'.")


def validate_configuration() -> tuple[Path, Path, Path]:
    """Validate global options and create the output directories."""
    if ATTENTION_STORAGE_DTYPE not in {"float16", "float32"}:
        raise ValueError(
            "ATTENTION_STORAGE_DTYPE must be 'float16' or 'float32'."
        )
    if LATENT_TOP_K <= 0 or MAX_OUTPUT_TOKENS <= 0:
        raise ValueError("LATENT_TOP_K and MAX_OUTPUT_TOKENS must be positive.")
    if BEST_OF != 1:
        raise ValueError("This attention replay requires BEST_OF=1.")
    if ANSWER_SCOPE != "all_non_latent_text":
        raise ValueError("Only ANSWER_SCOPE='all_non_latent_text' is supported.")
    if LATENT_TOPK_SOURCE != "contextual_hidden_state":
        raise ValueError(
            "Only LATENT_TOPK_SOURCE='contextual_hidden_state' is supported."
        )
    if not USE_CACHE:
        raise ValueError("Attention replay requires USE_CACHE=True.")
    if MIN_PIXELS <= 0 or MAX_PIXELS < MIN_PIXELS:
        raise ValueError("MIN_PIXELS/MAX_PIXELS are invalid.")
    if STOP is not None:
        raise ValueError(
            "STOP is retained for configuration compatibility but only None "
            "is supported by this fixed-token replay script."
        )

    model_path = Path(MODEL_PATH).expanduser()
    dataset_path = Path(HRBENCH_PATH).expanduser()
    output_path = Path(OUTPUT_DIR).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {model_path}. Edit the global value."
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"HRBENCH_PATH does not exist: {dataset_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / ATTENTION_SUBDIR).mkdir(exist_ok=True)
    (output_path / PLOT_SUBDIR).mkdir(exist_ok=True)
    return model_path.resolve(), dataset_path.resolve(), output_path.resolve()


def load_hrbench_rows(
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], list[int]]:
    import pandas as pd

    dataframe = pd.read_parquet(dataset_path)
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(
            "Unexpected HR-Bench schema; missing columns: "
            + ", ".join(sorted(missing))
        )
    selected = select_sample_indices(len(dataframe))
    return [dict(dataframe.iloc[index]) for index in selected], selected


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_pca_helpers():
    """Import the existing, tested HR-Bench prompt/image helpers."""
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from evaluation.utils.hrbench_pca_analysis import (  # noqa: PLC0415
        build_question,
        decode_hrbench_image,
    )

    return build_question, decode_hrbench_image


def _torch_dtype_from_name(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(
            "TORCH_DTYPE must be 'float16', 'bfloat16', or 'float32'."
        ) from exc


def discover_lvr_token_ids(tokenizer: Any, vocab_size: int) -> dict[int, str]:
    """Return checkpoint-resident single-token <lvr>, <lvr1>, ... entries."""
    found: dict[int, str] = {}
    for token, token_id in tokenizer.get_vocab().items():
        token_id = int(token_id)
        if LVR_TOKEN_PATTERN.fullmatch(str(token)) and 0 <= token_id < vocab_size:
            found[token_id] = str(token)
    if not found:
        raise RuntimeError(
            "No <lvr*> token was found in the checkpoint tokenizer. The "
            "analysis will not add or resize tokenizer entries."
        )
    return dict(sorted(found.items()))


def _locate_decoder_layers(model: Any) -> list[tuple[str, Any]]:
    candidates = (
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.layers",
    )
    for dotted_name in candidates:
        value = model
        for component in dotted_name.split("."):
            value = getattr(value, component, None)
            if value is None:
                break
        if value is not None:
            layers = list(value)
            if layers and all(hasattr(layer, "self_attn") for layer in layers):
                return [
                    (f"{dotted_name}.{index}.self_attn", layer.self_attn)
                    for index, layer in enumerate(layers)
                ]
    raise RuntimeError(
        "Could not locate contiguous Qwen2.5-VL decoder self-attention layers."
    )


def load_model_and_processor(model_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load LaViT with eager decoder attention for one-row replay."""
    import torch
    from transformers import Qwen2VLProcessor

    training_src = _project_root() / "training" / "src"
    if str(training_src) not in sys.path:
        sys.path.insert(0, str(training_src))
    from modeling_lavit import LaViTConfig, LaViTQwen2VL  # noqa: PLC0415

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE={DEVICE!r}, but CUDA is not available.")
    dtype = _torch_dtype_from_name(TORCH_DTYPE)
    processor = Qwen2VLProcessor.from_pretrained(
        str(model_path),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    config = LaViTConfig.from_pretrained(
        str(model_path), trust_remote_code=TRUST_REMOTE_CODE
    )
    # Both forms are set because Transformers releases differ in which one is
    # inspected when selecting the attention implementation.
    config._attn_implementation = "eager"
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = "eager"
    model = LaViTQwen2VL.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=dtype,
        device_map=DEVICE,
        trust_remote_code=TRUST_REMOTE_CODE,
        attn_implementation="eager",
    )
    model.eval()
    embedding = model.get_input_embeddings()
    vocab_size, hidden_size = map(int, embedding.weight.shape)
    lvr_tokens = discover_lvr_token_ids(processor.tokenizer, vocab_size)
    decoder_layers = _locate_decoder_layers(model)
    details = {
        "backend": "huggingface",
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "input_device": str(embedding.weight.device),
        "torch_dtype": str(dtype),
        "attention_implementation": "eager",
        "decoder_layer_count": len(decoder_layers),
        "decoder_layer_names": [name for name, _ in decoder_layers],
        "lvr_token_ids": {str(key): value for key, value in lvr_tokens.items()},
    }
    print("[LaViT attention] model configuration:")
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return model, processor, details


def _move_processor_inputs(inputs: Any, device: Any) -> dict[str, Any]:
    moved = {}
    for key, value in dict(inputs).items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _set_random_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generation_kwargs(tokenizer: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": MAX_OUTPUT_TOKENS,
        "do_sample": TEMPERATURE > 0,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "num_return_sequences": BEST_OF,
        "use_cache": USE_CACHE,
    }
    if TEMPERATURE > 0:
        kwargs["temperature"] = TEMPERATURE
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.eos_token_id
    return kwargs


def _get_full_prompt_positions(model: Any, inputs: dict[str, Any]):
    """Calculate Qwen mRoPE positions once for the unsplit prompt."""
    getter_owner = getattr(model, "model", None)
    getter = getattr(getter_owner, "get_rope_index", None)
    if not callable(getter):
        raise RuntimeError("LaViT/Qwen model.get_rope_index is unavailable.")
    return getter(
        inputs["input_ids"],
        inputs.get("image_grid_thw"),
        inputs.get("video_grid_thw"),
        inputs["attention_mask"],
    )


def _generated_position_ids(
    prompt_length: int, output_index: int, rope_deltas: Any, device: Any
):
    """Build the three identical post-prompt mRoPE position channels."""
    import torch

    delta = rope_deltas.to(device=device, dtype=torch.long).reshape(1, -1, 1)
    base = torch.full(
        (3, delta.shape[1], 1),
        prompt_length + output_index,
        dtype=torch.long,
        device=device,
    )
    return base + delta


def _extract_query_attention(
    attentions: Any, expected_layers: int, source_count: int
) -> np.ndarray:
    """Return [decoder_layer, source] head-mean attention for one query."""
    import torch

    if attentions is None or len(attentions) != expected_layers:
        observed = 0 if attentions is None else len(attentions)
        raise RuntimeError(
            "Eager attention replay returned an unexpected layer count: "
            f"{observed} != {expected_layers}."
        )
    rows = []
    for layer_index, weights in enumerate(attentions):
        if not isinstance(weights, torch.Tensor) or weights.ndim != 4:
            raise RuntimeError(
                f"Layer {layer_index} did not return [batch, head, query, key] "
                "attention weights. Ensure attn_implementation='eager'."
            )
        if int(weights.shape[0]) != 1 or int(weights.shape[-2]) != 1:
            raise RuntimeError(
                "Replay must materialize exactly one query row, received "
                f"layer {layer_index} shape {tuple(weights.shape)}."
            )
        if int(weights.shape[-1]) < source_count:
            raise RuntimeError(
                f"Layer {layer_index} has only {weights.shape[-1]} sources; "
                f"expected {source_count}."
            )
        row = weights[0, :, 0, :source_count].float().mean(dim=0)
        rows.append(row.to(device="cpu").numpy())
    matrix = np.stack(rows).astype(np.float32, copy=False)
    sums = matrix.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=2e-4, rtol=2e-4):
        raise RuntimeError(
            "Captured head-mean attention does not sum to one; layer sums "
            f"range from {sums.min():.6f} to {sums.max():.6f}."
        )
    return matrix


def select_answer_token_indices(
    generated_token_ids: list[int],
    latent_token_ids: set[int],
    special_token_ids: set[int],
) -> tuple[list[int], bool]:
    """Select every generated readable, non-latent text token."""
    answer_indices = [
        index
        for index, token_id in enumerate(generated_token_ids)
        if token_id not in latent_token_ids and token_id not in special_token_ids
    ]
    return answer_indices, not any(
        token_id in latent_token_ids for token_id in generated_token_ids
    )


def plan_query_alignment(
    prompt_length: int,
    generated_token_ids: list[int],
    latent_token_ids: set[int],
    special_token_ids: set[int],
) -> dict[str, list[dict[str, int]]]:
    """Describe query/token alignment without requiring Torch or a model.

    An autoregressive token at output index ``i`` is predicted by sequence
    position ``prompt_length + i - 1``. A latent's own attention is produced
    only after that latent is consumed, at ``prompt_length + i``.
    """
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive.")
    answer_indices, _ = select_answer_token_indices(
        generated_token_ids, latent_token_ids, special_token_ids
    )
    latent_records = []
    answer_records = []
    latent_index = 0
    for output_index, token_id in enumerate(generated_token_ids):
        if output_index in answer_indices:
            answer_records.append({
                "query_sequence_position": prompt_length + output_index - 1,
                "output_index": output_index,
                "predicted_token_id": int(token_id),
            })
        if token_id in latent_token_ids:
            latent_records.append({
                "query_sequence_position": prompt_length + output_index,
                "output_index": output_index,
                "latent_index": latent_index,
            })
            latent_index += 1
    return {"latent_records": latent_records, "answer_records": answer_records}


def _finish_reason(output_ids: list[int], tokenizer: Any) -> str:
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos_ids.discard(None)
    return "stop" if output_ids and output_ids[-1] in eos_ids else "length"


def _run_replay_step(
    model: Any,
    *,
    token_ids: Any,
    attention_mask: Any,
    position_ids: Any,
    past_key_values: Any,
    cache_position: Any,
    rope_deltas: Any,
):
    return model(
        input_ids=token_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        rope_deltas=rope_deltas,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )


def replay_generated_sequence(
    model: Any,
    inputs: dict[str, Any],
    generated_token_ids: list[int],
    *,
    layer_count: int,
    latent_token_ids: set[int],
    special_token_ids: set[int],
) -> dict[str, Any]:
    """Replay one generated sequence and capture aligned query rows."""
    import torch

    prompt_ids = inputs["input_ids"]
    prompt_mask = inputs["attention_mask"]
    if tuple(prompt_ids.shape[:1]) != (1,) or tuple(prompt_mask.shape[:1]) != (1,):
        raise RuntimeError("Attention replay supports exactly one sample at a time.")
    prompt_length = int(prompt_ids.shape[1])
    if prompt_length < 2:
        raise RuntimeError("The multimodal chat prompt unexpectedly has fewer than 2 tokens.")

    full_position_ids, rope_deltas = _get_full_prompt_positions(model, inputs)
    input_device = prompt_ids.device
    prefix_length = prompt_length - 1
    prefix_kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "position_ids"}
    }
    prefix_output = model(
        input_ids=prompt_ids[:, :prefix_length],
        attention_mask=prompt_mask[:, :prefix_length],
        position_ids=full_position_ids[:, :, :prefix_length],
        cache_position=torch.arange(prefix_length, device=input_device),
        rope_deltas=rope_deltas,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **prefix_kwargs,
    )
    past_key_values = prefix_output.past_key_values
    del prefix_output

    last_prompt_output = _run_replay_step(
        model,
        token_ids=prompt_ids[:, -1:],
        attention_mask=prompt_mask,
        position_ids=full_position_ids[:, :, -1:],
        past_key_values=past_key_values,
        cache_position=torch.tensor([prefix_length], device=input_device),
        rope_deltas=rope_deltas,
    )
    past_key_values = last_prompt_output.past_key_values
    prediction_attention = _extract_query_attention(
        last_prompt_output.attentions, layer_count, prompt_length
    )
    prediction_query_position = prompt_length - 1
    del last_prompt_output

    answer_index_set = set(select_answer_token_indices(
        generated_token_ids, latent_token_ids, special_token_ids
    )[0])
    alignment = plan_query_alignment(
        prompt_length,
        generated_token_ids,
        latent_token_ids,
        special_token_ids,
    )
    answer_alignment = {
        record["output_index"]: record for record in alignment["answer_records"]
    }
    latent_alignment = {
        record["output_index"]: record for record in alignment["latent_records"]
    }
    latent_records: list[dict[str, Any]] = []
    answer_records: list[dict[str, Any]] = []
    latent_topk: list[dict[str, Any]] = []
    latent_positions: list[int] = []
    latent_index = 0
    running_mask = prompt_mask
    storage_dtype = np.dtype(ATTENTION_STORAGE_DTYPE)

    for output_index, token_id in enumerate(generated_token_ids):
        if output_index in answer_index_set:
            expected = answer_alignment[output_index]
            if prediction_query_position != expected["query_sequence_position"]:
                raise RuntimeError("Answer prediction/query alignment drifted.")
            answer_records.append({
                "query_sequence_position": prediction_query_position,
                "output_index": output_index,
                "predicted_token_id": token_id,
                "matrix": prediction_attention.astype(
                    storage_dtype, copy=False
                ),
            })

        sequence_position = prompt_length + output_index
        one = torch.ones(
            (1, 1), dtype=running_mask.dtype, device=running_mask.device
        )
        running_mask = torch.cat((running_mask, one), dim=1)
        step_ids = torch.tensor(
            [[token_id]], dtype=prompt_ids.dtype, device=input_device
        )
        step_output = _run_replay_step(
            model,
            token_ids=step_ids,
            attention_mask=running_mask,
            position_ids=_generated_position_ids(
                prompt_length, output_index, rope_deltas, input_device
            ),
            past_key_values=past_key_values,
            cache_position=torch.tensor([sequence_position], device=input_device),
            rope_deltas=rope_deltas,
        )
        past_key_values = step_output.past_key_values
        consumed_attention = _extract_query_attention(
            step_output.attentions, layer_count, sequence_position + 1
        )
        if token_id in latent_token_ids:
            expected = latent_alignment[output_index]
            if sequence_position != expected["query_sequence_position"]:
                raise RuntimeError("Latent consumption/query alignment drifted.")
            logits = step_output.logits[0, -1].float()
            k = min(LATENT_TOP_K, int(logits.shape[-1]))
            values, ids = torch.topk(logits, k=k, dim=-1)
            latent_positions.append(sequence_position)
            latent_records.append({
                "query_sequence_position": sequence_position,
                "output_index": output_index,
                "latent_index": latent_index,
                "matrix": consumed_attention.astype(storage_dtype, copy=False),
            })
            latent_topk.append({
                "query_sequence_position": sequence_position,
                "latent_index": latent_index,
                "token_ids": ids.to(device="cpu").numpy().astype(int).tolist(),
                "logits": values.to(device="cpu").numpy().astype(float).tolist(),
            })
            latent_index += 1
        prediction_attention = consumed_attention
        prediction_query_position = sequence_position
        del step_output

    return {
        "prompt_token_ids": [int(value) for value in prompt_ids[0].tolist()],
        "prompt_length": prompt_length,
        "generated_token_ids": generated_token_ids,
        "latent_positions": latent_positions,
        "latent_records": latent_records,
        "answer_records": answer_records,
        "latent_topk": latent_topk,
        "no_latent_fallback": len(latent_positions) == 0,
    }


def classify_source_positions(
    source_positions: np.ndarray,
    prompt_length: int,
    image_positions: set[int],
    latent_positions: set[int],
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    special_token_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    kinds = np.empty(len(source_positions), dtype=np.uint8)
    token_ids = np.full(len(source_positions), -1, dtype=np.int32)
    for index, position_value in enumerate(source_positions):
        position = int(position_value)
        if position < prompt_length:
            token_id = int(prompt_token_ids[position])
        else:
            output_index = position - prompt_length
            token_id = (
                int(generated_token_ids[output_index])
                if 0 <= output_index < len(generated_token_ids)
                else -1
            )
        token_ids[index] = token_id
        if position in latent_positions:
            kinds[index] = SOURCE_LATENT
        elif position in image_positions:
            kinds[index] = SOURCE_INPUT_VISUAL
        elif position < prompt_length:
            kinds[index] = (
                SOURCE_SPECIAL if token_id in special_token_ids
                else SOURCE_INPUT_TEXT
            )
        elif token_id in special_token_ids:
            kinds[index] = SOURCE_SPECIAL
        else:
            kinds[index] = SOURCE_GENERATED_TEXT
    return kinds, token_ids


def normalize_attention_groups(
    raw: np.ndarray,
    source_kinds: np.ndarray,
    target_kind_codes: tuple[int, ...],
) -> np.ndarray:
    normalized = np.zeros_like(raw, dtype=np.float32)
    mask = np.isin(source_kinds, target_kind_codes)
    if not mask.any():
        return normalized
    denominator = raw[:, mask].sum(axis=1, keepdims=True, dtype=np.float32)
    valid = denominator[:, 0] > 0
    normalized[np.ix_(valid, mask)] = (
        raw[np.ix_(valid, mask)].astype(np.float32) / denominator[valid]
    )
    return normalized


def assemble_sample_archive(
    capture: dict[str, Any],
    *,
    image_positions: set[int],
    special_token_ids: set[int],
    layer_names: list[str],
) -> dict[str, np.ndarray]:
    """Assemble the same ragged NPZ layout as the Monet reference script."""
    dtype = np.dtype(ATTENTION_STORAGE_DTYPE)
    layer_count = len(layer_names)
    prompt_length = int(capture["prompt_length"])
    prompt_ids = list(map(int, capture["prompt_token_ids"]))
    generated_ids = list(map(int, capture["generated_token_ids"]))
    latent_positions = set(map(int, capture["latent_positions"]))

    query_source_offsets = [0]
    query_kinds = []
    query_positions = []
    query_output_indices = []
    query_predicted_ids = []
    query_latent_indices = []
    source_positions_all = []
    source_kinds_all = []
    source_token_ids_all = []
    raw_blocks = []
    normalized_blocks = []
    category_mass = []

    streams = [
        ("latent_records", QUERY_LATENT,
         (SOURCE_INPUT_TEXT, SOURCE_INPUT_VISUAL)),
        ("answer_records", QUERY_ANSWER,
         (SOURCE_INPUT_TEXT, SOURCE_INPUT_VISUAL, SOURCE_LATENT)),
    ]
    for record_key, query_kind, normalization_kinds in streams:
        for record in capture[record_key]:
            raw = np.asarray(record["matrix"], dtype=dtype)
            if raw.ndim != 2 or raw.shape[0] != layer_count:
                raise RuntimeError(
                    f"Unexpected captured matrix shape {raw.shape}; expected "
                    f"({layer_count}, source_count)."
                )
            source_count = int(raw.shape[1])
            source_positions = np.arange(source_count, dtype=np.int32)
            source_kinds, source_token_ids = classify_source_positions(
                source_positions,
                prompt_length,
                image_positions,
                latent_positions,
                prompt_ids,
                generated_ids,
                special_token_ids,
            )
            tolerance = 5e-3 if dtype == np.dtype("float16") else 2e-4
            if not np.allclose(
                raw.astype(np.float32).sum(axis=1),
                1.0,
                atol=tolerance,
                rtol=tolerance,
            ):
                raise RuntimeError(
                    "Stored attention does not sum to one for query at "
                    f"position {record['query_sequence_position']}."
                )
            normalized = normalize_attention_groups(
                raw, source_kinds, normalization_kinds
            )
            masses = np.stack([
                raw[:, source_kinds == kind].astype(np.float32).sum(axis=1)
                for kind in range(len(SOURCE_KIND_NAMES))
            ], axis=-1)
            raw_blocks.append(raw)
            normalized_blocks.append(normalized.astype(dtype, copy=False))
            category_mass.append(masses)
            source_positions_all.append(source_positions)
            source_kinds_all.append(source_kinds)
            source_token_ids_all.append(source_token_ids)
            query_source_offsets.append(query_source_offsets[-1] + source_count)
            query_kinds.append(query_kind)
            query_positions.append(int(record["query_sequence_position"]))
            # Keep the Monet archive convention: output_index aligns answer
            # predictions, while latent queries use latent_index/position.
            query_output_indices.append(
                int(record.get("output_index", -1))
                if query_kind == QUERY_ANSWER else -1
            )
            query_predicted_ids.append(int(record.get("predicted_token_id", -1)))
            query_latent_indices.append(int(record.get("latent_index", -1)))

    total_sources = query_source_offsets[-1]
    if raw_blocks:
        raw_attention = np.concatenate(raw_blocks, axis=1)
        normalized_attention = np.concatenate(normalized_blocks, axis=1)
        mass_array = np.stack(category_mass, axis=0).astype(np.float32)
    else:
        raw_attention = np.empty((layer_count, 0), dtype=dtype)
        normalized_attention = np.empty((layer_count, 0), dtype=dtype)
        mass_array = np.empty(
            (0, layer_count, len(SOURCE_KIND_NAMES)), dtype=np.float32
        )
    if raw_attention.shape != (layer_count, total_sources):
        raise RuntimeError("Ragged attention assembly produced an invalid shape.")

    topk_records = capture["latent_topk"]
    if len(topk_records) != len(capture["latent_records"]):
        raise RuntimeError("Every latent query must have one output-head top-k row.")
    if topk_records:
        topk_token_ids = np.asarray(
            [record["token_ids"] for record in topk_records], dtype=np.int32
        )
        topk_logits = np.asarray(
            [record["logits"] for record in topk_records], dtype=np.float32
        )
    else:
        topk_token_ids = np.empty((0, LATENT_TOP_K), dtype=np.int32)
        topk_logits = np.empty((0, LATENT_TOP_K), dtype=np.float32)

    def concatenate(values: list[np.ndarray], output_dtype: Any) -> np.ndarray:
        return (
            np.concatenate(values).astype(output_dtype, copy=False)
            if values else np.empty(0, dtype=output_dtype)
        )

    return {
        "raw_attention": raw_attention,
        "group_normalized_attention": normalized_attention,
        "query_source_offsets": np.asarray(query_source_offsets, dtype=np.int64),
        "query_kind_codes": np.asarray(query_kinds, dtype=np.uint8),
        "query_sequence_positions": np.asarray(query_positions, dtype=np.int32),
        "query_output_indices": np.asarray(query_output_indices, dtype=np.int32),
        "query_predicted_token_ids": np.asarray(
            query_predicted_ids, dtype=np.int32
        ),
        "query_latent_indices": np.asarray(query_latent_indices, dtype=np.int32),
        "source_sequence_positions": concatenate(source_positions_all, np.int32),
        "source_kind_codes": concatenate(source_kinds_all, np.uint8),
        "source_token_ids": concatenate(source_token_ids_all, np.int32),
        "category_attention_mass": mass_array,
        "source_kind_names": SOURCE_KIND_NAMES,
        "query_kind_names": QUERY_KIND_NAMES,
        "layer_names": np.asarray(layer_names, dtype=np.str_),
        "latent_topk_token_ids": topk_token_ids,
        "latent_topk_logits": topk_logits,
        "latent_topk_sequence_positions": np.asarray([
            record["query_sequence_position"] for record in topk_records
        ], dtype=np.int32),
        "latent_topk_indices": np.asarray([
            record["latent_index"] for record in topk_records
        ], dtype=np.int32),
        "no_latent_fallback": np.asarray(bool(capture["no_latent_fallback"])),
    }


def token_piece(tokenizer: Any, token_id: int) -> str:
    if token_id < 0:
        return "<unknown>"
    return str(tokenizer.convert_ids_to_tokens(token_id)).replace("\n", "\\n")


def source_labels(data: dict[str, np.ndarray], tokenizer: Any) -> list[str]:
    labels = []
    for position, kind, token_id in zip(
        data["source_sequence_positions"],
        data["source_kind_codes"],
        data["source_token_ids"],
    ):
        if int(kind) == SOURCE_INPUT_VISUAL:
            label = f"visual@{int(position)}"
        elif int(kind) == SOURCE_LATENT:
            label = f"latent@{int(position)}"
        else:
            label = f"{token_piece(tokenizer, int(token_id))}@{int(position)}"
        labels.append(label)
    return labels


def query_labels(data: dict[str, np.ndarray], tokenizer: Any) -> list[str]:
    labels = []
    for kind, position, predicted, latent_index in zip(
        data["query_kind_codes"],
        data["query_sequence_positions"],
        data["query_predicted_token_ids"],
        data["query_latent_indices"],
    ):
        if int(kind) == QUERY_LATENT:
            labels.append(f"latent[{int(latent_index)}]@{int(position)}")
        else:
            labels.append(
                f"answer:{token_piece(tokenizer, int(predicted))}@{int(position)}"
            )
    return labels


def _ragged_heatmap(
    data: dict[str, np.ndarray],
    tokenizer: Any,
    query_kind: int,
    layer_index: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    indices = np.flatnonzero(data["query_kind_codes"] == query_kind)
    if len(indices) == 0:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            [],
            [],
        )
    offsets = data["query_source_offsets"]
    width = max(int(offsets[index + 1] - offsets[index]) for index in indices)
    raw = np.full((len(indices), width), np.nan, dtype=np.float32)
    normalized = np.full_like(raw, np.nan)
    all_source_labels = source_labels(data, tokenizer)
    all_query_labels = query_labels(data, tokenizer)
    selected_source_labels = [f"position@{index}" for index in range(width)]
    selected_query_labels = []
    for row_index, query_index in enumerate(indices):
        start, end = map(int, offsets[query_index:query_index + 2])
        length = end - start
        raw[row_index, :length] = data["raw_attention"][layer_index, start:end]
        normalized[row_index, :length] = data[
            "group_normalized_attention"
        ][layer_index, start:end]
        if length == width:
            selected_source_labels = all_source_labels[start:end]
        selected_query_labels.append(all_query_labels[query_index])
    return raw, normalized, selected_source_labels, selected_query_labels


def _tick_positions(count: int) -> np.ndarray:
    if count <= PLOT_MAX_TOKEN_LABELS:
        return np.arange(count)
    return np.unique(np.linspace(
        0, count - 1, PLOT_MAX_TOKEN_LABELS, dtype=int
    ))


def plot_attention_heatmap(
    data: dict[str, np.ndarray],
    tokenizer: Any,
    query_kind: int,
    plot_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    layer_count = len(data["layer_names"])
    layer_index = PLOT_LAYER if PLOT_LAYER >= 0 else layer_count + PLOT_LAYER
    if layer_index < 0 or layer_index >= layer_count:
        raise ValueError(f"PLOT_LAYER={PLOT_LAYER} is invalid for {layer_count} layers.")
    raw, normalized, xlabels, ylabels = _ragged_heatmap(
        data, tokenizer, query_kind, layer_index
    )
    if raw.size == 0:
        return
    height = max(4.5, min(30.0, len(ylabels) * PLOT_ROW_HEIGHT + 2.5))
    fig, axes = plt.subplots(
        2, 1, figsize=(PLOT_FIGURE_WIDTH, height * 2), constrained_layout=True
    )
    titles = ["raw head-mean attention", "target-group normalized attention"]
    for axis, matrix, title in zip(axes, [raw, normalized], titles):
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
        ticks = _tick_positions(len(xlabels))
        axis.set_xticks(ticks, [xlabels[i] for i in ticks], rotation=90, fontsize=6)
        axis.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=7)
        axis.set_title(
            f"{QUERY_KIND_NAMES[query_kind]} - layer {layer_index} - {title}"
        )
        axis.set_xlabel("causally visible source position")
        axis.set_ylabel("query")
        fig.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_category_attention(
    data: dict[str, np.ndarray], tokenizer: Any, plot_path: Path
) -> None:
    import matplotlib.pyplot as plt

    masses = data["category_attention_mass"]
    if len(masses) == 0:
        return
    layer_count = masses.shape[1]
    layer_index = PLOT_LAYER if PLOT_LAYER >= 0 else layer_count + PLOT_LAYER
    labels = query_labels(data, tokenizer)
    fig, axes = plt.subplots(
        2, 2, figsize=(PLOT_FIGURE_WIDTH, 11), constrained_layout=True
    )
    for column, query_kind in enumerate((QUERY_LATENT, QUERY_ANSWER)):
        indices = np.flatnonzero(data["query_kind_codes"] == query_kind)
        title = str(QUERY_KIND_NAMES[query_kind])
        if len(indices) == 0:
            axes[0, column].axis("off")
            axes[1, column].axis("off")
            continue
        query_matrix = masses[indices, layer_index, :]
        image = axes[0, column].imshow(
            query_matrix, aspect="auto", vmin=0.0, vmax=1.0
        )
        axes[0, column].set_xticks(
            np.arange(len(SOURCE_KIND_NAMES)), SOURCE_KIND_NAMES, rotation=30
        )
        axes[0, column].set_yticks(
            np.arange(len(indices)), [labels[index] for index in indices], fontsize=7
        )
        axes[0, column].set_title(f"{title}: per query at layer {layer_index}")
        fig.colorbar(image, ax=axes[0, column], fraction=0.03, pad=0.02)
        layer_matrix = masses[indices].mean(axis=0)
        image = axes[1, column].imshow(
            layer_matrix, aspect="auto", vmin=0.0, vmax=1.0
        )
        axes[1, column].set_xticks(
            np.arange(len(SOURCE_KIND_NAMES)), SOURCE_KIND_NAMES, rotation=30
        )
        axes[1, column].set_yticks(np.arange(layer_count))
        axes[1, column].set_title(f"{title}: query-mean mass by layer")
        axes[1, column].set_ylabel("decoder layer")
        fig.colorbar(image, ax=axes[1, column], fraction=0.03, pad=0.02)
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)


def decode_topk(data: dict[str, np.ndarray], tokenizer: Any) -> list[dict[str, Any]]:
    records = []
    for latent_index, position, token_ids, logits in zip(
        data["latent_topk_indices"],
        data["latent_topk_sequence_positions"],
        data["latent_topk_token_ids"],
        data["latent_topk_logits"],
    ):
        candidates = []
        for rank, (token_id, logit) in enumerate(zip(token_ids, logits), start=1):
            token_id = int(token_id)
            candidates.append({
                "rank": rank,
                "token_id": token_id,
                "token_piece": str(tokenizer.convert_ids_to_tokens(token_id)),
                "decoded_text": tokenizer.decode(
                    [token_id], skip_special_tokens=False
                ),
                "raw_logit": float(logit),
            })
        records.append({
            "latent_index": int(latent_index),
            "sequence_position": int(position),
            "candidates": candidates,
        })
    return records


def category_attention_csv_fieldnames() -> list[str]:
    return [
        "sample_ordinal", "dataset_ordinal", "dataset_index", "request_id",
        "query_ordinal", "query_kind", "query_sequence_position",
        "query_output_index", "query_predicted_token_id",
        "query_predicted_text", "query_latent_index", "layer_index",
        "layer_name", *SOURCE_KIND_NAMES.tolist(),
    ]


def latent_topk_csv_fieldnames(top_k: int = LATENT_TOP_K) -> list[str]:
    fields = [
        "sample_ordinal", "dataset_ordinal", "dataset_index", "request_id",
        "latent_ordinal", "latent_index", "sequence_position",
    ]
    for rank in range(1, top_k + 1):
        fields.extend([
            f"top{rank}_text", f"top{rank}_token_id", f"top{rank}_raw_logit",
        ])
    return fields


def build_category_attention_csv_rows(
    data: dict[str, np.ndarray],
    tokenizer: Any,
    *,
    sample_ordinal: int,
    dataset_ordinal: int,
    dataset_index: Any,
    request_id: str,
) -> list[dict[str, Any]]:
    masses = data["category_attention_mass"]
    layer_names = data["layer_names"]
    expected = (
        len(data["query_kind_codes"]), len(layer_names), len(SOURCE_KIND_NAMES)
    )
    if masses.shape != expected:
        raise ValueError(f"Unexpected category_attention_mass shape: {masses.shape}")
    rows = []
    for query_ordinal in range(len(data["query_kind_codes"])):
        query_kind = int(data["query_kind_codes"][query_ordinal])
        predicted_token_id = int(
            data["query_predicted_token_ids"][query_ordinal]
        )
        predicted_text = (
            tokenizer.decode([predicted_token_id], skip_special_tokens=False)
            if predicted_token_id >= 0 else ""
        )
        common = {
            "sample_ordinal": sample_ordinal,
            "dataset_ordinal": int(dataset_ordinal),
            "dataset_index": dataset_index,
            "request_id": request_id,
            "query_ordinal": query_ordinal,
            "query_kind": str(QUERY_KIND_NAMES[query_kind]),
            "query_sequence_position": int(
                data["query_sequence_positions"][query_ordinal]
            ),
            "query_output_index": int(data["query_output_indices"][query_ordinal]),
            "query_predicted_token_id": predicted_token_id,
            "query_predicted_text": predicted_text,
            "query_latent_index": int(data["query_latent_indices"][query_ordinal]),
        }
        for layer_index, layer_name in enumerate(layer_names):
            row = {
                **common,
                "layer_index": layer_index,
                "layer_name": str(layer_name),
            }
            for kind_index, kind_name in enumerate(SOURCE_KIND_NAMES):
                row[str(kind_name)] = float(
                    masses[query_ordinal, layer_index, kind_index]
                )
            rows.append(row)
    return rows


def build_latent_topk_csv_rows(
    decoded_topk: list[dict[str, Any]],
    *,
    sample_ordinal: int,
    dataset_ordinal: int,
    dataset_index: Any,
    request_id: str,
    top_k: int = LATENT_TOP_K,
) -> list[dict[str, Any]]:
    rows = []
    for latent_ordinal, latent_record in enumerate(decoded_topk):
        candidates = latent_record["candidates"]
        if len(candidates) != top_k:
            raise ValueError(f"Expected {top_k} top-k candidates, received {len(candidates)}.")
        row = {
            "sample_ordinal": sample_ordinal,
            "dataset_ordinal": int(dataset_ordinal),
            "dataset_index": dataset_index,
            "request_id": request_id,
            "latent_ordinal": latent_ordinal,
            "latent_index": int(latent_record["latent_index"]),
            "sequence_position": int(latent_record["sequence_position"]),
        }
        for rank, candidate in enumerate(candidates, start=1):
            row[f"top{rank}_text"] = candidate["decoded_text"]
            row[f"top{rank}_token_id"] = int(candidate["token_id"])
            row[f"top{rank}_raw_logit"] = float(candidate["raw_logit"])
        rows.append(row)
    return rows


def write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def clean_lvr_text(text: str) -> str:
    return LVR_TOKEN_PATTERN.sub(CLEAN_LATENT_MARKER, text)


def global_config_snapshot() -> dict[str, Any]:
    names = [
        "MODEL_PATH", "HRBENCH_PATH", "OUTPUT_DIR", "RESULTS_FILE",
        "RUN_CONFIG_FILE", "CATEGORY_ATTENTION_CSV_FILE",
        "LATENT_TOPK_CSV_FILE", "ATTENTION_SUBDIR", "PLOT_SUBDIR",
        "SELECTION_MODE", "START_INDEX", "NUM_SAMPLES", "RANDOM_SEED",
        "DEVICE", "TORCH_DTYPE", "TRUST_REMOTE_CODE", "USE_CACHE",
        "MIN_PIXELS", "MAX_PIXELS", "MAX_OUTPUT_TOKENS", "TEMPERATURE",
        "TOP_K", "TOP_P", "REPETITION_PENALTY", "BEST_OF", "STOP",
        "LATENT_TOP_K", "ATTENTION_STORAGE_DTYPE", "ANSWER_SCOPE",
        "LATENT_TOPK_SOURCE", "CLEAN_LATENT_MARKER", "PLOT_LAYER",
        "PLOT_DPI", "PLOT_MAX_TOKEN_LABELS", "PLOT_FIGURE_WIDTH",
        "PLOT_ROW_HEIGHT",
    ]
    return {name: globals()[name] for name in names}


def _process_one_sample(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    *,
    sample_ordinal: int,
    dataset_ordinal: int,
    dataset_dir: Path,
    output_path: Path,
    layer_names: list[str],
    latent_token_ids: set[int],
    special_token_ids: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    build_question, decode_hrbench_image = _import_pca_helpers()
    image = decode_hrbench_image(row["image"], dataset_dir)
    request_id = f"hf-{sample_ordinal:06d}"
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_question(row)},
            ],
        }]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt], images=[image], padding=False, return_tensors="pt"
        )
        input_device = model.get_input_embeddings().weight.device
        inputs = _move_processor_inputs(inputs, input_device)
        prompt_ids = inputs["input_ids"][0]
        prompt_length = int(prompt_ids.shape[0])
        image_token_id = int(model.config.image_token_id)
        image_positions = set(
            int(value)
            for value in (prompt_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()
        )
        if not image_positions:
            raise RuntimeError("The processed multimodal prompt has no image tokens.")
        if max(image_positions) >= prompt_length - 1:
            raise RuntimeError(
                "The final prompt token overlaps the image span, so the "
                "memory-safe split-prefill replay cannot be used."
            )

        with torch.inference_mode():
            generated = model.generate(**inputs, **_generation_kwargs(processor.tokenizer))
            full_ids = generated[0]
            output_ids = [int(value) for value in full_ids[prompt_length:].tolist()]
            capture = replay_generated_sequence(
                model,
                inputs,
                output_ids,
                layer_count=len(layer_names),
                latent_token_ids=latent_token_ids,
                special_token_ids=special_token_ids,
            )
        capture["image_positions"] = sorted(image_positions)
        data = assemble_sample_archive(
            capture,
            image_positions=image_positions,
            special_token_ids=special_token_ids,
            layer_names=layer_names,
        )
        stem = f"sample_{sample_ordinal:06d}"
        archive_rel = Path(ATTENTION_SUBDIR) / f"{stem}.npz"
        np.savez_compressed(output_path / archive_rel, **data)

        latent_plot_rel = Path(PLOT_SUBDIR) / f"{stem}_latent_attention.png"
        answer_plot_rel = Path(PLOT_SUBDIR) / f"{stem}_answer_attention.png"
        category_plot_rel = Path(PLOT_SUBDIR) / f"{stem}_category_attention.png"
        plot_attention_heatmap(
            data, processor.tokenizer, QUERY_LATENT, output_path / latent_plot_rel
        )
        plot_attention_heatmap(
            data, processor.tokenizer, QUERY_ANSWER, output_path / answer_plot_rel
        )
        plot_category_attention(
            data, processor.tokenizer, output_path / category_plot_rel
        )

        raw_output_text = processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        answer_indices, no_latent_fallback = select_answer_token_indices(
            output_ids, latent_token_ids, special_token_ids
        )
        answer_ids = [output_ids[index] for index in answer_indices]
        decoded_topk = decode_topk(data, processor.tokenizer)
        category_rows = build_category_attention_csv_rows(
            data,
            processor.tokenizer,
            sample_ordinal=sample_ordinal,
            dataset_ordinal=dataset_ordinal,
            dataset_index=_json_compatible(row["index"]),
            request_id=request_id,
        )
        topk_rows = build_latent_topk_csv_rows(
            decoded_topk,
            sample_ordinal=sample_ordinal,
            dataset_ordinal=dataset_ordinal,
            dataset_index=_json_compatible(row["index"]),
            request_id=request_id,
        )
        result = {
            "sample_ordinal": sample_ordinal,
            "dataset_ordinal": int(dataset_ordinal),
            "dataset_index": _json_compatible(row["index"]),
            "question": _json_compatible(row["question"]),
            "answer": _json_compatible(row["answer"]),
            "category": _json_compatible(row["category"]),
            "cycle_category": _json_compatible(row["cycle_category"]),
            "choices": {
                letter: _json_compatible(row[letter]) for letter in "ABCD"
            },
            "request_id": request_id,
            "raw_output_text": raw_output_text,
            "cleaned_output_text": clean_lvr_text(raw_output_text),
            "output_token_ids": output_ids,
            "finish_reason": _finish_reason(output_ids, processor.tokenizer),
            "no_latent_fallback": no_latent_fallback,
            "answer_output_indices": answer_indices,
            "answer_token_ids": answer_ids,
            "answer_text": processor.tokenizer.decode(
                answer_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "latent_output_head_topk": decoded_topk,
            "attention_archive": str(archive_rel),
            "plots": {
                "latent_attention": (
                    str(latent_plot_rel)
                    if (output_path / latent_plot_rel).exists() else None
                ),
                "answer_attention": (
                    str(answer_plot_rel)
                    if (output_path / answer_plot_rel).exists() else None
                ),
                "category_attention": (
                    str(category_plot_rel)
                    if (output_path / category_plot_rel).exists() else None
                ),
            },
        }
        statistics = {
            "sample_ordinal": sample_ordinal,
            "prompt_tokens": prompt_length,
            "generated_tokens": len(output_ids),
            "latent_queries": len(capture["latent_records"]),
            "answer_queries": len(capture["answer_records"]),
            "ragged_source_entries": int(data["query_source_offsets"][-1]),
        }
        return result, category_rows, topk_rows, statistics
    finally:
        image.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    model_path, dataset_path, output_path = validate_configuration()
    rows, selected_indices = load_hrbench_rows(dataset_path)
    _set_random_seed(RANDOM_SEED)
    model, processor, model_details = load_model_and_processor(model_path)
    layer_names = list(model_details["decoder_layer_names"])
    latent_token_ids = {
        int(value) for value in model_details["lvr_token_ids"].keys()
    }
    special_token_ids = {
        int(value) for value in processor.tokenizer.all_special_ids
    }
    special_token_ids.update(latent_token_ids)

    results = []
    category_rows = []
    topk_rows = []
    capture_statistics = []
    for sample_ordinal, (row, dataset_ordinal) in enumerate(
        zip(rows, selected_indices)
    ):
        result, sample_category_rows, sample_topk_rows, statistics = (
            _process_one_sample(
                model,
                processor,
                row,
                sample_ordinal=sample_ordinal,
                dataset_ordinal=int(dataset_ordinal),
                dataset_dir=dataset_path.parent,
                output_path=output_path,
                layer_names=layer_names,
                latent_token_ids=latent_token_ids,
                special_token_ids=special_token_ids,
            )
        )
        results.append(result)
        category_rows.extend(sample_category_rows)
        topk_rows.extend(sample_topk_rows)
        capture_statistics.append(statistics)
        print(
            f"[{sample_ordinal + 1}/{len(rows)}] dataset={dataset_ordinal}, "
            f"latent={statistics['latent_queries']}, "
            f"answer={statistics['answer_queries']}, "
            f"output={statistics['generated_tokens']}"
        )

    write_jsonl(output_path / RESULTS_FILE, results)
    write_csv(
        output_path / CATEGORY_ATTENTION_CSV_FILE,
        category_attention_csv_fieldnames(),
        category_rows,
    )
    write_csv(
        output_path / LATENT_TOPK_CSV_FILE,
        latent_topk_csv_fieldnames(),
        topk_rows,
    )
    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "huggingface",
        "config": global_config_snapshot(),
        "selected_dataset_ordinals": selected_indices,
        "model_details": model_details,
        "capture_statistics": capture_statistics,
        "schema": {
            "source_kind_names": SOURCE_KIND_NAMES.tolist(),
            "query_kind_names": QUERY_KIND_NAMES.tolist(),
            "raw_attention": "[decoder_layer, concatenated_ragged_source_entry]",
            "group_normalized_attention": (
                "same layout; latent targets input_text+input_visual; answer "
                "targets input_text+input_visual+latent"
            ),
            "category_attention_mass": "[query, decoder_layer, source_kind]",
            "query_source_offsets": "offsets into concatenated source arrays",
            "answer_alignment": "query that generated the recorded token",
            "answer_scope": "all generated non-latent, non-special tokens",
            "latent_topk_source": (
                "contextual latent hidden state after decoder/final norm"
            ),
        },
        "outputs": {
            "results": RESULTS_FILE,
            "category_attention_csv": CATEGORY_ATTENTION_CSV_FILE,
            "latent_topk_csv": LATENT_TOPK_CSV_FILE,
            "attention_directory": ATTENTION_SUBDIR,
            "plot_directory": PLOT_SUBDIR,
        },
        "note": (
            "Sampling parameters match the Monet reference, but Hugging Face "
            "and vLLM random streams are not byte-identical."
        ),
    }
    (output_path / RUN_CONFIG_FILE).write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Attention analysis complete: {output_path}")


if __name__ == "__main__":
    main()
