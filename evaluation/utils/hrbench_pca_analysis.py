"""Run LaViT on local HR-Bench data and export joint PCA coordinates.

Edit the global variables in the configuration section below, then run:

    python evaluation/utils/hrbench_pca_analysis.py

There is deliberately no command-line interface.  The saved ``.npz`` files
follow the schema used by Monet's ``plot_hrbench_joint_pca.py``.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


# =============================================================================
# Global configuration -- edit values here; no CLI arguments are used
# =============================================================================

MODEL_PATH = "/home/fit/renjujty/WORK/jty/lmllms/lavit/"
HRBENCH_PATH = "/home/fit/renjujty/WORK/jty/lmllms/hrbench/hr_bench_4k.parquet"
OUTPUT_DIR = "outputs/hrbench_pca"

JOINT_PCA_FILE = "joint_pca_3d.npz"
LATENT_TRAJECTORY_FILE = "latent_trajectories.npz"
RESULTS_FILE = "results.jsonl"
RUN_CONFIG_FILE = "run_config.json"

# "sequential": START_INDEX ... START_INDEX + NUM_SAMPLES
# "random": deterministic sampling without replacement using RANDOM_SEED
SELECTION_MODE = "random"
START_INDEX = 0
NUM_SAMPLES = 800
RANDOM_SEED = 0

DEVICE = "cuda"
TORCH_DTYPE = "bfloat16"
TRUST_REMOTE_CODE = True

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 8192 * 28 * 28
MAX_NEW_TOKENS = 512
USE_CACHE = True

VOCAB_EMBEDDING_BATCH_SIZE = 8192
PCA_TRANSFORM_BATCH_SIZE = 8192

# Keep captures only when an error occurs and manual diagnosis is useful.
KEEP_TEMP_CAPTURE_ON_ERROR = False


KIND_NAMES = np.asarray(
    ["vocabulary_embedding", "image_feature", "latent"], dtype=np.str_
)
REQUIRED_COLUMNS = {
    "index",
    "question",
    "answer",
    "category",
    "A",
    "B",
    "C",
    "D",
    "cycle_category",
    "image",
}
MAX_PATH_CANDIDATE_LENGTH = 4096
LVR_TOKEN_PATTERN = re.compile(r"<lvr\d*>", flags=re.IGNORECASE)


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
    if count <= 0:
        raise ValueError("NUM_SAMPLES must be positive.")
    if count > total:
        raise ValueError(f"NUM_SAMPLES={count} exceeds dataset size {total}.")
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
    """Validate global options and return resolved paths."""
    if VOCAB_EMBEDDING_BATCH_SIZE <= 0 or PCA_TRANSFORM_BATCH_SIZE <= 0:
        raise ValueError("Vocabulary and PCA batch sizes must be positive.")
    if MAX_NEW_TOKENS <= 0:
        raise ValueError("MAX_NEW_TOKENS must be positive.")
    if MIN_PIXELS <= 0 or MAX_PIXELS < MIN_PIXELS:
        raise ValueError("MIN_PIXELS/MAX_PIXELS are invalid.")

    model_path = Path(MODEL_PATH).expanduser()
    dataset_path = Path(HRBENCH_PATH).expanduser()
    output_path = Path(OUTPUT_DIR).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {model_path}. Edit the global variable."
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"HRBENCH_PATH does not exist: {dataset_path}. Edit the global variable."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return model_path.resolve(), dataset_path.resolve(), output_path.resolve()


def load_hrbench_rows(
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Load only the selected rows from a local parquet file."""
    import pandas as pd

    dataframe = pd.read_parquet(dataset_path)
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(
            "Unexpected HR-Bench schema; missing columns: " + ", ".join(sorted(missing))
        )
    selected = select_sample_indices(len(dataframe))
    rows = [dict(dataframe.iloc[index]) for index in selected]
    return rows, selected


def _open_image_path(path_value: str, dataset_dir: Path) -> Image.Image | None:
    if not path_value or "\x00" in path_value:
        return None
    try:
        possible_path = Path(path_value).expanduser()
        if not possible_path.is_absolute():
            possible_path = dataset_dir / possible_path
        if not possible_path.is_file():
            return None
    except (OSError, RuntimeError):
        return None
    try:
        return Image.open(possible_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to open image file: {possible_path}") from exc


def _open_image_bytes(image_bytes: bytes, description: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to decode {description} as an image.") from exc


def _decode_base64_image(encoded: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(encoded, validate=False)
    except Exception as exc:
        raise ValueError("The HR-Bench image contains invalid base64 data.") from exc
    return _open_image_bytes(image_bytes, "HR-Bench base64 data")


def decode_hrbench_image(value: Any, dataset_dir: Path) -> Image.Image:
    """Decode all image representations used by published HR-Bench parquet files."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _open_image_bytes(value["bytes"], "HR-Bench byte data")
        if value.get("path"):
            image = _open_image_path(str(value["path"]), dataset_dir)
            if image is not None:
                return image
            raise ValueError(f"Image path does not exist: {value['path']}")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _open_image_bytes(bytes(value), "HR-Bench byte data")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported HR-Bench image value: {type(value)!r}")

    value = value.strip()
    if value.startswith("data:image/"):
        if "," not in value:
            raise ValueError("Malformed image data URI: missing comma separator.")
        return _decode_base64_image(value.split(",", 1)[1])
    if len(value) <= MAX_PATH_CANDIDATE_LENGTH:
        image = _open_image_path(value, dataset_dir)
        if image is not None:
            return image
    return _decode_base64_image(value)


def build_question(row: dict[str, Any]) -> str:
    choices = "\n".join(f"({letter}) {row[letter]}" for letter in "ABCD")
    return (
        f"Question: {row['question']} The choices are listed below:\n"
        f"{choices}\nPut your final answer in \\boxed{{}}."
    )


def _get_dotted_attribute(root: Any, dotted_name: str) -> Any | None:
    value = root
    for component in dotted_name.split("."):
        value = getattr(value, component, None)
        if value is None:
            return None
    return value


def locate_capture_modules(model: Any) -> tuple[Any, str, Any, str]:
    """Locate Qwen2.5-VL's visual tower and final text normalization module."""
    import torch.nn as nn

    visual_candidates = (
        "model.visual",
        "visual",
    )
    norm_candidates = (
        "model.language_model.norm",
        "model.language_model.model.norm",
        "model.norm",
        "language_model.model.norm",
        "language_model.norm",
    )

    visual = None
    visual_name = ""
    for candidate in visual_candidates:
        module = _get_dotted_attribute(model, candidate)
        if isinstance(module, nn.Module):
            visual, visual_name = module, candidate
            break
    final_norm = None
    norm_name = ""
    for candidate in norm_candidates:
        module = _get_dotted_attribute(model, candidate)
        if isinstance(module, nn.Module):
            final_norm, norm_name = module, candidate
            break
    if visual is None:
        raise RuntimeError(
            "Could not locate the Qwen2.5-VL visual module. Checked: "
            + ", ".join(visual_candidates)
        )
    if final_norm is None:
        raise RuntimeError(
            "Could not locate the language model final norm. Checked: "
            + ", ".join(norm_candidates)
        )
    return visual, visual_name, final_norm, norm_name


def _iter_tensors(value: Any) -> Iterable[Any]:
    """Yield tensors recursively without depending on a ModelOutput version."""
    import torch

    if isinstance(value, torch.Tensor):
        yield value
        return
    preferred_names = ("pooler_output", "last_hidden_state")
    for name in preferred_names:
        tensor = getattr(value, name, None)
        if isinstance(tensor, torch.Tensor):
            yield tensor
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            yield from _iter_tensors(nested)


def _select_feature_tensor(output: Any, hidden_size: int, label: str) -> Any:
    """Select a [tokens, hidden] compatible tensor from a hooked output."""
    candidates = []
    seen = set()
    for tensor in _iter_tensors(output):
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        if tensor.ndim in (2, 3) and int(tensor.shape[-1]) == hidden_size:
            candidates.append(tensor)
    if not candidates:
        shapes = [tuple(tensor.shape) for tensor in _iter_tensors(output)]
        raise RuntimeError(
            f"{label} hook returned no tensor with hidden size {hidden_size}. "
            f"Observed tensor shapes: {shapes}"
        )
    # ``_iter_tensors`` deliberately yields pooler_output before auxiliary
    # tensors. Preserve that semantic ordering while preferring the normal 2D
    # merged-vision representation over a 3D batch wrapper.
    for tensor in candidates:
        if tensor.ndim == 2:
            return tensor
    return candidates[0]


class GenerationCapture:
    """Capture merged image inputs and consumed-token final hidden states."""

    def __init__(self, visual: Any, final_norm: Any, hidden_size: int):
        self.visual = visual
        self.final_norm = final_norm
        self.hidden_size = hidden_size
        self.image_features: list[np.ndarray] = []
        self.decoder_last_states: list[np.ndarray] = []
        self._handles: list[Any] = []

    def _visual_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        import torch

        features = _select_feature_tensor(output, self.hidden_size, "Visual")
        if features.ndim == 3:
            if int(features.shape[0]) != 1:
                raise RuntimeError(
                    "Analysis expects one inference sample at a time, but the "
                    f"visual hook returned shape {tuple(features.shape)}."
                )
            features = features[0]
        self.image_features.append(
            features.detach().to(device="cpu", dtype=torch.float16).numpy()
        )

    def _norm_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        import torch

        hidden = _select_feature_tensor(output, self.hidden_size, "Final norm")
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if int(hidden.shape[0]) != 1:
            raise RuntimeError(
                "Analysis expects batch size 1, but the final norm returned "
                f"shape {tuple(hidden.shape)}."
            )
        self.decoder_last_states.append(
            hidden[0, -1].detach().to(device="cpu", dtype=torch.float16).numpy()
        )

    def __enter__(self) -> "GenerationCapture":
        self._handles = [
            self.visual.register_forward_hook(self._visual_hook),
            self.final_norm.register_forward_hook(self._norm_hook),
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def merged_image_features(self) -> np.ndarray:
        if not self.image_features:
            raise RuntimeError("The visual hook did not capture any image features.")
        if len(self.image_features) != 1:
            raise RuntimeError(
                "The visual module ran more than once for a single generation "
                f"request ({len(self.image_features)} calls). Refusing to "
                "silently duplicate image features."
            )
        return self.image_features[0]

    def consumed_output_states(self) -> np.ndarray:
        if not self.decoder_last_states:
            raise RuntimeError("The final norm hook did not run during generation.")
        # The first norm call is prompt prefill. Each later call consumes the
        # previously sampled output token; the final sampled token has no state.
        if len(self.decoder_last_states) == 1:
            return np.empty((0, self.hidden_size), dtype=np.float16)
        return np.stack(self.decoder_last_states[1:], axis=0)


def discover_lvr_token_ids(tokenizer: Any, vocab_size: int) -> dict[int, str]:
    """Return all single-token vocabulary entries named <lvr>, <lvr1>, ... ."""
    found: dict[int, str] = {}
    for token, token_id in tokenizer.get_vocab().items():
        if LVR_TOKEN_PATTERN.fullmatch(str(token)):
            token_id = int(token_id)
            if 0 <= token_id < vocab_size:
                found[token_id] = str(token)
    if not found:
        raise RuntimeError(
            "No <lvr*> token was found in the checkpoint tokenizer. Do not add "
            "new tokens for analysis; use the tokenizer saved with the trained "
            "LaViT checkpoint."
        )
    return dict(sorted(found.items()))


def _torch_dtype_from_name(name: str) -> Any:
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


def load_model_and_processor(model_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load LaViT exactly through the repository's Hugging Face model class."""
    import torch
    from transformers import Qwen2VLProcessor

    project_root = Path(__file__).resolve().parents[2]
    training_src = project_root / "training" / "src"
    if str(training_src) not in sys.path:
        sys.path.insert(0, str(training_src))
    from modeling_lavit import LaViTConfig, LaViTQwen2VL

    dtype = _torch_dtype_from_name(TORCH_DTYPE)
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE={DEVICE!r}, but CUDA is not available.")

    processor = Qwen2VLProcessor.from_pretrained(
        str(model_path),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    config = LaViTConfig.from_pretrained(
        str(model_path), trust_remote_code=TRUST_REMOTE_CODE
    )
    model = LaViTQwen2VL.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=dtype,
        device_map=DEVICE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model.eval()

    embedding = model.get_input_embeddings()
    vocab_size, hidden_size = map(int, embedding.weight.shape)
    lvr_tokens = discover_lvr_token_ids(processor.tokenizer, vocab_size)
    visual, visual_name, final_norm, norm_name = locate_capture_modules(model)
    details = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "embedding_dtype": str(embedding.weight.dtype),
        "embedding_device": str(embedding.weight.device),
        "visual_module": visual_name,
        "final_norm_module": norm_name,
        "lvr_token_ids": {str(key): value for key, value in lvr_tokens.items()},
    }
    print("[LaViT analysis] model capture configuration:")
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return model, processor, details


def export_vocabulary_embeddings(
    model: Any, temporary_dir: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """Copy the complete input embedding table to a float16 disk memmap."""
    import torch

    weight = model.get_input_embeddings().weight
    vocab_size, hidden_size = map(int, weight.shape)
    path = temporary_dir / "vocabulary_embeddings.npy"
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=(vocab_size, hidden_size),
    )
    with torch.inference_mode():
        for start in range(0, vocab_size, VOCAB_EMBEDDING_BATCH_SIZE):
            end = min(start + VOCAB_EMBEDDING_BATCH_SIZE, vocab_size)
            output[start:end] = (
                weight[start:end].detach().to(device="cpu", dtype=torch.float16).numpy()
            )
    output.flush()
    del output
    return np.load(path, mmap_mode="r"), {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "dtype": "float16",
        "batch_size": VOCAB_EMBEDDING_BATCH_SIZE,
    }


def _move_processor_inputs(inputs: Any, device: Any) -> Any:
    try:
        return inputs.to(device)
    except TypeError:
        return {key: value.to(device) for key, value in inputs.items()}


def run_inference(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    selected_indices: list[int],
    dataset_dir: Path,
    capture_dir: Path,
    model_details: dict[str, Any],
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    """Generate one sample at a time and spool image/latent captures to disk."""
    import torch

    hidden_size = int(model_details["hidden_size"])
    vocab_size = int(model_details["vocab_size"])
    lvr_token_ids = discover_lvr_token_ids(processor.tokenizer, vocab_size)
    visual, _, final_norm, _ = locate_capture_modules(model)
    input_device = model.get_input_embeddings().weight.device
    image_token_id = int(model.config.image_token_id)

    capture_paths = []
    sample_records = []
    per_sample_image_counts = []
    per_sample_latent_counts = []

    for sample_ordinal, (row, dataset_ordinal) in enumerate(
        zip(rows, selected_indices)
    ):
        image = decode_hrbench_image(row["image"], dataset_dir)
        request_id = f"hf-{sample_ordinal:06d}"
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": build_question(row)},
                    ],
                }
            ]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_input],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = _move_processor_inputs(inputs, input_device)
            prompt_ids = inputs["input_ids"][0]
            prompt_length = int(prompt_ids.shape[0])
            image_positions = (
                (prompt_ids == image_token_id)
                .nonzero(as_tuple=True)[0]
                .to(device="cpu", dtype=torch.int64)
                .numpy()
                .astype(np.int32, copy=False)
            )

            with (
                torch.inference_mode(),
                GenerationCapture(visual, final_norm, hidden_size) as capture,
            ):
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=USE_CACHE,
                )

            full_ids = generated[0]
            output_ids = (
                full_ids[prompt_length:]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .numpy()
            )
            image_vectors = capture.merged_image_features()
            consumed_states = capture.consumed_output_states()
            if len(image_vectors) != len(image_positions):
                raise RuntimeError(
                    "Image feature/token count mismatch for sample "
                    f"{sample_ordinal}: {len(image_vectors)} != "
                    f"{len(image_positions)}."
                )
            consumed_count = min(len(consumed_states), len(output_ids))
            latent_output_positions = np.asarray(
                [
                    index
                    for index, token_id in enumerate(output_ids[:consumed_count])
                    if int(token_id) in lvr_token_ids
                ],
                dtype=np.int32,
            )
            if len(latent_output_positions):
                latent_vectors = consumed_states[latent_output_positions]
            else:
                latent_vectors = np.empty((0, hidden_size), dtype=np.float16)
            latent_indices = np.arange(len(latent_vectors), dtype=np.int32)
            latent_sequence_positions = (
                latent_output_positions + prompt_length
            ).astype(np.int32, copy=False)

            capture_path = capture_dir / f"capture_{sample_ordinal:06d}.npz"
            np.savez(
                capture_path,
                image_vectors=image_vectors.astype(np.float16, copy=False),
                image_sequence_positions=image_positions,
                latent_vectors=latent_vectors.astype(np.float16, copy=False),
                latent_sequence_positions=latent_sequence_positions,
                latent_generation_steps=latent_output_positions,
                latent_indices=latent_indices,
            )
            capture_paths.append(capture_path)
            per_sample_image_counts.append(int(len(image_vectors)))
            per_sample_latent_counts.append(int(len(latent_vectors)))

            decoded = processor.batch_decode(
                [output_ids.tolist()],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            sample_records.append(
                {
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
                    "raw_output_text": decoded,
                    "output_token_ids": [int(value) for value in output_ids],
                    "prompt_token_count": prompt_length,
                    "consumed_output_token_count": consumed_count,
                    "unconsumed_output_token_ids": [
                        int(value) for value in output_ids[consumed_count:]
                    ],
                    "capture_counts": {
                        "image_feature": int(len(image_vectors)),
                        "latent": int(len(latent_vectors)),
                    },
                    "latent_generation_steps": latent_output_positions.tolist(),
                }
            )
            print(
                f"[{sample_ordinal + 1}/{len(rows)}] "
                f"dataset={dataset_ordinal}, image={len(image_vectors)}, "
                f"latent={len(latent_vectors)}, output={len(output_ids)}"
            )
        finally:
            image.close()
        del inputs, generated, full_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        capture_paths,
        sample_records,
        {
            "per_sample_image_counts": per_sample_image_counts,
            "per_sample_latent_counts": per_sample_latent_counts,
        },
    )


def sample_image_positions(
    available_count: int, target_count: int
) -> tuple[np.ndarray, bool]:
    if available_count <= 0:
        raise ValueError("No image features were captured for PCA.")
    if target_count <= 0:
        raise ValueError("The image feature sample size must be positive.")
    replace = available_count < target_count
    rng = np.random.default_rng(RANDOM_SEED)
    positions = rng.choice(available_count, size=target_count, replace=replace)
    return np.sort(positions.astype(np.int64, copy=False)), replace


def extract_sampled_images_and_latents(
    capture_paths: list[Path],
    target_image_count: int,
    hidden_size: int,
    temporary_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Sample global image positions and concatenate every latent trajectory."""
    image_counts = np.zeros(len(capture_paths), dtype=np.int64)
    latent_counts = np.zeros(len(capture_paths), dtype=np.int64)
    for sample_ordinal, path in enumerate(capture_paths):
        with np.load(path, allow_pickle=False) as data:
            image_counts[sample_ordinal] = len(data["image_vectors"])
            latent_counts[sample_ordinal] = len(data["latent_vectors"])

    total_images = int(image_counts.sum())
    total_latents = int(latent_counts.sum())
    chosen_images, used_replacement = sample_image_positions(
        total_images, target_image_count
    )

    image_path = temporary_dir / "sampled_image_embeddings.npy"
    image_vectors = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.float16,
        shape=(target_image_count, hidden_size),
    )
    latent_vectors = np.empty((total_latents, hidden_size), dtype=np.float16)
    image_sample_ordinals = np.empty(target_image_count, dtype=np.int32)
    image_sequence_positions = np.empty(target_image_count, dtype=np.int32)
    image_generation_steps = np.full(target_image_count, -1, dtype=np.int32)
    latent_sample_ordinals = np.empty(total_latents, dtype=np.int32)
    latent_sequence_positions = np.empty(total_latents, dtype=np.int32)
    latent_generation_steps = np.empty(total_latents, dtype=np.int32)
    latent_indices = np.empty(total_latents, dtype=np.int32)
    latent_trajectory_steps = np.empty(total_latents, dtype=np.int32)

    image_global_start = 0
    latent_output_start = 0
    for sample_ordinal, path in enumerate(capture_paths):
        with np.load(path, allow_pickle=False) as data:
            captured_images = data["image_vectors"]
            captured_latents = data["latent_vectors"]
            if captured_images.ndim != 2 or captured_images.shape[1] != hidden_size:
                raise RuntimeError(
                    f"Image hidden-size mismatch in {path}: {captured_images.shape}."
                )
            if captured_latents.ndim != 2 or captured_latents.shape[1] != hidden_size:
                raise RuntimeError(
                    f"Latent hidden-size mismatch in {path}: {captured_latents.shape}."
                )

            image_global_end = image_global_start + len(captured_images)
            selected_start = np.searchsorted(
                chosen_images, image_global_start, side="left"
            )
            selected_end = np.searchsorted(chosen_images, image_global_end, side="left")
            if selected_end > selected_start:
                local_positions = (
                    chosen_images[selected_start:selected_end] - image_global_start
                )
                image_vectors[selected_start:selected_end] = captured_images[
                    local_positions
                ]
                image_sample_ordinals[selected_start:selected_end] = sample_ordinal
                image_sequence_positions[selected_start:selected_end] = data[
                    "image_sequence_positions"
                ][local_positions]
            image_global_start = image_global_end

            latent_output_end = latent_output_start + len(captured_latents)
            if latent_output_end > latent_output_start:
                output_slice = slice(latent_output_start, latent_output_end)
                latent_vectors[output_slice] = captured_latents
                latent_sample_ordinals[output_slice] = sample_ordinal
                latent_sequence_positions[output_slice] = data[
                    "latent_sequence_positions"
                ]
                latent_generation_steps[output_slice] = data["latent_generation_steps"]
                latent_indices[output_slice] = data["latent_indices"]
                latent_trajectory_steps[output_slice] = np.arange(
                    len(captured_latents), dtype=np.int32
                )
            latent_output_start = latent_output_end

    image_vectors.flush()
    latent_offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(latent_counts, dtype=np.int64),
        )
    )
    metadata = {
        "image_sample_ordinal": image_sample_ordinals,
        "image_sequence_positions": image_sequence_positions,
        "image_generation_steps": image_generation_steps,
        "latent_sample_ordinal": latent_sample_ordinals,
        "latent_sequence_positions": latent_sequence_positions,
        "latent_generation_steps": latent_generation_steps,
        "latent_indices": latent_indices,
        "latent_trajectory_steps": latent_trajectory_steps,
        "latent_sample_offsets": latent_offsets,
    }
    statistics = {
        "available_image_features": total_images,
        "sampled_image_features": target_image_count,
        "image_sampling_with_replacement": used_replacement,
        "latent_vectors": total_latents,
        "per_sample_image_counts": image_counts.tolist(),
        "per_sample_latent_counts": latent_counts.tolist(),
    }
    return image_vectors, latent_vectors, metadata, statistics


def _copy_float32_blocks(destination: Any, start: int, source: np.ndarray) -> int:
    for source_start in range(0, len(source), PCA_TRANSFORM_BATCH_SIZE):
        source_end = min(source_start + PCA_TRANSFORM_BATCH_SIZE, len(source))
        count = source_end - source_start
        destination[start : start + count] = source[source_start:source_end]
        start += count
    return start


def _project_vectors(pca: PCA, vectors: np.ndarray) -> np.ndarray:
    coordinates = np.empty((len(vectors), 3), dtype=np.float32)
    for start in range(0, len(vectors), PCA_TRANSFORM_BATCH_SIZE):
        end = min(start + PCA_TRANSFORM_BATCH_SIZE, len(vectors))
        coordinates[start:end] = pca.transform(
            vectors[start:end].astype(np.float32, copy=False)
        )
    return coordinates


def fit_and_project_joint_pca(
    vocabulary_vectors: np.ndarray,
    image_vectors: np.ndarray,
    latent_vectors: np.ndarray,
    temporary_dir: Path,
) -> tuple[PCA, list[np.ndarray]]:
    """Fit one PCA over all three sources and project each source separately."""
    vector_sources = [vocabulary_vectors, image_vectors, latent_vectors]
    hidden_sizes = {int(vectors.shape[1]) for vectors in vector_sources}
    if len(hidden_sizes) != 1:
        raise ValueError(f"Embedding hidden sizes do not match: {hidden_sizes}")
    total_points = sum(len(vectors) for vectors in vector_sources)
    if total_points < 3:
        raise ValueError("Fewer than three vectors are available for PCA.")

    hidden_size = hidden_sizes.pop()
    fit_path = temporary_dir / "joint_pca_fit.float32.mmap"
    fit_matrix = np.memmap(
        fit_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_points, hidden_size),
    )
    try:
        destination_start = 0
        for vectors in vector_sources:
            destination_start = _copy_float32_blocks(
                fit_matrix, destination_start, vectors
            )
        fit_matrix.flush()
        pca = PCA(
            n_components=3,
            svd_solver="randomized",
            random_state=RANDOM_SEED,
            copy=False,
        )
        pca.fit(fit_matrix)
    finally:
        del fit_matrix
        gc.collect()
        if fit_path.exists():
            fit_path.unlink()

    projected = [_project_vectors(pca, vectors) for vectors in vector_sources]
    return pca, projected


def assemble_joint_points(
    projected: list[np.ndarray], metadata: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    vocabulary_coordinates, image_coordinates, latent_coordinates = projected
    vocabulary_count = len(vocabulary_coordinates)
    image_count = len(image_coordinates)
    latent_count = len(latent_coordinates)
    return {
        "coordinates": np.concatenate(projected, axis=0),
        "kind_codes": np.concatenate(
            (
                np.zeros(vocabulary_count, dtype=np.uint8),
                np.ones(image_count, dtype=np.uint8),
                np.full(latent_count, 2, dtype=np.uint8),
            )
        ),
        "token_ids": np.concatenate(
            (
                np.arange(vocabulary_count, dtype=np.int32),
                np.full(image_count + latent_count, -1, dtype=np.int32),
            )
        ),
        "sample_ordinal": np.concatenate(
            (
                np.full(vocabulary_count, -1, dtype=np.int32),
                metadata["image_sample_ordinal"],
                metadata["latent_sample_ordinal"],
            )
        ),
        "sequence_positions": np.concatenate(
            (
                np.full(vocabulary_count, -1, dtype=np.int32),
                metadata["image_sequence_positions"],
                metadata["latent_sequence_positions"],
            )
        ),
        "generation_steps": np.concatenate(
            (
                np.full(vocabulary_count, -1, dtype=np.int32),
                metadata["image_generation_steps"],
                metadata["latent_generation_steps"],
            )
        ),
        "latent_indices": np.concatenate(
            (
                np.full(vocabulary_count + image_count, -1, dtype=np.int32),
                metadata["latent_indices"],
            )
        ),
        "trajectory_steps": np.concatenate(
            (
                np.full(vocabulary_count + image_count, -1, dtype=np.int32),
                metadata["latent_trajectory_steps"],
            )
        ),
    }


def save_pca_archives(
    output_path: Path,
    points: dict[str, np.ndarray],
    latent_coordinates: np.ndarray,
    metadata: dict[str, np.ndarray],
    pca: PCA,
    sample_records: list[dict[str, Any]],
) -> None:
    dataset_indices = np.asarray(
        [str(record["dataset_index"]) for record in sample_records], dtype=np.str_
    )
    request_ids = np.asarray(
        [record["request_id"] for record in sample_records], dtype=np.str_
    )
    np.savez_compressed(
        output_path / JOINT_PCA_FILE,
        **points,
        kind_names=KIND_NAMES,
        dataset_indices=dataset_indices,
        request_ids=request_ids,
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    np.savez_compressed(
        output_path / LATENT_TRAJECTORY_FILE,
        coordinates=latent_coordinates,
        sample_ordinal=metadata["latent_sample_ordinal"],
        sequence_positions=metadata["latent_sequence_positions"],
        generation_steps=metadata["latent_generation_steps"],
        latent_indices=metadata["latent_indices"],
        trajectory_steps=metadata["latent_trajectory_steps"],
        sample_offsets=metadata["latent_sample_offsets"],
        dataset_indices=dataset_indices,
        request_ids=request_ids,
    )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_default(value: Any) -> Any:
    converted = _json_compatible(value)
    if converted is value:
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )
    return converted


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
            )


def global_config_snapshot() -> dict[str, Any]:
    names = [
        "MODEL_PATH",
        "HRBENCH_PATH",
        "OUTPUT_DIR",
        "JOINT_PCA_FILE",
        "LATENT_TRAJECTORY_FILE",
        "RESULTS_FILE",
        "RUN_CONFIG_FILE",
        "SELECTION_MODE",
        "START_INDEX",
        "NUM_SAMPLES",
        "RANDOM_SEED",
        "DEVICE",
        "TORCH_DTYPE",
        "TRUST_REMOTE_CODE",
        "MIN_PIXELS",
        "MAX_PIXELS",
        "MAX_NEW_TOKENS",
        "USE_CACHE",
        "VOCAB_EMBEDDING_BATCH_SIZE",
        "PCA_TRANSFORM_BATCH_SIZE",
    ]
    return {name: globals()[name] for name in names}


def main() -> None:
    model_path, dataset_path, output_path = validate_configuration()
    rows, selected_indices = load_hrbench_rows(dataset_path)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".lavit_capture_", dir=output_path))
    succeeded = False
    try:
        model, processor, model_details = load_model_and_processor(model_path)
        vocabulary_vectors, vocabulary_export = export_vocabulary_embeddings(
            model, temporary_dir
        )
        capture_paths, sample_records, inference_statistics = run_inference(
            model,
            processor,
            rows,
            selected_indices,
            dataset_path.parent,
            temporary_dir,
            model_details,
        )
        image_vectors, latent_vectors, metadata, capture_statistics = (
            extract_sampled_images_and_latents(
                capture_paths,
                target_image_count=len(vocabulary_vectors),
                hidden_size=int(vocabulary_vectors.shape[1]),
                temporary_dir=temporary_dir,
            )
        )
        pca, projected = fit_and_project_joint_pca(
            vocabulary_vectors, image_vectors, latent_vectors, temporary_dir
        )
        points = assemble_joint_points(projected, metadata)
        save_pca_archives(
            output_path,
            points,
            projected[2],
            metadata,
            pca,
            sample_records,
        )
        write_jsonl(output_path / RESULTS_FILE, sample_records)

        run_config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": global_config_snapshot(),
            "selected_dataset_ordinals": selected_indices,
            "model_capture": model_details,
            "vocabulary_export": vocabulary_export,
            "inference_statistics": inference_statistics,
            "capture_statistics": capture_statistics,
            "pca_input_counts": {
                "vocabulary_embedding": int(len(vocabulary_vectors)),
                "image_feature": int(len(image_vectors)),
                "latent": int(len(latent_vectors)),
            },
            "pca_explained_variance_ratio": (pca.explained_variance_ratio_.tolist()),
            "total_projected_points": int(len(points["coordinates"])),
            "outputs": {
                "joint_pca": JOINT_PCA_FILE,
                "latent_trajectories": LATENT_TRAJECTORY_FILE,
                "results": RESULTS_FILE,
            },
            "note": (
                "Each trajectory contains final-normalized decoder hidden states "
                "for generated <lvr*> tokens that were actually consumed by a "
                "subsequent generation forward pass. The final sampled token is "
                "reported as unconsumed and is never assigned a fabricated state."
            ),
        }
        (output_path / RUN_CONFIG_FILE).write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        # Explicitly close NumPy memmaps before removing the temporary capture
        # directory. This is required on Windows and harmless on Linux.
        del vocabulary_vectors, image_vectors, latent_vectors, projected, points
        gc.collect()
        succeeded = True
        print(f"Analysis complete: {output_path}")
        print(f"Joint PCA: {output_path / JOINT_PCA_FILE}")
        print(f"Latent trajectories: {output_path / LATENT_TRAJECTORY_FILE}")
    finally:
        if succeeded or not KEEP_TEMP_CAPTURE_ON_ERROR:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        elif temporary_dir.exists():
            print(f"Temporary captures kept for debugging: {temporary_dir}")
        gc.collect()


if __name__ == "__main__":
    main()
