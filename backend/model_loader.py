import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import ViTForImageClassification

MODEL_FILENAME = "vit_small_emotion.pth"
BASE_MODEL_NAME = "google/vit-base-patch16-224"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / MODEL_FILENAME
ID_TO_LABEL = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class ModelUnavailableError(RuntimeError):
    """Raised when real image inference assets are not available."""


def _candidate_model_paths() -> list[Path]:
    env_path = os.getenv("MINDCARE_MODEL_PATH")
    candidates: list[Path] = []

    if env_path:
        resolved_env_path = Path(env_path).expanduser()
        if resolved_env_path.is_dir():
            candidates.append(resolved_env_path / MODEL_FILENAME)
        else:
            candidates.append(resolved_env_path)

    candidates.extend(
        [
            DEFAULT_MODEL_PATH,
            PROJECT_ROOT / "models" / MODEL_FILENAME,
            PROJECT_ROOT / MODEL_FILENAME,
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve(strict=False)
        if resolved_candidate not in seen:
            seen.add(resolved_candidate)
            unique_candidates.append(resolved_candidate)

    return unique_candidates


def resolve_model_path() -> Path:
    for candidate in _candidate_model_paths():
        if candidate.is_file():
            return candidate

    checked_paths = "\n".join(f"- {path}" for path in _candidate_model_paths())
    raise ModelUnavailableError(
        "Real image model weights are missing. "
        f"Add `{MODEL_FILENAME}` to `backend/models/` or set `MINDCARE_MODEL_PATH`.\n"
        f"Checked:\n{checked_paths}"
    )


def get_model_status() -> dict[str, str | bool]:
    try:
        model_path = resolve_model_path()
    except ModelUnavailableError as exc:
        return {
            "available": False,
            "path": str(DEFAULT_MODEL_PATH),
            "message": str(exc),
        }

    return {
        "available": True,
        "path": str(model_path),
        "message": "Real image model is available.",
    }


@lru_cache(maxsize=1)
def load_vit_model():
    print("Loading trained ViT model...")
    model_path = resolve_model_path()

    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=7,
        ignore_mismatched_sizes=True,
    )

    print("Loading trained weights from:", model_path)

    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.config.id2label = ID_TO_LABEL
    model.config.label2id = {label: idx for idx, label in ID_TO_LABEL.items()}
    model.to(device)
    model.eval()

    print("✅ Trained model loaded successfully")

    return model
