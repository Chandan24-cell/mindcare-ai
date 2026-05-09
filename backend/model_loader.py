import gc
import logging
import os
import sys
import threading
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

# Set up logging
logger = logging.getLogger(__name__)


MODEL_FILENAME = "vit_small_emotion.pth"
QUANTIZED_MODEL_FILENAME = "vit_small_emotion_int8.pt"
RENDER_FREE_MEMORY_LIMIT_MB = 512

warnings.filterwarnings(
    "ignore",
    message="Currently, qnnpack incorrectly ignores reduce_range.*",
    category=UserWarning,
)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch only allows this to be set before parallel work starts.
    logger.debug("Torch interop threads were already initialized.")
torch.set_grad_enabled(False)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Correct backend/models path
DEFAULT_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / MODEL_FILENAME
DEFAULT_QUANTIZED_MODEL_PATH = (
    PROJECT_ROOT / "backend" / "models" / QUANTIZED_MODEL_FILENAME
)

# Optional environment override
MODEL_PATH = Path(
    os.getenv("MINDCARE_MODEL_PATH", str(DEFAULT_MODEL_PATH))
)

logger.info(f"Using model path: {MODEL_PATH}")
logger.info(f"Model exists: {MODEL_PATH.exists()}")


ID_TO_LABEL = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


device = torch.device("cpu")

_MODEL_LOCK = threading.Lock()
_MODEL: torch.nn.Module | None = None
_MODEL_LOAD_ERROR: str | None = None
_LOADED_MODEL_PATH: Path | None = None


class ModelUnavailableError(RuntimeError):
    """Raised when real image inference assets are not available."""


class MemoryBudgetExceededError(RuntimeError):
    """Raised when the process is too close to the Render memory limit."""


def _is_render_environment() -> bool:
    return any(
        os.getenv(name)
        for name in (
            "RENDER",
            "RENDER_SERVICE_ID",
            "RENDER_EXTERNAL_HOSTNAME",
            "RENDER_GIT_COMMIT",
        )
    )


def get_memory_limit_mb() -> float | None:
    configured_limit = os.getenv("MINDCARE_MEMORY_LIMIT_MB")
    if configured_limit:
        try:
            return float(configured_limit)
        except ValueError:
            logger.warning(
                "Invalid MINDCARE_MEMORY_LIMIT_MB=%r; ignoring.",
                configured_limit,
            )

    if _is_render_environment():
        return float(RENDER_FREE_MEMORY_LIMIT_MB)

    return None


def get_memory_usage_mb() -> float | None:
    status_path = Path("/proc/self/status")
    try:
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass

    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except (OSError, ValueError):
        return None


def release_unused_memory() -> None:
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            logger.debug("malloc_trim unavailable", exc_info=True)


def get_memory_report() -> dict[str, float | bool | None]:
    usage_mb = get_memory_usage_mb()
    limit_mb = get_memory_limit_mb()
    exceeded = (
        usage_mb is not None
        and limit_mb is not None
        and usage_mb >= limit_mb
    )

    return {
        "rss_mb": round(usage_mb, 1) if usage_mb is not None else None,
        "limit_mb": limit_mb,
        "exceeded": exceeded,
    }


def log_memory_usage(stage: str) -> None:
    report = get_memory_report()
    usage_mb = report["rss_mb"]
    limit_mb = report["limit_mb"]

    if usage_mb is None:
        logger.info("Memory %s: unavailable", stage)
        return

    if limit_mb is None:
        logger.info("Memory %s: rss=%s MB", stage, usage_mb)
        return

    percent = (float(usage_mb) / float(limit_mb)) * 100.0
    logger.info(
        "Memory %s: rss=%s MB / %.0f MB (%.1f%%)",
        stage,
        usage_mb,
        limit_mb,
        percent,
    )
    if percent >= 90.0:
        logger.warning(
            "Memory %s is close to the configured limit: %.1f%% used",
            stage,
            percent,
        )


def ensure_memory_within_limit(stage: str, hard_fraction: float = 0.98) -> None:
    usage_mb = get_memory_usage_mb()
    limit_mb = get_memory_limit_mb()

    if usage_mb is None or limit_mb is None:
        return

    threshold_mb = limit_mb * hard_fraction
    if usage_mb >= threshold_mb:
        raise MemoryBudgetExceededError(
            "Server memory is too close to the Render free tier limit "
            f"before {stage}: {usage_mb:.1f} MB used of {limit_mb:.0f} MB. "
            "Please retry in a moment."
        )


def is_out_of_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, (MemoryError, MemoryBudgetExceededError)):
        return True

    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in (
            "out of memory",
            "cannot allocate memory",
            "defaultcpuallocator",
            "memoryerror",
        )
    )


def _candidate_model_paths() -> list[Path]:
    env_path = os.getenv("MINDCARE_MODEL_PATH")

    candidates: list[Path] = []

    # Environment variable override
    if env_path:
        resolved_env_path = Path(env_path).expanduser()

        if resolved_env_path.is_dir():
            candidates.append(resolved_env_path / MODEL_FILENAME)
        else:
            candidates.append(resolved_env_path)

    # Default fallback paths
    candidates.extend(
        [
            MODEL_PATH,
            DEFAULT_MODEL_PATH,
            PROJECT_ROOT / "models" / MODEL_FILENAME,
            PROJECT_ROOT / MODEL_FILENAME,
        ]
    )

    # Remove duplicates
    unique_candidates: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        resolved_candidate = candidate.resolve(strict=False)

        if resolved_candidate not in seen:
            seen.add(resolved_candidate)
            unique_candidates.append(resolved_candidate)

    return unique_candidates


def _candidate_quantized_model_paths() -> list[Path]:
    env_path = os.getenv("MINDCARE_QUANTIZED_MODEL_PATH")

    candidates: list[Path] = []
    if env_path:
        resolved_env_path = Path(env_path).expanduser()

        if resolved_env_path.is_dir():
            candidates.append(resolved_env_path / QUANTIZED_MODEL_FILENAME)
        else:
            candidates.append(resolved_env_path)

    candidates.append(DEFAULT_QUANTIZED_MODEL_PATH)

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
        logger.debug(f"Checking model path: {candidate}")

        if candidate.is_file():
            logger.info(f"✅ Found model at: {candidate}")
            return candidate

    checked_paths = "\n".join(f"- {path}" for path in _candidate_model_paths())

    raise ModelUnavailableError(
        "Real image model weights are missing.\n"
        f"Add `{MODEL_FILENAME}` to `backend/models/` "
        f"or set `MINDCARE_MODEL_PATH`.\n\n"
        f"Checked:\n{checked_paths}"
    )


def resolve_quantized_model_path() -> Path | None:
    for candidate in _candidate_quantized_model_paths():
        logger.debug("Checking optimized model path: %s", candidate)
        if candidate.is_file():
            logger.info("Found optimized int8 model at: %s", candidate)
            return candidate

    return None


def get_model_status() -> dict[str, str | bool]:
    if _MODEL is not None:
        return {
            "available": True,
            "loaded": True,
            "path": str(_LOADED_MODEL_PATH or resolve_model_path()),
            "message": "Real image model is loaded and available.",
        }

    if _MODEL_LOAD_ERROR:
        return {
            "available": False,
            "loaded": False,
            "path": str(DEFAULT_MODEL_PATH),
            "message": _MODEL_LOAD_ERROR,
        }

    try:
        model_path = resolve_quantized_model_path() or resolve_model_path()

    except ModelUnavailableError as exc:
        return {
            "available": False,
            "loaded": False,
            "path": str(DEFAULT_MODEL_PATH),
            "message": str(exc),
        }

    return {
        "available": True,
        "loaded": False,
        "path": str(model_path),
        "message": "Real image model weights are available.",
    }


def _load_state_dict(model_path: Path) -> dict[str, Any]:
    try:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise ModelUnavailableError(
            f"Unsupported checkpoint format in `{model_path}`."
        )

    return state_dict


def _build_model_shell() -> torch.nn.Module:
    from transformers import ViTConfig, ViTForImageClassification

    label_to_id = {label: idx for idx, label in ID_TO_LABEL.items()}
    config = ViTConfig(
        num_labels=7,
        id2label=ID_TO_LABEL,
        label2id=label_to_id,
    )

    try:
        with torch.device("meta"):
            return ViTForImageClassification(config)
    except Exception as exc:
        logger.warning(
            "Meta-device model initialization failed; falling back to CPU init: %s",
            exc,
        )
        return ViTForImageClassification(config)


def _configure_quantized_engine() -> None:
    supported_engines = torch.backends.quantized.supported_engines
    for engine in ("qnnpack", "fbgemm", "x86"):
        if engine in supported_engines:
            torch.backends.quantized.engine = engine
            logger.info("Using torch quantized engine: %s", engine)
            return

    logger.warning("No preferred torch quantized engine available: %s", supported_engines)


def _load_quantized_vit_model(model_path: Path) -> torch.nn.Module:
    logger.info("Loading optimized int8 ViT model on CPU from: %s", model_path)
    _configure_quantized_engine()
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def _load_fp32_vit_model() -> torch.nn.Module:
    logger.info("Loading fp32 trained ViT model on CPU...")
    log_memory_usage("before model load")

    model_path = resolve_model_path()
    logger.info("Loading trained weights from: %s", model_path)

    model = _build_model_shell()
    state_dict = _load_state_dict(model_path)

    try:
        model.load_state_dict(state_dict, assign=True)
    except TypeError:
        if next(model.parameters()).is_meta:
            model = model.to_empty(device=device)
        model.load_state_dict(state_dict)
    finally:
        del state_dict
        release_unused_memory()

    model.to(device)
    model.eval()

    release_unused_memory()
    log_memory_usage("after model load")
    logger.info("Trained ViT model loaded successfully")

    return model


def _load_vit_model_once() -> tuple[torch.nn.Module, Path]:
    log_memory_usage("before model load")
    quantized_model_path = resolve_quantized_model_path()
    if quantized_model_path is not None:
        model = _load_quantized_vit_model(quantized_model_path)
        release_unused_memory()
        log_memory_usage("after optimized model load")
        try:
            ensure_memory_within_limit("optimized model load", hard_fraction=0.96)
        except MemoryBudgetExceededError:
            del model
            release_unused_memory()
            raise
        logger.info("Optimized int8 ViT model loaded successfully")
        return model, quantized_model_path

    model = _load_fp32_vit_model()
    try:
        ensure_memory_within_limit("fp32 model load", hard_fraction=0.96)
    except MemoryBudgetExceededError:
        del model
        release_unused_memory()
        raise
    return model, resolve_model_path()


def load_vit_model() -> torch.nn.Module:
    global _MODEL, _MODEL_LOAD_ERROR, _LOADED_MODEL_PATH

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        if _MODEL_LOAD_ERROR is not None:
            raise ModelUnavailableError(_MODEL_LOAD_ERROR)

        try:
            _MODEL, _LOADED_MODEL_PATH = _load_vit_model_once()
        except MemoryBudgetExceededError:
            release_unused_memory()
            raise
        except Exception as exc:
            release_unused_memory()
            if is_out_of_memory_error(exc):
                _MODEL_LOAD_ERROR = (
                    "Real image model could not be loaded because the server "
                    "is out of memory."
                )
                raise MemoryBudgetExceededError(_MODEL_LOAD_ERROR) from exc

            _MODEL_LOAD_ERROR = f"Real image model could not be loaded: {exc}"
            raise

    return _MODEL
