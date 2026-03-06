# =============================================================================
# ML Model Loader Module
# =============================================================================
# This module handles lazy loading of the Vision Transformer (ViT) model.
# It implements the singleton pattern to ensure the model is loaded only once.
# 
# Why Lazy Loading in ML Systems:
# - Reduces startup time for non-ML endpoints
# - Saves memory when model isn't needed
# - Allows server to handle requests before model loads
# - Essential for serverless/containerized deployments
# =============================================================================

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from typing import Tuple, Optional
import os

# Global model instances (singleton pattern)
_vit_model: Optional[ViTForImageClassification] = None
_vit_processor: Optional[ViTImageProcessor] = None
_device: Optional[torch.device] = None


def get_device() -> torch.device:
    """
    Get the compute device (CUDA or CPU).
    
    Returns:
        torch.device: The appropriate device for model inference
    """
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_vit_model() -> Tuple[ViTForImageClassification, ViTImageProcessor]:
    """
    Load the Vision Transformer model and processor.
    Uses lazy loading - model is loaded only on first call.
    
    The function:
    1. Loads the base ViT model from HuggingFace or local cache
    2. Loads custom trained weights from vit_small_emotion.pth
    3. Configures the emotion labels
    4. Moves model to appropriate device (CPU/GPU)
    
    Returns:
        Tuple containing:
            - ViTForImageClassification: The loaded model
            - ViTImageProcessor: The image preprocessor
    """
    global _vit_model, _vit_processor
    
    # Return cached model if already loaded
    if _vit_model is not None:
        return _vit_model, _vit_processor
    
    # Get compute device
    device = get_device()
    print(f"Loading ViT model on device: {device}")
    
    # -----------------------------------------------------------------------------
    # Step 1: Load the image processor
    # -----------------------------------------------------------------------------
    # The processor handles image preprocessing (resize, normalize, tokenize)
    try:
        print("Attempting to load processor from local cache...")
        _vit_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224",
            local_files_only=False  # Allow download if not cached
        )
        print("✅ Processor loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading processor: {e}")
        raise
    
    # -----------------------------------------------------------------------------
    # Step 2: Load the base Vision Transformer model
    # -----------------------------------------------------------------------------
    # We use vit-base-patch16-224 with 7 output classes for emotion detection
    try:
        print("Attempting to load model from local files...")
        _vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=7,  # Number of emotion classes
            ignore_mismatched_sizes=True,  # Handle size mismatches
            local_files_only=True  # Try local first
        )
        print("✅ Base model loaded from local cache")
    except Exception as e:
        print(f"⚠️ Local model not found, downloading from HuggingFace: {e}")
        _vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=7,
            ignore_mismatched_sizes=True,
            local_files_only=False  # Download if needed
        )
        print("✅ Base model downloaded from HuggingFace")
    
    # -----------------------------------------------------------------------------
    # Step 3: Load custom trained weights
    # -----------------------------------------------------------------------------
    # Override base model weights with our trained emotion detection weights
    model_path = "vit_small_emotion.pth"
    
    # Check if we're in the backend subdirectory
    if not os.path.exists(model_path):
        # Try parent directory (when running from backend/)
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vit_small_emotion.pth")
    
    try:
        print(f"Loading trained weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        _vit_model.load_state_dict(state_dict)
        print("✅ Trained weights loaded successfully")
    except FileNotFoundError:
        print(f"⚠️ {model_path} not found - using base model weights")
        print("   (For production, ensure trained weights are available)")
    except Exception as e:
        print(f"⚠️ Could not load trained weights: {e}")
        print("   Using base model weights instead")
    
    # -----------------------------------------------------------------------------
    # Step 4: Configure model and move to device
    # -----------------------------------------------------------------------------
    # Set up the emotion label mapping (class index -> label name)
    _vit_model.config.id2label = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"
    }
    
    # Move model to compute device and set to evaluation mode
    _vit_model.to(device)
    _vit_model.eval()
    
    print("✅ Model ready for inference")
    
    return _vit_model, _vit_processor


def get_model() -> Tuple[ViTForImageClassification, ViTImageProcessor]:
    """
    Public accessor for the model.
    Ensures model is loaded before returning.
    
    Returns:
        Tuple containing the model and processor
    """
    if _vit_model is None:
        load_vit_model()
    return _vit_model, _vit_processor

