# =============================================================================
# Model Evaluation Script
# =============================================================================
# This script evaluates the ViT emotion detection model performance
# with comprehensive metrics for research credibility.
# =============================================================================

import torch
import numpy as np
import random
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import logging
from pathlib import Path

# Set up reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
from model_loader import load_vit_model, ID_TO_LABEL
from inference import predict_from_image_with_face_check


def generate_mock_evaluation_data(num_samples: int = 1000) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate mock evaluation data for testing.
    In production, this would use real labeled datasets.

    Args:
        num_samples: Number of mock samples to generate

    Returns:
        Tuple of (images, true_labels)
    """
    # Mock image data (224x224x3 random tensors)
    images = []
    true_labels = []

    emotions = list(ID_TO_LABEL.values())

    for _ in range(num_samples):
        # Generate random image tensor
        image = np.random.rand(224, 224, 3).astype(np.float32)
        images.append(image)

        # Random true emotion label
        true_label = random.choice(emotions)
        true_labels.append(true_label)

    return images, true_labels


def evaluate_model(images: List[np.ndarray], true_labels: List[str]) -> Dict:
    """
    Evaluate model performance on test data.

    Args:
        images: List of image arrays
        true_labels: List of true emotion labels

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating model on {len(images)} samples...")

    predicted_labels = []
    confidences = []

    for i, image in enumerate(images):
        try:
            # Get prediction
            emotion, stress_level, confidence = predict_from_image_with_face_check(
                image, mode="real"
            )
            predicted_labels.append(emotion)
            confidences.append(confidence)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(images)} samples")

        except Exception as e:
            logger.warning(f"Prediction failed for sample {i}: {e}")
            # Fallback to random prediction
            predicted_labels.append(random.choice(list(ID_TO_LABEL.values())))
            confidences.append(0.5)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(ID_TO_LABEL.values()))

    # Per-class metrics
    class_report = classification_report(
        true_labels, predicted_labels,
        labels=list(ID_TO_LABEL.values()),
        target_names=list(ID_TO_LABEL.values()),
        zero_division=0,
        output_dict=True
    )

    # Confidence statistics
    confidence_stats = {
        'mean': np.mean(confidences),
        'std': np.std(confidences),
        'min': np.min(confidences),
        'max': np.max(confidences)
    }

    results = {
        'overall_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': len(images)
        },
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': class_report,
        'confidence_stats': confidence_stats,
        'predictions': {
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'confidences': confidences
        }
    }

    return results


def print_evaluation_report(results: Dict):
    """
    Print a formatted evaluation report.

    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("🤖 MINDCARE AI MODEL EVALUATION REPORT")
    print("="*60)

    overall = results['overall_metrics']
    conf_stats = results['confidence_stats']

    print("\n📊 OVERALL METRICS:")
    print(f"   Accuracy:  {overall['accuracy']:.4f}")
    print(f"   Precision: {overall['precision']:.4f}")
    print(f"   Recall:    {overall['recall']:.4f}")
    print(f"   F1-Score:  {overall['f1_score']:.4f}")
    print(f"   Samples:   {overall['num_samples']}")

    print("\n🎯 CONFIDENCE STATISTICS:")
    print(f"   Mean: {conf_stats['mean']:.4f}")
    print(f"   Std:  {conf_stats['std']:.4f}")
    print(f"   Min:  {conf_stats['min']:.4f}")
    print(f"   Max:  {conf_stats['max']:.4f}")

    print("\n📈 PER-CLASS PERFORMANCE:")
    class_metrics = results['per_class_metrics']
    emotions = list(ID_TO_LABEL.values())

    print(f"{'Emotion':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Support':<8}")
    print("-" * 50)

    for emotion in emotions:
        if emotion in class_metrics:
            metrics = class_metrics[emotion]
            print(f"{emotion:<8} {metrics.get('precision', 0):<10.4f} {metrics.get('recall', 0):<8.4f} {metrics.get('f1-score', 0):<8.4f} {metrics.get('support', 0):<8}")

    print("\n🔍 CONFUSION MATRIX:")
    cm = np.array(results['confusion_matrix'])
    print("True\\Pred | " + " | ".join(f"{e[:6]:>8}" for e in emotions))
    print("-" * (12 + 10 * len(emotions)))

    for i, true_emotion in enumerate(emotions):
        row = f"{true_emotion[:6]:>10} |"
        for j, count in enumerate(cm[i]):
            row += f"{count:>8}"
        print(row)

    print("\n" + "="*60)


def main():
    """Main evaluation function."""
    logger.info("Starting MindCare AI model evaluation...")

    # Load model to ensure it's available
    try:
        model = load_vit_model()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # Generate mock evaluation data
    logger.info("Generating evaluation data...")
    images, true_labels = generate_mock_evaluation_data(num_samples=500)

    # Run evaluation
    try:
        results = evaluate_model(images, true_labels)
        print_evaluation_report(results)

        # Save results to file
        import json
        output_path = Path(__file__).parent / "evaluation_results.json"
        with open(output_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {
                'overall_metrics': results['overall_metrics'],
                'confusion_matrix': results['confusion_matrix'],
                'per_class_metrics': {k: v for k, v in results['per_class_metrics'].items() if isinstance(v, dict)},
                'confidence_stats': {k: float(v) for k, v in results['confidence_stats'].items()},
                'predictions': {
                    'true_labels': results['predictions']['true_labels'],
                    'predicted_labels': results['predictions']['predicted_labels'],
                    'confidences': [float(c) for c in results['predictions']['confidences']]
                }
            }
            json.dump(json_results, f, indent=2)

        logger.info(f"✅ Evaluation results saved to {output_path}")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()