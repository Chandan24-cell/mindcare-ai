# =============================================================================
# Face Detection Module
# =============================================================================
# This module handles face detection using a hybrid approach.
# 
# WHY FACE DETECTION IS REQUIRED:
# Our ViT emotion model was trained on face images. If we pass images without
# faces, the model will make random predictions - which is unreliable for
# medical-style applications.
#
# DETECTION PIPELINE (in order of priority):
# 1. RetinaFace - Most accurate, best for small faces
# 2. MediaPipe - Fast, good for real-time webcam
# 3. OpenCV Haar Cascade - Last fallback
#
# If no face is detected at any stage, we return an error instead of
# allowing the model to make unreliable predictions.
# =============================================================================

import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, NamedTuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# RETINAFACE DETECTION (Primary - Most Accurate)
# =============================================================================
# RetinaFace is the most accurate face detector but requires the retinaface package.
# It can detect very small faces and provides precise bounding boxes.
# =============================================================================

def detect_face_retinaface(image: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], int]]:
    """
    Detect face using RetinaFace.
    
    RetinaFace is a state-of-the-art face detector that performs
    face detection in a single shot. It can detect faces at various
    scales and is particularly good for small faces.
    
    Args:
        image: Image as numpy array (RGB format)
    
    Returns:
        Bounding box as (x1, y1, x2, y2) or None if no face found
    """
    try:
        # Import retinaface
        from retinaface import RetinaFace
        
        # Detect faces
        faces = RetinaFace.detect_faces(image)
        
        if faces is None or len(faces) == 0:
            logger.info("RetinaFace: No face detected")
            return None
        
        # Get the largest face (in case multiple faces detected)
        # This is important - we want the main subject
        largest_face = None
        largest_area = 0
        num_faces = len(faces)
        
        for face_key in faces:
            face = faces[face_key]
            facial_area = face['facial_area']
            
            # Calculate area
            x1, y1, x2, y2 = facial_area
            area = (x2 - x1) * (y2 - y1)
            
            if area > largest_area:
                largest_area = area
                largest_face = facial_area
        
        logger.info(f"RetinaFace: Detected face at {largest_face}")
        return tuple(largest_face), num_faces
        
    except ImportError:
        logger.warning("RetinaFace not installed, skipping")
        return None
    except Exception as e:
        logger.error(f"RetinaFace error: {e}")
        return None


# =============================================================================
# MEDIAPIPE DETECTION (Fallback - Fast)
# =============================================================================
# MediaPipe is Google's fast face detection solution, optimized for
# real-time applications like webcam processing.
# =============================================================================

def detect_face_mediapipe(image: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], int]]:
    """
    Detect face using MediaPipe.
    
    MediaPipe offers fast face detection optimized for real-time
    applications. It's less accurate than RetinaFace but much faster.
    
    Args:
        image: Image as numpy array (RGB format)
    
    Returns:
        Bounding box as (x1, y1, x2, y2) or None if no face found
    """
    try:
        import mediapipe as mp
        from mediapipe.python.solutions.face_detection import FaceDetection
        
        # Create face detector
        # model_selection=0 for short-range (webcam), 1 for long-range
        with FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            # Convert RGB to RGB (MediaPipe expects RGB)
            results = face_detection.process(image)
            
            if not results.detections:
                logger.info("MediaPipe: No face detected")
                return None
            
            # Get the first (most confident) detection
            detection = results.detections[0]
            
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            
            # Convert to image coordinates
            h, w, _ = image.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            
            logger.info(f"MediaPipe: Detected face at ({x1}, {y1}, {x2}, {y2})")
            return (x1, y1, x2, y2), len(results.detections)
            
    except ImportError:
        logger.warning("MediaPipe not installed, skipping")
        return None
    except Exception as e:
        logger.error(f"MediaPipe error: {e}")
        return None


# =============================================================================
# OPENCV HAAR CASCADE (Last Fallback)
# =============================================================================
# OpenCV's Haar Cascade is a classic face detection method.
# It's less accurate but always available with OpenCV.
# =============================================================================

def detect_face_haar(image: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], int]]:
    """
    Detect face using OpenCV Haar Cascade.
    
    Haar Cascade is a traditional computer vision approach.
    It's less accurate but works as a reliable last fallback.
    
    Args:
        image: Image as numpy array (RGB format)
    
    Returns:
        Bounding box as (x1, y1, x2, y2) or None if no face found
    """
    try:
        import cv2
        
        # Convert RGB to grayscale (Haar Cascade works on grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Load pre-trained Haar Cascade classifier
        # Using the frontal face default cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Decrease for more detections
            minNeighbors=5,         # Higher = fewer false positives
            minSize=(30, 30)       # Minimum face size
        )
        
        if len(faces) == 0:
            logger.info("Haar Cascade: No face detected")
            return None
        
        # Get the largest face
        largest_face = None
        largest_area = 0
        
        for (x, y, w, h) in faces:
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, x + w, y + h)
        
        logger.info(f"Haar Cascade: Detected face at {largest_face}")
        return largest_face, len(faces)
        
    except ImportError:
        logger.warning("OpenCV not installed, skipping")
        return None
    except Exception as e:
        logger.error(f"Haar Cascade error: {e}")
        return None


# =============================================================================
# MASTER FACE DETECTION FUNCTION
# =============================================================================
# This is the main function that tries all detection methods in order.
# It ensures we always get a face before allowing model prediction.
# =============================================================================

class FaceDetectionResult(NamedTuple):
    bbox: Tuple[int, int, int, int]
    num_faces: int
    detector: str


def detect_face(image: np.ndarray) -> Optional[FaceDetectionResult]:
    """
    Detect face using hybrid approach.
    
    This function tries multiple face detection methods in order
    of accuracy. If one method fails, it falls back to the next.
    
    Detection Order:
    1. RetinaFace (most accurate, best for small faces)
    2. MediaPipe (fast, good for webcam)
    3. OpenCV Haar Cascade (last fallback)
    
    WHY THIS APPROACH:
    - Different methods work better in different conditions
    - Small faces need RetinaFace
    - Webcam needs MediaPipe speed
    - Some images only work with Haar
    
    Args:
        image: Image as numpy array (RGB format)
    
    Returns:
        Bounding box as (x1, y1, x2, y2) or None if no face found
    
    Example:
        >>> img = np.array(Image.open("photo.jpg"))
        >>> bbox = detect_face(img)
        >>> if bbox:
        ...     x1, y1, x2, y2 = bbox
        ...     print(f"Face found at ({x1}, {y1}) to ({x2}, {y2})")
    """
    # Try each detection method in order of accuracy
    
    # Step 1: Try RetinaFace (most accurate)
    logger.info("Attempting face detection with RetinaFace...")
    result = detect_face_retinaface(image)
    if result is not None:
        bbox, count = result
        return FaceDetectionResult(bbox=bbox, num_faces=count, detector="retinaface")
    
    # Step 2: Try MediaPipe (fast)
    logger.info("Attempting face detection with MediaPipe...")
    result = detect_face_mediapipe(image)
    if result is not None:
        bbox, count = result
        return FaceDetectionResult(bbox=bbox, num_faces=count, detector="mediapipe")
    
    # Step 3: Try Haar Cascade (last fallback)
    logger.info("Attempting face detection with Haar Cascade...")
    result = detect_face_haar(image)
    if result is not None:
        bbox, count = result
        return FaceDetectionResult(bbox=bbox, num_faces=count, detector="haar")
    
    # All methods failed
    logger.warning("Face detection failed: No face found with any method")
    return None


# =============================================================================
# FACE CROP AND PREPROCESSING
# =============================================================================

def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int], 
              margin: float = 0.2) -> np.ndarray:
    """
    Crop the face from the image with a margin.
    
    We add a margin around the detected face because:
    1. The emotion model was trained on face + some context
    2. It helps the model see the full face including hair, ears
    3. It prevents cutting off important facial features
    
    Args:
        image: Original image as numpy array
        bbox: Bounding box (x1, y1, x2, y2)
        margin: Margin as fraction of face size (default 20%)
    
    Returns:
        Cropped face image
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate face dimensions
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Add margin
    margin_x = int(face_width * margin)
    margin_y = int(face_height * margin)
    
    # Apply margin with bounds checking
    h, w, _ = image.shape
    
    x1_new = max(0, x1 - margin_x)
    y1_new = max(0, y1 - margin_y)
    x2_new = min(w, x2 + margin_x)
    y2_new = min(h, y2 + margin_y)
    
    # Crop
    cropped = image[y1_new:y2_new, x1_new:x2_new]
    
    logger.info(f"Cropped face from ({x1}, {y1}, {x2}, {y2}) to ({x1_new}, {y1_new}, {x2_new}, {y2_new})")
    
    return cropped


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_and_crop_face(pil_image: Image.Image) -> Tuple[Optional[Image.Image], Optional[FaceDetectionResult]]:
    """
    Detect and crop face from PIL Image.
    
    This is the main entry point used by the inference pipeline.
    It handles the entire process of detecting and cropping a face.
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        Tuple of (cropped_image, success)
        - If success=True: cropped_image contains the face
        - If success=False: cropped_image is None
    """
    # Convert PIL to numpy
    image_array = np.array(pil_image)
    
    detection = detect_face(image_array)
    
    if detection is None:
        return None, None
    
    # Crop face
    cropped_array = crop_face(image_array, detection.bbox)
    
    # Convert back to PIL
    cropped_image = Image.fromarray(cropped_array)
    
    return cropped_image, detection


# =============================================================================
# ERROR CLASSES
# =============================================================================

class NoFaceDetectedError(Exception):
    """
    Exception raised when no face is detected in the image.
    
    This exception is raised when all face detection methods fail.
    Instead of making unreliable predictions, we stop and inform
    the user to show their face.
    """
    def __init__(self, message: str = "No face detected. Please show your face to the camera."):
        self.message = message
        super().__init__(self.message)


class MultipleFacesDetectedError(Exception):
    """Raised when more than one face is visible and single-subject is required."""
    def __init__(self, message: str = "Multiple faces detected. Please show only one face."):
        self.message = message
        super().__init__(self.message)


# =============================================================================
# TEST FUNCTION
# =============================================================================

if __name__ == "__main__":
    # Simple test
    print("Testing face detection module...")
    
    # Create a simple test image (random noise)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Try detection
    result = detect_face(test_image)
    
    if result is None:
        print("✓ No face detected in blank image (expected)")
    else:
        print(f"✗ Face detected unexpectedly: {result}")
