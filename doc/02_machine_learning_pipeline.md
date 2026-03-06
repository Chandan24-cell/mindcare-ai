# Machine Learning Pipeline

The ML pipeline processes raw user input and converts it into
emotion and stress predictions.

---

## Pipeline Stages

1 Input Acquisition  
2 Face Detection  
3 Image Preprocessing  
4 Model Inference  
5 Emotion Classification  
6 Stress Mapping  
7 Response Generation

---

## Step 1: Input Acquisition

The system receives input from:

• Live camera feed
• Uploaded images
• Manual mood data
• Sensor metrics

---

## Step 2: Face Detection

Before inference the system validates the presence of a human face.

Multiple detectors are used for robustness:

RetinaFace  
MediaPipe  
OpenCV Haar Cascade

If no face is detected the system stops prediction.

---

## Step 3: Preprocessing

Images are converted into a format compatible with the ViT model.

Steps include:

• Resize to 224x224
• RGB normalization
• Tensor conversion

---

## Step 4: Model Inference

The Vision Transformer analyzes the face and predicts emotion.

Output:
Probability distribution across emotion classes.

---

## Step 5: Emotion Classification

The highest probability class becomes the predicted emotion.

Example:

Happy  
Sad  
Neutral  
Angry  
Fear  
Disgust  
Surprise

---

## Step 6: Stress Mapping

Emotion is mapped to stress level using a rule-based system.

Example:

Happy → Low Stress  
Angry → High Stress

---

## Step 7: Response Generation

The API returns:

Emotion  
Stress Level  
Confidence Score  
Suggestion